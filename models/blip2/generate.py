from typing import Optional
import torch
from transformers import Blip2Processor
from transformers import Blip2ForConditionalGeneration
from transformers import BitsAndBytesConfig
from relevancy_utils import *


class Blip2ForAnalysis():

    def __init__(self, args):
        if args.load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif args.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quant_config = None
        self.model = Blip2withGradient.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            device_map="auto")
        self.args = args
        self.processor = Blip2Processor.from_pretrained(args.model_name)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.model.num_img_patches = (self.model.vision_model.config.image_size // self.model.vision_model.config.patch_size)**2
        self.model.num_img_tokens = self.model.config.num_query_tokens
        self.model.num_llm_layers = self.model.language_model.config.num_hidden_layers
        self.model.lm_head = self.model.language_model.lm_head
    
    def register_hooks(self):
        self.model.vit_satt, self.model.q_catt, self.model.q_satt, self.model.lm_satt = [], [], [], []

        # create hooks to capture attentions and their gradients
        vit_forward_hook = create_hook(self.model.vit_satt)
        qformer_cross_forward_hook = create_hook(self.model.q_catt)
        qformer_self_forward_hook = create_hook(self.model.q_satt)
        lm_forward_hook = create_hook(self.model.lm_satt)

        self.hooks = []
        # register hooks with corresponding locations
        for layer in self.model.vision_model.encoder.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(vit_forward_hook))
        for layer in self.model.qformer.encoder.layer:
            if layer.has_cross_attention:
                self.hooks.append(layer.crossattention.attention.register_forward_hook(qformer_cross_forward_hook))
            self.hooks.append(layer.attention.attention.register_forward_hook(qformer_self_forward_hook))
        for layer in self.model.language_model.model.decoder.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(lm_forward_hook))
    
    def remove_hooks(self):        
        for hook in self.hooks:
            hook.remove()

    @torch.no_grad()
    def chat(self, image, text):
        image = image.convert("RGB")
        prompt = f"Question: {text} Answer:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
    
        answer_ids = self.model.generate(
            **inputs,
            num_beams=self.args.num_beams,
            temperature=self.args.temperature,
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            max_length=self.args.max_length
        )
        answer_ids = answer_ids[0]
        return answer_ids
    
    @torch.enable_grad()
    def forward_with_grads(self, image, text, answer):
        image = image.convert("RGB")
        prompt = f"Question: {text} Answer:{answer}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

        self.qus_tokens = self.tokenizer.tokenize(text)

        # "<image></s>Question: {question} Answer:"
        self.img_start_idx = 0
        self.img_end_idx = self.img_start_idx + self.model.num_img_tokens
        self.qus_start_idx = self.img_end_idx + 3  # "</s>Question: "
        self.qus_end_idx = self.qus_start_idx + len(self.qus_tokens)
        self.ans_start_idx = self.qus_end_idx + 2   # " Answer:"
    
        outputs = self.model(
            **inputs,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        outputs.logits = outputs.language_model_outputs.logits[0, self.ans_start_idx-1:-1]
        outputs.hidden_states = [h[0, self.ans_start_idx-1:-1] for h in outputs.language_model_outputs.hidden_states]
        return outputs
    
    def compute_relevancy(self, word_idx):
        R_q_i = cal_qformer_relevancy(self.model, qformer_layers=self.model.qformer.encoder.layer)   # (num_query, 257)
        R_t_t_per_layer = cal_llm_relevancy(self.model, self.ans_start_idx+word_idx)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = torch.matmul(R['img'], R_q_i.cpu())[:, 1:]   # (num_layers, 256)
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        return R
    
    def compute_relevancy_cached(self, word_idx, gradcam_cache):
        R_q_i = cal_qformer_relevancy(self.model)   # (num_query, 257)
        R_t_t_per_layer, gradcam_cache = cal_llm_relevancy_cached(self.model, word_idx*self.model.num_llm_layers, gradcam_cache)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = torch.matmul(R['img'], R_q_i.cpu())[:, 1:]   # (num_layers, 256)
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        return R, gradcam_cache
    
    def preprocess_image_for_visualize(self, image):
        return self.image_processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'].squeeze(0)



class Blip2withGradient(Blip2ForConditionalGeneration):

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        # TODO (joao, raushan): refactor `generate` to avoid these operations with VLMs
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
