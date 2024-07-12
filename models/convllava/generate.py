import torch
from PIL import Image
"""
conv-llava/llava->convllava

ConvLLaVA-sft-1536,1024,768
LAION-CLIP-ConvNeXt-Large-512
"""
from convllava.constants import DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from convllava.conversation import conv_templates, SeparatorStyle
from convllava.model import *
from convllava.model.builder import load_pretrained_model
from convllava.utils import disable_torch_init
from convllava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from transformers.generation.utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from relevancy_utils import *


class ConvLLaVAForAnalysis():

    def __init__(self, args):
        
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(args.model_name, args.model_name, args.load_8bit, args.load_4bit, args.part_on_cpu, device='cuda')
        if args.part_on_cpu:
            self.model.model.vision_tower.to('cpu', dtype=torch.float32)
        self.args = args
        self.model.model.vision_tower.part_on_cpu = args.part_on_cpu
        
        self.model.num_img_tokens = self.model.model.vision_tower.num_patches // self.model.model.vision_tower.config.patch_size
        self.model.num_img_patches = self.model.num_img_tokens
        self.model.num_llm_layers = self.model.model.config.num_hidden_layers
        self.model.lm_head = self.model.lm_head
    
    def register_hooks(self):
        self.model.lm_satt = []

        lm_forward_hook = create_hook(self.model.lm_satt)

        self.hooks = []
        for layer in self.model.model.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(lm_forward_hook))

    def remove_hooks(self):        
        for hook in self.hooks:
            hook.remove()
    
    def ask(self, image, text):
        conv_temp = conv_templates['llava_v1']
        conv = conv_temp.copy()
        image = image.convert("RGB")

        image_tensor = process_images([image], self.image_processor, self.model.config)
        if self.args.part_on_cpu:
            image_tensor = image_tensor.to('cpu', dtype=torch.float32)
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + text

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        return conv, image_tensor
    
    
    @torch.no_grad()
    def chat(self, image, text):
        conv, image_tensor = self.ask(image, text)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device) # (1, )
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=self.args.do_sample, 
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_length, #self.args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            num_beams=self.args.num_beams,
            )[0]
        
        return output_ids

    @torch.enable_grad()
    def forward_with_grads(self, image, text, answer):
        conv, image_tensor = self.ask(image, text)
        conv.messages[-1][-1] = answer[:-4]
        # self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(answer)[:-1]) #avoid</s></s>
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device) # (1, )
        

        #"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIn this image, how many eyes can you see on the animal? ASSISTANT:" 
        self.qus_tokens = self.tokenizer.tokenize('\n'+text) 
        # \ndescribe->['▁', '<0x0A>', 'des', 'cribe']
        #describe->['describe']
        self.img_start_idx = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].data.item() # 35
        self.img_end_idx = self.img_start_idx + self.model.num_img_tokens # 144 256 576 
        self.qus_start_idx = self.img_end_idx + 2   # '\n'->'▁', '<0x0A>'
        self.qus_end_idx = self.qus_start_idx + len(self.qus_tokens)-2
        self.ans_start_idx = self.qus_end_idx + 5   # '▁A', 'SS', 'IST', 'ANT', ':'

        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        # for idx, logits in enumerate(outputs.logits):
        #     print(f"logits[{idx}].requires_grad: {logits.requires_grad}")
        outputs.logits = outputs.logits[0, self.ans_start_idx-1:-1] #
        outputs.hidden_states = [h[0, self.ans_start_idx-1:-1] for h in outputs.hidden_states] #33 hidden layer
        return outputs
    
    def compute_relevancy(self, word_idx):
        R_t_t_per_layer = cal_llm_relevancy(self.model, self.ans_start_idx+word_idx)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = R['img']
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        return R

    def compute_relevancy_cached(self, word_idx, gradcam_cache):
        R_t_t_per_layer, gradcam_cache = cal_llm_relevancy_cached(self.model, word_idx*self.model.num_llm_layers, gradcam_cache)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = R['img']
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        return R, gradcam_cache
    
    def preprocess_image_for_visualize(self, image):
        return self.model.model.vision_tower.image_processor.preprocess(image, do_rescale=False, do_normalize=False, return_tensors='pt')['pixel_values'].squeeze(0)

