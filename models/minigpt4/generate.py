from typing import Optional
import torch
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from transformers.generation.utils import *
from transformers import BitsAndBytesConfig
from relevancy_utils import *


class MiniGPT4ForAnalysis():

    def __init__(self, args):
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config)
        if not args.part_on_cpu:
            self.model = self.model.cuda()
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).cuda() for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.args = args
        self.cfg = cfg
        self.tokenizer = self.model.llama_tokenizer
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.image_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model.num_img_patches = self.model.visual_encoder.patch_embed.num_patches
        self.model.num_img_tokens = model_config.num_query_token
        self.model.num_llm_layers = self.model.llama_model.config.num_hidden_layers
        self.model.lm_head = self.model.llama_model.lm_head
    
    def register_hooks(self):
        self.model.vit_satt, self.model.q_catt, self.model.q_satt, self.model.lm_satt = [], [], [], []

        # create hooks to capture attentions and their gradients
        vit_forward_hook = create_hook(self.model.vit_satt)
        qformer_cross_forward_hook = create_hook(self.model.q_catt)
        qformer_self_forward_hook = create_hook(self.model.q_satt)
        lm_forward_hook = create_hook(self.model.lm_satt)

        self.hooks = []
        # register hooks to corresponding locations
        for layer in self.model.visual_encoder.blocks:
            self.hooks.append(layer.attn.register_forward_hook(vit_forward_hook))
        for layer in self.model.Qformer.bert.encoder.layer:
            if layer.has_cross_attention:
                self.hooks.append(layer.crossattention.self.register_forward_hook(qformer_cross_forward_hook))
            self.hooks.append(layer.attention.self.register_forward_hook(qformer_self_forward_hook))
        for layer in self.model.llama_model.model.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(lm_forward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def ask(self, image, text):
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}
        CONV_VISION = conv_dict[self.cfg.model_cfg.model_type]
        conv = CONV_VISION.copy()
        image = image.convert("RGB")
        img_list = []
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)

        self.encode_img(img_list)

        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)
        return conv, img_list
    
    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.image_processor(raw_image).unsqueeze(0).to(self.model.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.image_processor(raw_image).unsqueeze(0).to(self.model.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.model.device)

        ######################## cpu or gpu
        if self.args.part_on_cpu:  
            image_emb, _ = self.model.encode_img(image.to('cpu'))
            img_list.append(image_emb.to('cuda').to(self.model.llama_model.dtype))
        else:
            image_emb, _ = self.model.encode_img(image)
            img_list.append(image_emb.to(self.model.llama_model.dtype))
    
    def answer_prepare(self, conv, img_list):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + self.args.max_length
        max_length = 2000
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length) 
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_length=self.args.max_length, ###max_new_tokens
            stopping_criteria=self.stopping_criteria,
            num_beams=self.args.num_beams,
            do_sample=self.args.do_sample,
            min_length=1,
            top_p=self.args.top_p,
            repetition_penalty=1.05,
            length_penalty=1,
            temperature=self.args.temperature
        )
        return generation_kwargs
    
    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output # self.tokenizer.decode(output[0])
    
    @torch.no_grad()
    def chat(self, image, text):
        conv, img_list = self.ask(image, text)
        generation_dict = self.answer_prepare(conv, img_list)
        answer_ids = self.model_generate(**generation_dict)[0]
        return answer_ids

    @torch.enable_grad()
    def forward_with_grads(self, image, text, answer):

        # answer = answer.split('###')[0] # conv.get_prompt() will add '###' : '######'->['####','##'], '###->['##','#']

        # vit & qformer forward
        conv, img_list = self.ask(image, text)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        embs = self.model.get_context_emb(prompt, img_list)

        self.qus_tokens = self.tokenizer.tokenize(text)
        
        # Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>
        self.img_start_idx = 42
        self.img_end_idx = self.img_start_idx + self.model.num_img_tokens
        self.qus_start_idx = self.img_end_idx + 3   # "</Img> "
        self.qus_end_idx = self.qus_start_idx + len(self.qus_tokens)
        self.ans_start_idx = self.qus_end_idx + 5   # "###Assistant: "

        # llama forward
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=embs,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
            )
        outputs.logits = outputs.logits[0, self.ans_start_idx-1:-1]
        outputs.hidden_states = [h[0, self.ans_start_idx-1:-1] for h in outputs.hidden_states]
        return outputs
    
    def compute_relevancy(self, word_idx):
        R_q_i = cal_qformer_relevancy(self.model, qformer_layers=self.model.Qformer.bert.encoder.layer)   # (num_query, 257)
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
        return self.image_processor.visualize(image)

