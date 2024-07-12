import gradio as gr
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

import shutil
import argparse
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import gc
import re

import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
plt.rcParams['figure.dpi'] = 220

from models import *

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/workspace/llava-1.5-7b-hf")  
    # llava-1.5-7b-hf, minigpt4-7b, blip2-opt-6.7b, ConvLLaVA-sft-1536,1024,768
    parser.add_argument("--cfg-path", type=str, default='/workspace/LVLM-Stethoscope/minigpt4/minigpt4_eval.yaml')
    parser.add_argument("--part-on-cpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

# global model
print('Initializing Chat')
args = parse_args()
if 'llava' in args.model_name:
    sys = LlavaForAnalysis(args)
elif 'ConvLLaVA' in args.model_name: 
    sys = ConvLLaVAForAnalysis(args)
elif 'blip2' in args.model_name:
    sys = Blip2ForAnalysis(args)
elif 'minigpt4' in args.model_name:
    sys = MiniGPT4ForAnalysis(args)
else:
    raise NotImplementedError


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def gradio_reset():
    flush()
    return gr.update(value=None), gr.update(placeholder='Please upload your image first', interactive=False), \
           gr.update(value="Upload Picture", interactive=False), gr.update(choices=[],value=None), gr.update(value=32), \
           None, None, None, None, None, None, None, None, None, \
           gr.update(value=1)


def upload_img(gr_img):
    if gr_img is None:
        return None, None, gr.update(interactive=True)

    return gr_img, gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False)


def gradio_ask(chatbot, text_input):
    if len(text_input) == 0:
        return chatbot, gr.update(interactive=True, placeholder='Input should not be empty!')
    flush()
    chatbot = chatbot + [[text_input, None]]

    return chatbot, None, None, None, None, None, None

def gradio_answer(chatbot, gr_img, question, temperature, num_beams):

    
    if 'ConvLLaVA' in args.model_name: 
        qus_tokens = sys.tokenizer.tokenize('\n'+question)[2:]
    else:
        qus_tokens = sys.tokenizer.tokenize(question)

    # first generation forward pass

    sys.args.temperature = temperature
    sys.args.do_sample = True if 0 < temperature < 1 else False
    sys.args.num_beams = num_beams

    ans_ids = sys.chat(gr_img, question)
    answer = sys.tokenizer.decode(ans_ids, skip_special_tokens=False)
    if 'minigpt4' in args.model_name:   # num_beams > 1
        answer = answer.split('###')[0]                 
        answer = answer.split('Assistant:')[-1].strip() 
        answer += '###'
        len_useful = len(sys.tokenizer.encode(answer, add_special_tokens=False))
        ans_ids = ans_ids[:len_useful]

    # second parrallel forward pass
    sys.register_hooks()
    outputs = sys.forward_with_grads(gr_img, question, answer)
    sys.remove_hooks()

    # per-token backward to compute relevancy scores
    relevancy_scores, probs = [], []
    for word_idx, logits in enumerate(outputs.logits):
        logits = logits.unsqueeze(0)
        token_id_one_hot = F.one_hot(ans_ids[word_idx], num_classes=logits.size(-1)).float().to(logits.device)
        token_id_one_hot = token_id_one_hot.view(1, -1)
        token_id_one_hot.requires_grad_(True)
        sys.model.zero_grad()
        logits.backward(gradient=token_id_one_hot, retain_graph=True)
        R = sys.compute_relevancy(word_idx)
        relevancy_scores.append(R)
        probs.append(torch.softmax(logits, dim=1).detach().max().item())
    
    chatbot[-1][1] = answer
    ans_tokens = [sys.tokenizer.decode(id) for id in ans_ids]
    ans_tokens_to_select = [f'{index}:{token}' for index, token in enumerate(ans_tokens)]
    
    # linear probe to obtain per-layer prediction evolution (greedy search)
    ans_hidden_states = torch.stack(outputs.hidden_states)
    lm_head = sys.model.lm_head
    with torch.no_grad():
        num_layers, ans_len, _ = ans_hidden_states.shape
        ans_hidden_states = ans_hidden_states.flatten(0, 1)
        ans_logits = lm_head(ans_hidden_states.to(lm_head.weight.device))
        ans_logits = ans_logits.view(num_layers, ans_len, -1)   # (1+32, ans_len, vocab_size)

    return chatbot, gr.update(choices=ans_tokens_to_select, interactive=True), relevancy_scores, probs, qus_tokens, ans_tokens, ans_logits


def prob_plot(ans_tokens, probs):

    if len(ans_tokens) <= 32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([probs], 
        linewidths=.1, square=True, cmap='Greens', vmax=1., cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"})
        ax.set_xticks(np.arange(len(probs))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels(['Probs'])
        fig.tight_layout()
    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [probs[i:i+32] for i in range(0, len(probs), 32)]
        if len(wrapped_tokens[-1]) < 32:
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate([wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))])

        num_subplots = len(wrapped_values)

        vmin = np.min(probs)
        vmax = np.max(probs)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Greens', norm=norm)

        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))

        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i] if num_subplots > 1 else axes
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Greens', vmin=vmin, vmax=vmax, cbar=False)
            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)#, fontsize = 8
            ax.set_yticklabels(['Probs'])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig)    


def token_evo_plot(selected_token, ans_logits):

    if not selected_token:
        print("Selected token is empty, returning empty figure and output.")
        return None
    else:
        index, _ = selected_token.split(':', 1)
        index = int(index)
    
    ans_logits = ans_logits[:, index]   # (1+32, vocab_size)
    per_layer_probs = torch.softmax(ans_logits, dim=-1).float().detach().cpu().numpy()   # (1+32, vocab_size)
    per_layer_confs = np.max(per_layer_probs, axis=-1)  # (1+32,)

    per_layer_ids = np.argmax(per_layer_probs, axis=-1)  # (1+32,)
    per_layer_words = [sys.tokenizer.decode(int(id)) for id in per_layer_ids]

    fig = plt.figure(figsize=(15, 2))
    ax = seaborn.heatmap([per_layer_confs], 
       linewidths=.5, square=True, cmap='Blues', vmax=1., cbar_kws={"orientation": "horizontal", "shrink": 0.3, "location": "top"})# 
    ax.set_xticks(np.arange(len(per_layer_confs))+0.5)
    ax.set_xticklabels(per_layer_words, rotation=60)
    ax.set_yticklabels(['Probs'])
    fig.tight_layout()

    return gr.update(value=fig)


def returnfig(qus_tokens, prev_ans_tokens, gr_img, R, layer_select, low_th=0.):
        
    gr_img = gr_img.convert('RGB')
    image_tensor = sys.preprocess_image_for_visualize(gr_img)

    plt.close('all')

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0))
    ax.axis('off')

    qus_tokens = [token.lstrip('▁Ġ') for token in qus_tokens] #Ġ-blip2

    if layer_select == 0:
        qus_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(qus_tokens)], 'R': [0] * len(qus_tokens)})
        return fig, qus_pd, None, 'Relevancy not available before layer 0.', 0.5
    
    else:
        r_raw_img = R['raw_img'][layer_select - 1]
        r_img = R['img'][layer_select - 1]
        r_qus = R['qus'][layer_select - 1]
        # top_img, _ = r_img.topk(k=topk)
        sum_img = r_img.sum()
        if R['ans'] is not None:
            r_ans = R['ans'][layer_select - 1]
            r_text = torch.cat([r_qus, r_ans])
        else:
            r_text = r_qus
        sum_text = r_text.sum()

        outstr = f'Sum R_img / R_text: {sum_img / sum_text :.2f}'
        max_img = r_raw_img.max().item()
        max_text = r_text.max().item()
        # max_value = max(max_img, max_text)

        h = w = int(sys.model.num_img_patches**0.5)
        reshaped_tensor = r_raw_img.reshape(h, w).unsqueeze(0).unsqueeze(0).float()
        reshaped_tensor[reshaped_tensor < low_th] = 0.

        interpolated_tensor = F.interpolate(reshaped_tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False)
        interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)
        tensor_np = np.float32(interpolated_tensor.detach().cpu())

        vis = ax.imshow(tensor_np, cmap='coolwarm', alpha=0.75, vmax=max_img)
        fig.colorbar(vis, ax=ax)
        
        qus_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(qus_tokens)], 'R': r_qus.tolist()})
        ans_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(prev_ans_tokens)], 'R': r_ans.tolist()}) if R['ans'] is not None else None

        return fig, qus_pd, ans_pd, outstr, max_img, max_text

def token_plot(selected_token, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select):

    print(f'draw {selected_token} at layer {layer_select}')
    
    if not selected_token:
        print("Selected token is empty, returning empty figure and output.")
        return None, None, None, None, None, None
    else:
        index, _ = selected_token.split(':', 1)
        index = int(index)
    
    R = relevancy_scores[index]
    prev_ans_tokens = ans_tokens[:index] if index > 0 else []
    fig, qus_pd, ans_pd, outstr, max_img, max_text = returnfig(qus_tokens, prev_ans_tokens, gr_img, R, layer_select)
    _step = max_img/15

    img_state = gr.update(value=fig)
    qus_state = gr.update(value=qus_pd, y_lim=[0,max_text])
    ans_state = gr.update(value=ans_pd, y_lim=[0,max_text]) if ans_pd is not None else None
    str_state = gr.update(value=outstr)

    return img_state, qus_state, ans_state, str_state, gr.update(maximum=max_img,value=max_img,step=_step)

def max_image_relevency_plot(selected_token, relevancy_scores, gr_img, layer_select, max_image_relevency):
    
    index, _ = selected_token.split(':', 1)
    index = int(index)
    low_th = 0.
    R = relevancy_scores[index]

    gr_img = gr_img.convert('RGB')
    image_tensor = sys.preprocess_image_for_visualize(gr_img)

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0))
    ax.axis('off')

    r_raw_img = R['raw_img'][layer_select - 1]
    r_img = R['img'][layer_select - 1]
    r_qus = R['qus'][layer_select - 1]
    sum_img = r_img.sum()
    if R['ans'] is not None:
        r_ans = R['ans'][layer_select - 1]
        r_text = torch.cat([r_qus, r_ans])
    else:
        r_text = r_qus
    sum_text = r_text.sum()

    # outstr = f'Sum R_img / R_text: {sum_img / sum_text :.2f}'
    # max_value = max(r_img.max().item(), r_text.max().item())

    h = w = int(sys.model.num_img_patches**0.5)
    reshaped_tensor = r_raw_img.reshape(h, w).unsqueeze(0).unsqueeze(0).float()
    reshaped_tensor[reshaped_tensor < low_th] = 0.

    interpolated_tensor = F.interpolate(reshaped_tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False)
    interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)
    tensor_np = np.float32(interpolated_tensor.detach().cpu())

    vis = ax.imshow(tensor_np, cmap='coolwarm', alpha=0.75, vmax=max_image_relevency)
    fig.colorbar(vis, ax=ax)
    img_state = gr.update(value=fig)
    return img_state


def cal_vc(relevancy_scores, vcmode):

    visual_confs = []
    for R in relevancy_scores:
        # (layer_idx, content)
        r_img = R['img']
        r_qus = R['qus']
        # top_img, _ = r_img.topk(k=int(topk), dim=1)
        if vcmode == "sum":
            sum_img = r_img.sum(dim=1)
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                # top_text, _ = r_text.topk(k=int(topk), dim=1)
                sum_text = r_text.sum(dim=1)
            else:
                # top_text, _ = r_qus.topk(k=int(topk), dim=1)
                sum_text = r_qus.sum(dim=1)
            vc = sum_img / sum_text   # (layer_idx,)
            _step = 0.1
            # _value = 3
        elif vcmode == "max":
            max_img = r_img.max(dim=1)[0]
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                max_text = r_text.max(dim=1)[0]
            else:
                max_text = r_qus.max(dim=1)[0]
            vc = max_img / max_text   # (layer_idx,)
            _step = 0.05
            # _value = 0.3
        elif vcmode == "mean":
            mean_img = r_img.mean(dim=1)
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                # top_text, _ = r_text.topk(k=int(topk), dim=1)
                mean_text = r_text.mean(dim=1)
            else:
                # top_text, _ = r_qus.topk(k=int(topk), dim=1)
                mean_text = r_qus.mean(dim=1)
            vc = mean_img / mean_text   # (layer_idx,)
            _step = 1e-3
        visual_confs.append(vc)
    
    # ->(num_layers, num_tokens)
    visual_confs = torch.stack(visual_confs, dim=0).detach().cpu().numpy().T
    visual_confs = visual_confs[::-1]
    max_last_layer = visual_confs[0].max()
    
    return visual_confs, gr.update(maximum=max_last_layer, step=_step, value=max_last_layer)

def vc_plot(visual_confs, ans_tokens, layer_select):
    
    all_layers, num_tokens = visual_confs.shape
    value = visual_confs[all_layers - layer_select,:]
    max_current_layer = value.max()
    min_current_layer = value.min()

    if len(ans_tokens)<=32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([value], 
            linewidths=.1, square=True, cmap='Reds', vmax=max_current_layer, cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"}
        )
        ax.set_xticks(np.arange(len(value))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels([layer_select])
        fig.tight_layout()


    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [value[i:i+32] for i in range(0, len(value), 32)]
        if len(wrapped_tokens[-1]) < 32: #对齐
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate( [wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))] )  # 使用0进行填充

        num_subplots = len(wrapped_values)

        vmin = np.min(value)
        vmax = np.max(value)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Reds', norm=norm)


        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))


        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i]
            vals = np.clip(vals, None, vmax)#-1e-9
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Reds', vmin=vmin, vmax=vmax, cbar=False)#-1e-8

            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)
            ax.set_yticklabels([layer_select])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig), gr.update(maximum=max_current_layer, minimum=min_current_layer, value=max_current_layer)


def max_vc_plot(visual_confs, ans_tokens, max_vc, layer_select):

    all_layers, num_tokens = visual_confs.shape
    value = visual_confs[all_layers - layer_select,:]

    if len(ans_tokens) <= 32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([value], 
            linewidths=.1, square=True, cmap='Reds', vmax=max_vc, cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"}
        )
        ax.set_xticks(np.arange(len(value))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels([layer_select])
        fig.tight_layout()

    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [value[i:i+32] for i in range(0, len(value), 32)]
        if len(wrapped_tokens[-1]) < 32: #对齐
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate([wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))])

        num_subplots = len(wrapped_values)

        vmin = np.min(value)
        vmax = max_vc
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Reds', norm=norm)

        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))

        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i]
            vals = np.clip(vals, None, vmax-1e-9)
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Reds', vmin=vmin-1e-8, vmax=vmax, cbar=False)
            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)
            ax.set_yticklabels([layer_select])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig)




title = """<h1 align="center">Demo of LVLM Interpretability</h1>"""


with gr.Blocks() as demo:
    
    gr.Markdown(title)

    inputs = gr.State()
    relevancy_scores = gr.State()
    probs = gr.State()
    qus_tokens = gr.State()
    ans_tokens = gr.State()
    ans_logits = gr.State()
    visual_confs = gr.State()

    gr.Markdown("""<h3>Chat Box</h3>""")
    with gr.Row():
        with gr.Column(scale=2):
            upload_button = gr.Button(scale=1,value="Upload & Start Chat", interactive=False, variant="primary")
            gr_img = gr.Image(scale=3,type="pil")
            # reset_button = gr.Button(scale=1,value="Reset", interactive=True, variant="primary")
        with gr.Column(scale=4):
            model_name = args.model_name.split('/')[-1]
            chatbot = gr.Chatbot(label=f'{model_name} (single-round conversion)')
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(label='Temperature', minimum=0.1, maximum=1, step=0.1, value=1, interactive=True) # (0 indicates None)
                with gr.Column(scale=1):
                    num_beams = gr.Slider(label='Num Beams', minimum=1, maximum=5, step=1, value=1, interactive=True)
            text_input = gr.Textbox(scale=1,label='User', placeholder='Please upload your image first.', interactive=False)

    gr.Markdown("""<h3>Visual Contribution</h3>""")
    with gr.Row():
        with gr.Column(scale=1):
            vc_layer_select = gr.Slider(label='LLM Layer', minimum=1, maximum=32, step=1, value=32, interactive=True)
        with gr.Column(scale=1):
            vcmode = gr.Dropdown(label='Calculation mode', choices=["max", "sum", "mean"], value="max")
        with gr.Column(scale=1):
            max_vc = gr.Slider(label='Max vc', minimum=0, maximum=3, step=0.1, value=3, interactive=True)
    with gr.Row():
        vc_img = gr.Plot(min_width=80, scale=1, label='Per token visual contribution')
    
    gr.Markdown("""<h3>Answer Probabilities</h3>""")
    with gr.Row():
        token_prob = gr.Plot(min_width=80, scale=1, label='')

    gr.Markdown("""<h3>Answer Tokens for Selection</h3>""")
    with gr.Row():
        ans_tokens_to_select = gr.Radio(choices=[], label="Tokens", info="Select one token from the answer.")
    
    gr.Markdown("""<h3>Prediction Evolution in LLM</h3>""")
    with gr.Row():
        token_evo = gr.Plot(min_width=80, scale=1, label='')
    
    gr.Markdown("""<h3>Relevancy Analysis</h3>""")
    with gr.Row():
        with gr.Column(scale=2):
            layer_select = gr.Slider(label='LLM Layer', minimum=1, maximum=32, step=1, value=32, interactive=True)
            result_img = gr.Plot(label='Relevancy to Image')
            max_image_relevency = gr.Slider(label='max image relevency', minimum=0, step=1e-3, interactive=True)
            printstr = gr.Textbox(label='Visual Contribution')
        with gr.Column(scale=4):
            result_qus = gr.BarPlot(scale=1, x='Tokens', y='R', x_label_angle=300, label='Relevancy to User Input', tooltip=['Tokens','R'], interactive=True) # height=145,
            result_ans = gr.BarPlot(scale=1, x='Tokens', y='R', x_label_angle=300, label='Relevancy to Previous Answer', tooltip=['Tokens','R'], interactive=True)# height=145,


    """image and text_input"""
    gr_img.upload(        
        gradio_reset, 
        [], 
        [chatbot, text_input, 
         upload_button, ans_tokens_to_select, layer_select, 
         token_prob, result_img, result_qus, result_ans, printstr, vc_img, relevancy_scores, probs, token_evo,
         max_vc],
        queue=False
    ).then(
        upload_img,
        [gr_img],
        [gr_img, text_input, upload_button]
    )

    text_input.submit(
        gradio_ask, 
        [chatbot, text_input],
        [chatbot, result_img, result_qus, result_ans, vc_img, token_prob, token_evo]
    ).then(
        gradio_answer,
        [chatbot, gr_img, text_input, temperature, num_beams],
        [chatbot, ans_tokens_to_select, relevancy_scores, probs, qus_tokens, ans_tokens, ans_logits]
    ).then(
        cal_vc,
        [relevancy_scores, vcmode],
        [visual_confs, max_vc]
    ).then(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    ).then(
        prob_plot,
        [ans_tokens, probs],
        [token_prob]
    )

    """visual contribution"""
    vc_layer_select.release(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    )
    vcmode.change(
        cal_vc,
        [relevancy_scores, vcmode],
        [visual_confs, max_vc]
    ).then(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    )
    max_vc.release(
        max_vc_plot,
        [visual_confs, ans_tokens, max_vc, vc_layer_select],
        [vc_img]
    )
    
    """token and layer"""
    ans_tokens_to_select.select(
        token_plot,
        [ans_tokens_to_select, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select], 
        [result_img, result_qus, result_ans, printstr, max_image_relevency]
    ).then(
        token_evo_plot,
        [ans_tokens_to_select, ans_logits],
        [token_evo]
    )
    layer_select.release(
        token_plot,
        [ans_tokens_to_select, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select], 
        [result_img, result_qus, result_ans, printstr, max_image_relevency]
    )
    max_image_relevency.release(
        max_image_relevency_plot,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, max_image_relevency],
        [result_img]
    )

    # gr_img.clear(
    #     gradio_reset, 
    #     [], 
    #     [chatbot, text_input, 
    #      upload_button, ans_tokens_to_select, layer_select, 
    #      token_prob, result_img, result_qus, result_ans, printstr, vc_img, relevancy_scores, probs, token_evo,
    #      max_vc],
    #     queue=False
    # )
    # reset_button.click(
    #     gradio_text_reset, 
    #     [], 
    #     [chatbot, text_input, 
    #     upload_button, ans_tokens_to_select, layer_select, 
    #     token_prob, result_img, result_qus, result_ans, printstr, vc_img, relevancy_scores, probs, token_evo,
    #     max_vc],
    # queue=False
    # )
    


demo.queue(max_size=10)
demo.launch(inbrowser=True, share=False)
