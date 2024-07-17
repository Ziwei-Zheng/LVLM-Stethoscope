# 🩺LVLM-Stethoscope


[![Static Badge](https://img.shields.io/badge/Project-Page-Green)]()
[![Static Badge](https://img.shields.io/badge/Paper-Arxiv-red)]()


Main contributor: [Le Yang](https://github.com/yangle15), [Ziwei Zheng](https://github.com/Ziwei-Zheng), [Boxu Chen](https://github.com/Chen-Boxu)

This repository contains the code of LVLM-Stethoscope, which can be used for diagnosing Large Vision-langurage models predictions, either in production or while developing models.


![LLaVA-1.5-7B-HF](demo.gif)


**🤔️ Why Building this Project?**

- Recent promising Large Vision-Language Models (LVLMs) are notorious for generating outputs that are inconsistent with the visual content, a challenge known as **hallucination**. However, our research team found that a powerful analysis tool is still absent to investigate what happens when these hallucinated decisions are made. This motivates us to present an interactive application to understand the internal mechanisms of LVLMs. Therefore, LVLM-Stethoscope is born👶.


**🔨 What can I do with LVLM-Stethoscope?**

The proposed LVLM-Stethoscope contains a series of ensembled functions that can help you to understand the under lying decision making process of the recent LVLMs. It can be used as but not limited to:

- **A powerful visualization tool for model diagnosing**: 
- **A useful evalutation tool**: 
- **A hallucination warning tool**: 



**⚙️ Tested and supporting models**

We have tested LVLM-Stethoscope on a series of recent relased LVLMs. More models are coming soon...

| Models | Link | Status |
|:------:|:----:|:------:|
| LLaVA | https://huggingface.co/docs/transformers/model_doc/llava | ✅ |
| BLIP-2 | https://huggingface.co/docs/transformers/model_doc/blip-2 | ✅ |
| MiniGPT-4 | https://github.com/Vision-CAIR/MiniGPT-4 | ✅ |
| Conv-LLaVA | https://github.com/alibaba/conv-llava | ✅ |
| InstructBLIP | https://huggingface.co/docs/transformers/model_doc/instructblip | TODO |
| mPLUG-Owl2 | https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2 | TODO |
| 🧑‍💻Your own | See how to visualize your own LVLMs  | - |

<!-- 
🔥We are very willing to **help everyone share and promote new projects** based on Segment-Anything, Please check out here for more amazing demos and works in the community: [Highlight Extension Projects](#highlighted-projects). You can submit a new issue (with `project` tag) or a new pull request to add new project's links.  -->

**🍇 Updates**
- **`2024/07/12`**: We release the demo code based on Gradio, four LVLMs are supported!


## Getting Started

### Installation

Git clone our repository, creating a python environment and activate it via the following command.

```bash
git clone https://github.com/Ziwei-Zheng/LVLM-Stethoscope.git
cd LVLM-Stethoscope
conda create --name lvlm-ss python=3.10
conda activate lvlm-ss
pip install -r requirements.txt
```

### Run supported models

Specify the name in [["llava-hf/llava-1.5-7b-hf"](https://huggingface.co/llava-hf/llava-1.5-7b-hf), ["Salesforce/blip2-opt-6.7b"](https://huggingface.co/Salesforce/blip2-opt-6.7b), "minigpt4-7b", [&#34;ConvLLaVA-sft-1536/1024/768&#34;](https://huggingface.co/ConvLLaVA)] to run the model. Then open the local URL to start conversation (default: http://127.0.0.1:7860)

```bash
CUDA_VISIBLE_DEVICES=[GPUS] python demo_meta.py --model-name [MODEL_NAME]
```

Note that since not all sub-models (e.g. ViT & Q-former) in MiniGPT-4 and ConvLLaVA codebases are using [transformers](https://huggingface.co/docs/transformers/v4.41.3/en/index) backends, they are not able to run in multi-GPUs. You can 1) Run the whole model within one single GPU (> 40G memory), or 2) Run sub-models that are not included in [transformers](https://huggingface.co/docs/transformers/v4.41.3/en/index) on CPU by parsing `--part-on-cpu` and other parts (e.g. LLaMA) on single or multi-GPUs to save memory.

We have tested llava-1.5-7b & blip2-opt-6.7b on 2 RTX 4090 GPUs with 24G memory, and minigpt4-7b & conv-llava-7b on 1 RTX 4090 GPU with `--part-on-cpu` enabled.


### Run customized models

The main functionalities and the visualization interface have been encapsulated and integrated into `demo_meta.py`, one only need to customize your own models in the following steps:

- Move the model specification and necessary files in the directory. e.g., `./minigpt4`.

- Create a folder in `./models` that indicates your own model, and create `generate.py`.

- Define [self.model, self.tokenizer, self.image_processor, self.model.num_img_patches, self.model.num_img_tokens, self.model.num_llm_layers, self.model.lm_head] as the necessary attributes in `__init__()`.

- Define `register_hooks()` to create hooks in self/cross-attention layers for relevancy analysis.

- Define `chat()` with image and user question as inputs, return the generated per-token answer ids.

- Define `forward_with_grads()` with the generated answer inserted in the conversation template for a single parallel forward to obtain per-token answer logits. Then carefully define the indexes of `<Img>`, `<Qus>` and `<Ans>` tokens according to your organization. These indexes are used to find out specific locations in the whole outputs for further analysis.

- Define `compute_relevancy()` to specialize how to obtain relevancy scores according to your model. We have provided computations of vanilla transformers like `ViT` and `LLaMA`, and architectures with mixed attentions like `Q-former` in `relevancy_utils.py`. You can also add customized functions if nedded.

If you have any questions on how to build with customized models, please feel free to open an issue or contact me at ziwei.zheng@stu.xjtu.edu.cn.



## Acknowledgement

- [Transformer-MM-Explainability](https://arxiv.org/abs/2103.15679): Introducing relevancy scores for explainability analysis.

- [LVLM-Intrepret](https://arxiv.org/abs/2404.03118): Another awesome tools to interpret LVLMs.


## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
<!-- ```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
``` -->

