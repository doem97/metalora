<div align="center">

# Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning
### 🌟 CVPR 2025 Highlight 🌟

**Zichen Tian¹**, **Yaoyao Liu²**, **Qianru Sun¹**

*¹Singapore Management University &emsp; ²University of Illinois Urbana-Champaign*

<p>
  <img src="assets/smu_logo.png" height="40" alt="SMU Logo" />
  &emsp;
  <img src="assets/uiuc_logo.png" height="40" alt="UIUC Logo" />
</p>


<p>
  <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Meta-Learning_Hyperparameters_for_Parameter_Efficient_Fine-Tuning_CVPR_2025_paper.html">
    <img src="https://img.shields.io/badge/paper-CVF-red" alt="Paper">
  </a>
  <a href="https://cvpr.thecvf.com/virtual/2025/poster/32721">
    <img src="https://img.shields.io/badge/poster-CVPR-blue" alt="Poster">
  </a>
</p>

</div>

Official source code for the paper **"Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning"** (CVPR 2025 Highlight). This repository provides a comprehensive framework for exploring various Parameter-Efficient Fine-Tuning (PEFT) methods on long-tailed datasets and introduces a novel meta-learning approach for optimizing their hyperparameters.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.8
- PyTorch 2.0
- Torchvision 0.15
- Tensorboard
- CUDA 11.7

Our experiments were primarily conducted on DGX V100 servers, but most can be reproduced on a single GPU with at least 20GB of memory.

### 2. Installation
```bash
# Create and activate a conda environment
conda create -n metalora python=3.8 -y
conda activate metalora

# Install core dependencies
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard

# Install other requirements
pip install -r requirements.txt
```

### 3. Dataset Preparation
We provide a convenient script in `data/hf_dataset.sh` to download and prepare all datasets from Hugging Face.

First, ensure you have `huggingface-cli` and the accelerated transfer library `hf_transfer` installed:
```bash
pip install -U "huggingface_hub[cli]"
pip install hf_transfer
```

Then, execute the script from the repository's root directory:
```bash
bash data/hf_dataset.sh
```

---

## 🔧 Usage

### Basic Command Structure
The general command to launch an experiment is:
```bash
python main.py --dataset [data_config] --model [model_config] --tuner [tuner_config] --opts [OPTIONS]
```
- `--dataset`: A data config from `configs/data` (e.g., `cifar100_ir100`).
- `--model`: A model config from `configs/model` (e.g., `clip_vit_b16`).
- `--tuner`: (Optional) A tuner config from `configs/tuner` (e.g., `adaptformer`).
- `--opts`: (Optional) A list of key=value pairs to override any setting from the base and YAML configs. For a full list of options, please see the files in the `configs` directory.

### Training Strategies

Our framework supports a wide range of fine-tuning strategies, from classic full fine-tuning to a rich set of PEFT methods.

#### Supported PEFT & Backbones

The following PEFT methods can be applied to **CLIP-ViT**, **timm-ViT**, and **SatMAE-ViT** backbones.

| Method Family | Options | Description |
| :--- | :--- | :--- |
| **Prompt Tuning** | `vpt_shallow`, `vpt_deep` | Adds learnable tokens to the input sequence. |
| **Adapter-based** | `adapter`, `adaptformer` | Inserts small, trainable modules between transformer layers. |
| **LoRA-based** | `lora`, `lora_mlp`, `use_flora` | Uses low-rank decomposition for weight updates. `FLoRA` offers fine-grained control. |
| **Feature Scaling**| `ssf_attn`, `ssf_mlp`, `ssf_ln`| Learns to scale and shift features within the network. |
| **Subset Tuning** | `bias_tuning`, `ln_tuning`, `mask` | Fine-tunes only a subset of existing parameters (biases, LayerNorms, or a random mask). |

#### Execution Modes
You can control which parts of the model are trained using the `--opts` flag.

- **PEFT (Default):** If a `--tuner` is specified, only the PEFT modules and the classifier head are trained.
  ```bash
  # Run AdaptFormer on CIFAR-100-LT
  python main.py --dataset cifar100_ir100 --model clip_vit_b16 --tuner adaptformer
  ```
- **Full Fine-tuning:** To update all weights of the backbone model, set `fine_tuning=True`.
  ```bash
  # Full fine-tuning on Places-LT
  python main.py --dataset places_lt --model clip_vit_b16 --opts fine_tuning=True
  ```
- **Linear Probing:** To train only the classifier head, set `head_only=True`.
  ```bash
  # Linear probing on CIFAR-100-LT
  python main.py --dataset cifar100_ir100 --model clip_vit_b16 --opts head_only=True
  ```

### Evaluation
- **Test a trained model:** Use `test_only=True` and specify the model's output directory.
  ```bash
  python main.py --dataset [data] --model [model] --opts test_only=True model_dir=path/to/your/checkpoint_dir
  ```
- **Evaluate on the training set:** Use `test_train=True`.
  ```bash
  python main.py --dataset [data] --model [model] --opts test_train=True model_dir=path/to/your/checkpoint_dir
  ```

---

## 💡 Meta-Training for Hyperparameter Optimization

This is the core contribution of our work: a framework to optimize PEFT hyperparameters via bi-level optimization.

### Concept
When enabled, the training data is split into a primary training set, a meta-train set, and a meta-validation set. The optimization proceeds in two nested loops:
1.  **Inner Loop:** The model's standard parameters (e.g., LoRA weights) are trained on batches from the primary training set.
2.  **Outer Loop:** Periodically, the framework simulates a training step on the meta-train set and evaluates the performance on the meta-validation set. The resulting validation loss is used to update the **meta-parameters** (e.g., the learning rates or ranks within the PEFT modules).

### How to Use
Enable meta-training by setting `use_meta=True`.

- **Example: Run FLoRA with Meta-Training on CIFAR-100-LT**
  ```bash
  python main.py \
      --dataset cifar100_ir100 \
      --model clip_vit_b16 \
      --tuner flora \
      --opts use_meta=True meta_lr=0.001
  ```
#### Key Meta-Training Options
- `use_meta`: Set to `True` to enable the feature.
- `meta_data_ratio`: Fraction of data to reserve for meta-learning (e.g., `0.1` for 10%).
- `meta_lr`: The learning rate for the outer-loop meta-optimizer.
- `meta_update_freq`: How many epochs between each meta-optimization step.
- `meta_inner_steps`: Number of optimization steps in the inner loop during a meta-update.

---

## ✍️ Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Tian_2025_CVPR,
    author    = {Tian, Zichen and Liu, Yaoyao and Sun, Qianru},
    title     = {Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23037-23047}
}
```


---

## 🙏 Acknowledgment

We thank the authors for the following repositories for code reference:
[[OLTR]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [[Classifier-Balancing]](https://github.com/facebookresearch/classifier-balancing), [[Dassl]](https://github.com/KaiyangZhou/Dassl.pytorch), [[CoOp]](https://github.com/KaiyangZhou/CoOp). Our code is largely re-implement based on [[LIFT]](https://github.com/shijxcs/LIFT), many thanks to LIFT authors' significant contributions!

