# [AAAI 2026] On the Robustness of Multimodal Language Models Towards Distractions

This repository contains the official implementation of the paper ["On the robustness of multimodal language model towards distractions"](https://arxiv.org/abs/2502.09818).

## Abstract

This work investigates the resilience of Vision-Language Models (VLMs) when exposed to distracting information in visual and textual contexts. We build a comprehensive benchmark using the ScienceQA dataset and introduce various types of distractions to evaluate the reasoning capabilities of state-of-the-art VLMs under noisy conditions.

## Key Findings

- Most state-of-the-art VLMs, including GPT-4, are vulnerable to various types of distractions
- Models show noticeable degradation in reasoning capabilities when confronted with distractions
- VLMs are more sensitive to textual distractions than visual ones
- InternVL2 demonstrates higher robustness compared to other models
- Significant opportunities exist for improving VLM performance under noisy conditions

## I-ScienceQA Dataset
Our dataset with injected textual/visual hints for testing model robustness. Features elementary/middle school questions across multiple subjects with various distractor types built upond ScienceQA dataset. Our dataset can be found on [[HuggingFace]](https://huggingface.co/datasets/pkulium/I-ScienceQA)

## Repository Structure

```
vlm_robustness/
├── scripts/                  # Evaluation scripts for different models
│   ├── run_hf.sh            # Run HuggingFace models (LLaVA variants)
│   ├── run_image.sh         # Run image distraction experiments
│   ├── run_text.sh          # Run text distraction experiments
│   ├── run_large.sh         # Run large model experiments
│   └── run_prompting.sh     # Run prompting strategy experiments
├── src/
│   ├── lmms_eval/           # Main evaluation framework
│   │   ├── tasks/           # Task implementations
│   │   │   └── I-scienceqa/ # ScienceQA with distractions
│   │   ├── models/          # Model implementations
│   │   └── evaluator.py     # Core evaluation logic
│   ├── cogvlm/              # CogVLM specific experiments
│   ├── tools/               # Utility tools
│   │   └── lite/            # Data processing tools
│   └── LLaMA-Factory/       # Finetune model code
└── logs/                    # Evaluation results (created during runs)
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-compatible GPU (recommended: 80GB+ VRAM for large models)
- PyTorch >= 2.1.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vlm_robustness.git
cd vlm_robustness
```

2. Install dependencies:
```bash
cd src
pip install -e .
```

This will install the `lmms_eval` package with all required dependencies including:
- accelerate >= 0.29.1
- transformers == 4.39.2
- torch >= 2.1.0
- torchvision >= 0.16.0
- datasets == 2.16.1
- And other dependencies specified in `pyproject.toml`

## Evaluation Tasks

The benchmark introduces four types of distractions to test VLM robustness:

### Visual Distractions
- **Add Image**: Add irrelevant images from different domains
- **Insert Image**: Insert distractor images into the original context
- **Add Image (Original)**: Add images from the same domain but different questions
- **Insert Image (Original)**: Insert domain-relevant but question-irrelevant images

### Textual Distractions
- **Add Hint**: Add misleading textual hints
- **Insert Hint**: Insert incorrect hints into the question
- **Add Hint (Original)**: Add hints from other questions in the dataset
- **Insert Hint (Original)**: Insert domain-relevant but misleading hints

## Running Experiments

### Quick Start

To evaluate a model on visual distractions:
```bash
bash scripts/run_image.sh
```

To evaluate a model on textual distractions:
```bash
bash scripts/run_text.sh
```

### Detailed Usage

The evaluation framework uses the `lmms_eval` command-line interface with the following structure:

```bash
accelerate launch --num_processes=8 -m lmms_eval \
    --model <model_name> \
    --model_args pretrained="<model_path>" \
    --tasks <task_name> \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix <suffix> \
    --output_path ./logs/
```

### Supported Models

The benchmark supports evaluation of multiple state-of-the-art VLMs:

- **LLaVA**: v1.5-7b, v1.5-13b, v1.6 variants
- **InstructBLIP**: vicuna-7b, vicuna-13b
- **InternVL2**: 2B, 8B
- **MiniCPM-V**
- **Phi3-V**
- **Fuyu**
- **GPT-4V** (requires API key)

### Example: Running LLaVA-7B on Image Distractions

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks I-ScienceQA_add_image,I-scienceqa_insert_image \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_7b_image_distractions" \
    --output_path ./logs/
```

### Custom Task Configuration

Task configurations are defined in YAML files under `src/lmms_eval/tasks/I-scienceqa/`. You can modify these files to adjust:
- Distraction types and sources
- Prompt templates
- Few-shot examples
- Evaluation metrics

## Results

Results are saved in the `logs/` directory with the following structure:
- JSON files containing detailed evaluation metrics
- Sample outputs for qualitative analysis
- Model-specific performance breakdowns

## Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@article{liu2025robustness,
  title={On the robustness of multimodal language model towards distractions},
  author={Liu, Ming and Chen, Hao and Wang, Jindong and Zhang, Wensheng},
  journal={arXiv preprint arXiv:2502.09818},
  year={2025}
}
```


## License

This project is released under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors through the paper's correspondence email.

## Acknowledgments

This work builds upon the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework for multimodal model evaluation. We thank the original authors for their contributions to the community.
