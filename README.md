# Breaking the Generator Barrier: Disentangled Representation for Generalizable AI-Text Detection

Detecting AI-generated text in open-world scenarios is fundamentally challenged by the rapid emergence of unseen generators, where model-specific artifacts undermine the generalization of existing detectors. Most approaches rely on identifying "fingerprints" of known Large Language Models (LLMs), ignoring the latent semantic discrepancies shared across different generators. In this work, we challenge this reliance on surface-level shortcuts and propose a progressively structured framework that explicitly disentangles AI-detection semantics from generator-aware artifacts. By employing compact latent encoding and perturbation-based regularization, our method minimizes semantic entanglement and aligns representations with task objectives. This design enables robust detection across diverse, unseen models without sacrificing sensitivity to AI-specific cues. Experiments on the MAGE benchmark, covering 20 representative LLMs, demonstrate consistent improvements over state-of-the-art methods, achieving up to 24.2% accuracy gain. Beyond performance, our approach offers a new perspective on generalizable detection, revealing that stripping away generator-specific noise is key to capturing the universal essence of machine-generated text.

## 🚀 Features

- **Progressive Framework**: Employs a structured approach—starting with compact latent encoding, followed by perturbation-based regularization, and ending with discriminative adaptation—for superior representation learning.
- **Scalable Performance**: Demonstrates strong scalability in open-set scenarios, with performance consistently improving as the diversity of training generators increases, achieving significant gains in accuracy and F1 score.
- **Comprehensive Metrics**: Evaluates performance using multiple metrics (Accuracy, F1, ASR.)


## 📋 Prerequisites

### System Requirements
- Python >= 3.8
- CUDA-compatible GPU (recommended for training)
- Sufficient RAM for processing large datasets

### Dependencies
Instead of installing packages manually, we have provided an environment.yml file for easier setup. You can create the environment and install all dependencies automatically by running:

```bash
conda env create -f environment.yml
```


## 📦 Data Preparation

Prepare your dataset in the following format:
- article
- label (machine/human)
- index (bloom/gpt3.5...)

## 🧰 Base Model

We employ the BERT model initialized with random weights (without pre-trained checkpoints) and conduct full fine-tuning on the downstream task.

## 🚀 Usage

### Training
To train the model, run:

```bash
python train.py
```

### Testing
To evaluate the trained model, run:

```bash
python test.py
```


## 🏗️ Architecture

The system is organized as follows:

```
DRGD/
├── data/                  # Data loading and preprocessing
├── module/                # Model implementations
├── tools/                 # Utilities (metrics, preprocessing, etc.)
├── train.py               # Main entry point
└── test.py                # Inference script
```

## 📊 Evaluation Metrics

The framework includes comprehensive evaluation metrics:
- Accuracy
- F1-score
- ASR (Attack Success Rate)

## 📚 Citation

If you use this code in your research, please cite:

```
@article{pu2026breaking,
  title={Breaking the Generator Barrier: Disentangled Representation for Generalizable {AI}-Text Detection},
  author={Pu, Xiao and Cheng, Zepeng and Yuan, Lin and Wu, Yu and Bi, Xiuli},
  journal={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```


## 🔐 License

This project is licensed under the MIT License - see the LICENSE file for details.
