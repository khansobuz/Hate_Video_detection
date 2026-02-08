# CL-ReGAF
**Continual Learning with Retrieval-Guided Graph Attention Fusion for Adaptive Multimodal Hate-Video Detection**

---

## Abstract
Hate-video detection is crucial for online content moderation, yet the dynamic and multimodal nature of online videos—combining text, visual, and audio signals—makes it a challenging task. Existing approaches struggle to adapt to emerging hateful patterns over time. CL-ReGAF addresses this problem by combining continual learning with retrieval-guided graph attention fusion, enabling models to incrementally adapt to new hate trends while preserving previously learned knowledge. 

Key innovations include stabilizing multimodal embeddings through frozen adaptive encoding, constructing retrieval-refined relational graphs for context-aware reasoning, prompt-calibrated continual memory learning to prevent forgetting, and uncertainty-aware modality fusion to balance contributions from text, audio, and visual modalities. Extensive experiments demonstrate that CL-ReGAF outperforms state-of-the-art baselines in accuracy, adaptability, and robustness on **HateMM**, **MultiHateClip-YouTube**, and **MultiHateClip-Bilibili** datasets.

---
Framwork:


## Repository Structure

data/ # Dataset metadata and splits
├── HateMM # HateMM dataset
└── MultiHateClip # MultiHateClip datasets (YouTube and Bilibili)
├── en # English subset
└── zh # Chinese subset

retrieval/ # Retrieval-related modules

src/ # Core implementation of CL-ReGAF
├── config # Training and experiment configuration files
├── model # Model architectures
├── utils # Helper functions for training and evaluation
├── data # Dataset loaders for MoRE
└── CL_EXP/ # Experiments for continual learning evaluation

---

## Dataset
We provide **video IDs** and official data splits for reproducibility. Due to copyright restrictions, raw videos are **not included**.

### Dataset Splits
- Training: 70%
- Validation: 10%
- Test: 20%  
Temporal and five-fold splits are included.

### Dataset Sources
- **HateMM**  
  [https://github.com/hate-alert/HateMM](https://github.com/hate-alert/HateMM)

- **MultiHateClip (YouTube & Bilibili)**  
  [https://github.com/Social-AI-Studio/MultiHateClip](https://github.com/Social-AI-Studio/MultiHateClip)  
  Official repository for ACM Multimedia’24 paper: “MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili”

---

## Environment Setup
We recommend **Python 3.12**:

```bash
conda create --name py312 python=3.12
conda activate py312

pip install torch transformers tqdm loguru pandas \
            torchmetrics scikit-learn colorama \
            wandb hydra-core
Data Preprocessing
The preprocessing pipeline extracts multimodal features from raw videos. Ensure dataset paths are correctly configured.

Steps
Sample 16 frames uniformly from each video

Extract on-screen text using PaddleOCR

Transcribe audio using Whisper-v3

Encode visual features using a pre-trained ViT model

Extract audio features (MFCC) using librosa

Encode textual features using a pre-trained BERT model

cd preprocess
Refer to preprocessing scripts for detailed configuration.

Source Code Structure Summary
data/ → dataset splits and video IDs

retrieval/ → retrieval modules

src/config/ → experiment configurations

src/model/ → model implementations

src/utils/ → helper functions

src/data/ → dataset loaders

src/CL_EXP/ → experiments for continual learning evaluation
