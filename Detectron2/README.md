# 🧠 Detectron2 Training and Evaluation (WSL + Conda)

This project trains and evaluates a custom object detection model using [Detectron2](https://github.com/facebookresearch/detectron2) with COCO-style datasets, running in **WSL (Ubuntu)** using a **Conda environment**.

---

## ⚙️ Requirements

- Windows with WSL (Ubuntu)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed in WSL
- Python ≥ 3.8

---

## 🧪 Conda Environment Setup

```bash
# Create and activate the environment
conda create -n detectron2_env python=3.10 -y
conda activate detectron2_env

# Install PyTorch with CUDA (adjust based on your GPU/CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install required Python packages
pip install opencv-python matplotlib ipykernel

# Install Detectron2 from source
pip install -U 'git+https://github.com/facebookresearch/detectron2.git'
install requirements.txt

Project structure:
project/
├── test.ipynb                    # Main notebook
├── scarecrow_coco_dataset/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── images/
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── images/
└── output/                       # Will be created to store model checkpoints

Running the Notebook

# From inside WSL terminal
conda activate detectron2_env
jupyter notebook

The notebook performs the following steps:

1. Registers COCO-style training and validation datasets

2. Configures the model using Detectron2

3. Trains using DefaultTrainer

4. Evaluates using COCOEvaluator

5. Optionally visualizes predictions
