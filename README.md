# GANFingerprint Deepfake Detection

A deep learning model for detecting GAN-generated deepfake images by analyzing subtle fingerprint patterns in both spatial and frequency domains.

## What are GAN Fingerprints?

GAN Fingerprints are distinctive patterns or traces that are unintentionally embedded in images generated by Generative Adversarial Networks (GANs). These GAN fingerprints are akin to real human fingerprints, with the comparison that humans unintentionally leave fingerprints on the items they touch, that can be used to trace their identities. Just like human fingerprints, these GAN Fingerprints are unique to the GAN architecture the images are generated from, due to these factors:
1.	 Each GAN architecture has its own unique way of generating images based on its specific design, loss functions, and optimization methods. 
2.	Even GANs with identical architectures but different training datasets, random initializations, or hyperparameters will produce images with subtly different characteristics. 

## Objective of the project

With GAN image generation images getting more advanced, there may be difficulties identifying deepfake images through existing methods, such as detecting distortions in facial features and image details. Through our project, we hope to create a deepfake detection model that can identify deepfake images reliably, no matter how realistic the generated images are to the human eye. By customizing and creating a model that can discriminate deepfake images from real ones through their GAN Fingerprint profiles, we hope to come up with a more sophisticated model which can capture details invisible to the human eye. 

# Technical Overview of the Model

## Features

- Multi-path feature extraction from different network levels
- Frequency domain analysis to detect GAN fingerprint artifacts
- Self-attention mechanism for focusing on discriminative regions
- Highly reproducible results with deterministic implementation
- Extensive logging of model performance and used hyperparameters
- Intuitive visualization of model performance in evaluation (Performance metrics, ROC curve, Precision Recall Curve, Confusion Matric)
- Detailed output for image inference, support for batch mode (Images with Grad-CAM heatmaps, csv files for inference results and performance metrics on batch mode)

## Model Architecture

The GANFingerprint model uses a multi-path feature extraction approach with specialized components:

- **Backbone**: ResNet34 separated into low, mid, and high-level feature extractors
- **Feature Fusion**: Combines spatial features from all levels with frequency domain analysis
- **Fingerprint Blocks**: Specialized layers with frequency awareness and spatial attention
- **Enhanced Classifier**: Multi-layer classifier with residual connections

## Directory Structure

```
deepfake_detector/
├── config.py                 # Configuration parameters
├── data_loader.py            # Dataset and dataloader implementation
├── models/
│   ├── __init__.py           # Module initialization
│   ├── fingerprint_net.py    # GANFingerprint model architecture
│   ├── layers.py             # Custom layers and blocks
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference on new images
├── GANFingerprint.ipynb      # Jupyter notebook file with step-by-step guidance on how to run the model
├── utils/
│   ├── __init__.py           # Utilities module initialization
│   ├── reproducibility.py    # Random seed and reproducibility utilities
│   ├── visualization.py      # Plotting and visualization tools
│   ├── metrics.py            # Performance metrics calculation
│   ├── augmentations.py      # Advanced augmentation techniques
|   ├── experiment.py         # Logging of information when training model
|   ├── gradcam.py            # Grad-CAM visualization of inference results
├── checkpoints/              # Directory for saved model checkpoints
├── logs/                     # TensorBoard logs and training records
├── requirements.txt          # Contains all required dependencies 
```

## Installation

### Prerequisites

- Python 3.7+ (model tested on Python 3.12.4)
- CUDA-compatible GPU (model tested on Cuda 11.8)
**cpu is not recommended for this model, as it would take very long to train**

## Requirements

This project requires the following dependencies, saved as requirements.txt:
```
torch>=1.13.0
torchvision>=0.14.0
torchaudio>=0.13.0
numpy>=1.20.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
Pillow>=9.0.0
tqdm>=4.62.0
pandas>=1.3.0
tensorboard>=2.10.0
protobuf>=3.19.0,<4.0.0
```

## How to run the model

The Jupyter Notebook file **GANFingerprint.ipynb** is created with instructions and executable cells, making the process simple to follow.

However, it is also possible to run the model through bash commands. The instructions below cover the full process.

### Option 1: Using pip

1. Clone the repository:
```bash
git clone https://github.com/LZ-sudo/deepfake-detection-GANFingerprint.git
cd deepfake-detection-GANFingerprint
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv ganfingerprint-env

# Activate on Linux/Mac
source ganfingerprint-env/bin/activate

# Activate on Windows
ganfingerprint-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Clone the repository:
```bash
git clone https://github.com/LZ-sudo/deepfake-detection-GANFingerprint.git
cd deepfake-detection-GANFingerprint
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate ganfingerprint
```

## Dataset Preparation

Prepare your dataset with the following structure:
```
data/
├── train/
│   ├── real/   # Real images
│   └── fake/   # Fake/deepfake images
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Configuration

Review and update `config.py` to match your environment:

The parameters below are the parameters used to create the following performance metrics:

```
Accuracy: 0.9476
Precision: 0.9546
Recall: 0.9392
F1-Score: 0.9468
AUC-ROC: 0.9890
```

```python
# Key parameters to check
DATA_ROOT = "data"  # Path to your dataset directory
INPUT_SIZE = 256    # Input image size
BACKBONE = "resnet34"  # Feature extractor backbone
BATCH_SIZE = 16     # Adjust based on your GPU memory
NUM_WORKERS = 10     # Number of data loading workers
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20
DEVICE = "cuda"     # Use "cuda" for GPU, "cpu" for CPU
SEED = 42           # Random seed for reproducibility
```

## Usage

### Training

```bash
# Basic training with default parameters
python train.py

# With custom parameters
python train.py --data_root /path/to/data --batch_size 32 --lr 0.0003 --epochs 30 --backbone resnet34 

# Disable mixed precision (for older GPUs)
python train.py --no_amp

# Resume from a checkpoint
python train.py --resume_checkpoint checkpoints/[pth_file_name]
```

### Monitoring Training

```bash
# Start TensorBoard server
tensorboard --logdir logs
```

Then open your browser at http://localhost:6006 to monitor training progress.

### Evaluation

```bash
# Evaluate the model on the test set
python evaluate.py --checkpoint checkpoints/[pth_file_name] --output_dir eval_results
```

### Inference

```bash
# For a single image
python inference.py --checkpoint checkpoints/[pth_file_name] --input path/to/image.jpg 

# For a directory of images
python inference.py --checkpoint checkpoints/[pth_file_name] --input path/to/images_dir --batch
```

## Reproducibility

The model is designed with reproducibility in mind:
- Fixed seeds for Python, NumPy, and PyTorch
- Deterministic operations in PyTorch
- Random state preservation in checkpoints
- Deterministic data loading

The dataset used to train the model is the 'deepfake and real images' dataset by Manjil Kariki.

Link: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

This dataset provides a large dataset of real and deepfake images, split into train, test and validation sets, making it one of the best datasets to be used for deepfake classification models.

As long as you use the same dataset and compatible library versions, you should get consistent results across different runs.

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in `config.py`
   - Try a lighter backbone (e.g., "resnet18")
   - Use smaller input image size

2. **Slow Training**:
   - Ensure you're using GPU acceleration
   - Increase `NUM_WORKERS` for faster data loading
   - Enable mixed precision with `USE_AMP = True`

3. **Library Compatibility Issues**:
   - If you encounter errors with newer PyTorch versions, try downgrading to 1.8.0-1.9.0
   - For any missing function errors, check the library documentation for version compatibility

## Development and Testing Device, IDE

This model was developed and tested on the **Lenovo Legion 5 Pro (2022)** laptop with the following specs:

### Computer Specs
**12th Gen Intel(R) Core(TM) i7-12700H   2.30 GHz**

**32.0GB of DDR5 RAM**

**Nvidia RTX 3070Ti GPU (mobile)**

### OS and IDE information
**OS: Windows 11**

**IDE: Visual Studio Code Version 1.97.0**

## Citation

If you use this code in your research, please cite my project:

```
@misc{ganfingerprint2025,
  author = {Chow Liang Zhi},
  title = {GANFingerprint: Deepfake Detection via GAN Fingerprint Analysis},
  year = {2025},
  publisher = {Chow Liang Zhi},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LZ-sudo/deepfake-detection-GANFingerprint.git}}
}
```
