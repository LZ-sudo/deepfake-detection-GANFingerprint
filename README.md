# GANFingerprint Deepfake Detection

A deep learning model for detecting GAN-generated deepfake images by analyzing subtle fingerprint patterns in both spatial and frequency domains.

## Features

- Multi-path feature extraction from different network levels
- Frequency domain analysis to detect GAN fingerprint artifacts
- Self-attention mechanism for focusing on discriminative regions
- Highly reproducible results with deterministic implementation
- State-of-the-art performance on deepfake detection benchmarks

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
├── utils/
│   ├── __init__.py           # Utilities module initialization
│   ├── reproducibility.py    # Random seed and reproducibility utilities
│   ├── visualization.py      # Plotting and visualization tools
│   ├── metrics.py            # Performance metrics calculation
│   ├── augmentations.py      # Advanced augmentation techniques
├── checkpoints/              # Directory for saved model checkpoints
├── logs/                     # TensorBoard logs and training records
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)

## Requirements

This project requires the following dependencies:
```
torch>=1.8.0,<2.0.0
torchvision>=0.9.0,<2.0.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.50.0
Pillow>=8.0.0
tensorboard>=2.4.0
```

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

```python
# Key parameters to check
DATA_ROOT = "data"  # Path to your dataset directory
INPUT_SIZE = 256    # Input image size
BACKBONE = "resnet34"  # Feature extractor backbone
BATCH_SIZE = 64     # Adjust based on your GPU memory
NUM_WORKERS = 4     # Number of data loading workers
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
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
python train.py --resume_checkpoint checkpoints/ganfingerprint_20250408_123456_best.pth
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
python evaluate.py --checkpoint checkpoints/ganfingerprint_20250408_123456_best.pth --output_dir eval_results
```

### Inference

```bash
# For a single image
python inference.py --checkpoint checkpoints/ganfingerprint_20250408_123456_best.pth --input path/to/image.jpg --output inference_results

# For a directory of images
python inference.py --checkpoint checkpoints/ganfingerprint_20250408_123456_best.pth --input path/to/images_dir --output inference_results --batch
```

## Reproducibility

The model is designed with reproducibility in mind:
- Fixed seeds for Python, NumPy, and PyTorch
- Deterministic operations in PyTorch
- Random state preservation in checkpoints
- Deterministic data loading

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

## License

[MIT License](LICENSE)

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
