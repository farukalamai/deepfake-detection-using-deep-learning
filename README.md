# Deep Learning-Based Real/Fake Image Detection

This project implements three deep learning models (EfficientNet-B7, Inception-V3, and ResNet50) to detect real and fake images.

## Models Used

- **EfficientNet-B7**: Latest iteration of the EfficientNet family, optimized for both accuracy and computational efficiency
- **Inception-V3**: Deep CNN architecture with inception modules for multi-scale processing
- **ResNet50**: 50-layer residual network architecture with skip connections

## Dataset

- **Source**: HuggingFace dataset
- **Size**: 600 images
- **Classes**: 2 (Real/Fake)
- **Split**: Training/Validation/Test (standard split ratios)

## Project Structure

```
├── models/
│   ├── efficientnet_b7/
│   ├── inception_v3/
│   └── resnet50/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── requirements.txt
```

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
efficientnet-pytorch
pillow
numpy
matplotlib
scikit-learn
```

## Installation

```bash
git clone [repository-url]
cd image-detection
pip install -r requirements.txt
```

## Usage

1. **Training**:
```bash
python src/train.py --model [efficientnet_b7|inception_v3|resnet50] --epochs 100
```

2. **Evaluation**:
```bash
python src/evaluate.py --model [model_name] --weights path/to/weights
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| EfficientNet-B7 | XX% | XX% | XX% | XX% |
| Inception-V3 | XX% | XX% | XX% | XX% |
| ResNet50 | XX% | XX% | XX% | XX% |

## Future Improvements

1. Increase dataset size
2. Implement ensemble methods
3. Add data augmentation techniques
4. Fine-tune hyperparameters

## License

[Add your license information]

## Contact

[Add your contact information]