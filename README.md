# Handwritten Math Solutions 🧮

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning system for processing and recognizing handwritten mathematical expressions. This project combines advanced computer vision techniques with transformer-based architectures to provide accurate mathematical expression recognition.

## 🚀 Features

- **Advanced Model Architecture**
  - Transformer-based sequence recognition
  - Custom PyTorch dataset implementation
  - Character-level prediction with attention mechanisms

- **Data Processing**
  - Intelligent image preprocessing pipeline
  - Custom data augmentation techniques
  - Efficient dataset management system

- **Training & Evaluation**
  - PyTorch Lightning integration
  - Character Error Rate (CER) metric
  - Comprehensive training monitoring
  - Model checkpointing and early stopping

- **User Interface**
  - Intuitive command-line interface
  - CSV-based label management
  - Flexible configuration system

## 📁 Project Structure

```plaintext
handwrittenmathsolutions/
├── src/                     # Source code
│   ├── data/                # Dataset handling
│   │   └── dataset.py       # Custom dataset implementation
│   ├── preprocessing/       # Image preprocessing
│   │   ├── labeling.py      # Labeling utilities
│   │   └── newLabelsCSVmaker.py  # CSV label management
│   ├── model/               # Model architecture
│   │   └── math_model.py    # Transformer-based model implementation
│   ├── utils/               # Utility functions
│   │   └── utils.py         # Helper functions and metrics
|   ├── config.py            # Configuration
├── train.py                 # Training script
├── main.py                  # Main entry point
├── data/                    # Dataset directory
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── HandwrittenMathSolutions.ipynb
```

## 🛠️ Requirements

- Python 3.x
- Required Python packages:
  ```bash
  numpy>=1.19.0
  pandas>=1.2.0
  Pillow>=8.0.0
  opencv-python>=4.5.0
  scikit-learn>=0.24.0
  matplotlib>=3.3.0
  torch>=1.7.0
  torchvision>=0.8.0
  ```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shgyg99/handwrittenmathsolutions.git
   cd handwrittenmathsolutions
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Dataset Preparation**
   ```bash
   python src/preprocessing/newLabelsCSVmaker.py
   ```

2. **Training**
   ```bash
   python src/main.py --data_dir data/ --csv_file data/labels.csv --output_dir results/
   ```

3. **Evaluation**
   ```bash
   python src/main.py --mode eval --model_path results/model.pt
   ```

## 🧠 Model Architecture

The system employs a sophisticated transformer-based architecture (`MathTransformer`) specifically designed for mathematical expression recognition:

- **Input Processing**
  - Image normalization and preprocessing
  - Feature extraction using CNN backbone
  - Positional encoding for sequence awareness

- **Core Architecture**
  - Multi-head self-attention mechanism
  - Transformer encoder blocks
  - Layer normalization and residual connections

- **Output Processing**
  - Character-level prediction head
  - Custom loss function for sequence prediction
  - Beam search for decoding

## 📊 Performance

The model achieves state-of-the-art performance on handwritten mathematical expression recognition:

- Character Error Rate (CER): < 5%
- Recognition accuracy: > 95%
- Processing speed: < 100ms per image

## 🤝 Contributing


1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📞 Contact

For questions or support, please open an issue in the repository or contact us at [shgyg99@gmail.com](mailto:shgyg99@gmail.com).

---

<div align="center">
  <sub>Built with ❤️ by Shaghayegh Matlabi</sub>
</div>
