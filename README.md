# Handwritten Math Solutions

This project provides tools and models for processing and recognizing handwritten mathematical expressions. It includes utilities for dataset preparation, image processing, and mathematical symbol recognition.

## Project Structure

```
handwrittenmathsolutions/
├── data/                    # Dataset and data processing scripts
│   ├── raw/                 # Raw dataset files
│   ├── processed/           # Processed dataset files
│   └── labels/              # Label files and CSV data
├── src/                     # Source code
│   ├── preprocessing/       # Image preprocessing scripts
│   ├── labeling/           # Labeling utilities
│   └── detection/          # Detection and recognition code
├── model/                   # Trained models and model-related code
└── docs/                    # Documentation and resources
```

## Features

- Image preprocessing and size normalization
- Dataset labeling and organization
- Mathematical symbol detection and recognition
- CSV-based label management
- PDF to LaTeX conversion support

## Requirements

- Python 3.x
- Required Python packages (to be listed in requirements.txt)

## Usage

1. Dataset Preparation:
   ```bash
   python src/preprocessing/sizeFixer.py
   python src/labeling/labeling.py
   ```

2. Model Training:
   ```bash
   python src/detection/train.py
   ```

3. Recognition:
   ```bash
   python src/detection/detect.py
   ```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add your contact information here]
