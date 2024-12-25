
# DermAI-Vision: Deep Learning for Skin Lesion Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-20.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI/CD](https://github.com/rodrigoguedes09/DermAI-Vision/actions)

</div>

## About

DermAI-Vision is an advanced deep learning system designed to assist in the detection and classification of skin lesions. Using state-of-the-art convolutional neural networks and the ISIC dataset, the system achieves high accuracy in distinguishing between benign and malignant skin lesions.

### Key Features

- Real-time skin lesion analysis
- User-friendly web interface
- RESTful API for integration
- Advanced CNN architecture with attention mechanisms
- Comprehensive visualization of model decisions
- Docker support for easy deployment

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- NVIDIA GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/DermAI-Vision.git
cd DermAI-Vision
```

2. **Using Docker (Recommended)**
```bash
make setup
```

3. **Manual Installation**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Usage

### Web Interface

Access the web interface at `http://localhost:80`

1. Upload an image of a skin lesion
2. Get instant analysis results
3. View detailed visualization of the model's decision

### API

Access the API at `http://localhost:80/api`

```python
import requests

def analyze_image(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:80/api/predict',
            files={'file': f}
        )
    return response.json()
```

## Architecture

The project uses a custom CNN architecture with:
- Residual connections
- Squeeze-and-Excitation blocks
- Attention mechanisms
- Advanced data augmentation

```
Input Image → CNN → Feature Extraction → Attention → Classification
```

## Performance

- Accuracy: 87.25%
- Sensitivity: 85.3%
- Specificity: 89.1%
- AUC-ROC: 0.912

## 🛠Development

### Running Tests
```bash
make test
```

### Building Documentation
```bash
make docs
```

### Monitoring
```bash
make monitor
```

## Project Structure

```
DermAI-Vision/
├── src/                # Source code
├── data/               # Dataset handling
├── models/             # Trained models
├── docker/             # Docker configuration
├── tests/              # Unit tests
└── docs/               # Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ISIC Archive](https://www.isic-archive.com/) for the dataset
- [PyTorch](https://pytorch.org/) team for the amazing framework

## Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for medical advice.
