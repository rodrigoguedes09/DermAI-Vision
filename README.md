# DermAI-Vision: Deep Learning for Skin Cancer Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-20.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

## About

SkinLesion Analyzer is a deep learning system developed to assist healthcare professionals in the preliminary analysis of skin lesions. The system utilizes advanced convolutional neural networks trained on the ISIC (International Skin Imaging Collaboration) dataset to classify skin lesions into three categories: benign, malignant, or indeterminate.

## Key Features

- Automated classification of skin lesions
- Probability assessment for each category
- Visualization of regions of interest
- Support for high-resolution dermatoscopic images
- Comprehensive reporting system

## Technical Specifications

### Model Architecture
- Custom CNN with attention mechanisms
- Transfer learning using ResNet architecture
- Advanced image preprocessing techniques
- Specialized data augmentation for medical imaging

### Performance Metrics
- Overall Accuracy: 87.25%
- Sensitivity: 85.3%
- Specificity: 89.1%
- AUC-ROC: 0.912

## Installation

### Using Docker (Recommended)
```bash
git clone https://github.com/yourusername/skinlesion-analyzer.git
cd skinlesion-analyzer
make setup
```

### Manual Installation
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Usage

### Web Interface
Access the system through a browser at `http://localhost:80`

The interface provides:
- Image upload functionality
- Real-time analysis results
- Probability distribution for each category
- Visualization of significant regions
- Downloadable reports

### API Integration
```python
import requests

def analyze_lesion(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:80/api/predict',
            files={'file': f}
        )
    return response.json()
```

## Development

### Running Tests

There are a few tests made for model avaliation.

#### GradCAM

Shows the image and a heat map of where the model mostly acts.
![image](https://github.com/user-attachments/assets/7975f2a1-e0f7-4e34-8140-954fdb6c35e7)

#### Streamlit App

Easy way to test the model uploading an image.

![image](https://github.com/user-attachments/assets/04ca0dbd-15d4-40cd-a1d3-fbf4a737d8cd)

![image](https://github.com/user-attachments/assets/8461352a-d261-450d-a93b-727cc26e07af)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Important Notice

This system is designed as a supplementary tool for healthcare professionals and should not be used as the sole basis for diagnosis. The final interpretation and diagnosis should always be made by qualified healthcare providers.
