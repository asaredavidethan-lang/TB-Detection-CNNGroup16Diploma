# Pulmonary TB Detection System

A web-based AI tool for detecting Tuberculosis from chest X-rays using Convolutional Neural Networks (CNN).

## Features
- Upload chest X-ray images (JPG, PNG, DICOM)
- AI-powered TB screening with confidence scores
- Visual attention maps (Grad-CAM) for interpretability
- FastAPI backend with PyTorch EfficientNet

## Requirements
- Python 3.10+
- PyTorch
- FastAPI
- OpenCV, Pillow, Pydicom

## Installation

1. Get full files from developers


2. Create virtual environment:
Bash

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


3. Install dependencies:
Bash

pip install -r requirements.txt


Dataset Setup (Optional - for training)
Organize your chest X-ray dataset as:

text

data/
├── normal/          # Normal X-ray images
└── tuberculosis/    # TB positive images


Usage
Quick Start (Demo Mode)
Run without training (uses image statistics for variable predictions):

Bash

python -m uvicorn app.main:app --reload
Open browser: http://localhost:8000



With Trained Model
Train the model:
Bash

python train.py
Run the app:
Bash

python -m uvicorn app.main:app --reload



Project Structure
text

├── app/
│   ├── main.py          # FastAPI backend
│   ├── model.py         # CNN architecture
│   ├── preprocessing.py # Image handling
│   └── gradcam.py       # Visualization
├── data/                # Training images (optional)
├── models/              # Saved model weights
├── static/              # CSS/JS files
├── templates/           # HTML templates
├── train.py             # Training script
└── requirements.txt


Note
This is an educational/demo project. For production medical use, additional validation, HIPAA compliance, and clinical testing are required.


# EXAMPLE