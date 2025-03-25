# Brain Tumor Detection using Deep Learning

## Overview
This project is a Flask-based web application that detects brain tumors from MRI images using a deep learning model. The model classifies images into four categories: `pituitary`, `glioma`, `notumor`, and `meningioma`.

## Features
- Upload MRI images for tumor classification.
- Uses a deep learning model (`model.h5`) trained on MRI scans.
- Displays prediction results along with confidence scores.
- Simple and user-friendly web interface.

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- Required Python libraries (listed in `requirements.txt`)

### Steps to Set Up Locally

1. **Clone the Repository**

   - git clone https://github.com/mtechbro94/brain-tumor-detection.git
   - cd brain-tumor-detection

# Project Structure

- **brain-tumor-detection/**
  - **models/**
    - `model.h5` - Pre-trained deep learning model
  - **uploads/** - Stores uploaded images
  - **static/** - Static assets (CSS, JS, images)
  - **templates/**
    - `index.html` - HTML file for web UI
  - `main.py` - Flask app
  - `requirements.txt` - Python dependencies
  - `README.md` - Project documentation

# Technologies Used
- **Flask (Python Web Framework)**

- **TensorFlow/Keras (Deep Learning)**

- **NumPy (Numerical Computation)**

- **Jinja2 (Template Rendering)**