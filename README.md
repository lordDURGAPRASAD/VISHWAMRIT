# Vishwamrit - Medicinal Leaves Classifier

A web-based tool built to classify medicinal plant leaves from images and provide information about their traditional uses and key compounds. Developed as part of our Smart India Hackathon project.

## Overview
Identifying medicinal plants accurately can be tricky, especially during field research. Vishwamrit simplifies this by letting users upload an image of a leaf to get identification results along with useful botanical details. It also includes a quick search feature to look through plant information in real time.

## Project Structure
- `app.py` - Main Flask server handling API endpoints and UI routing
- `templates/index.html` - Web dashboard and image upload interface
- `model/train_model.py` - Script for training the image classification model (MobileNetV2 architecture)
- `model/class_names.json` - Plant species mapping file
- `requirements.txt` - Project dependencies

## Tech Used
- **Python / Flask** - Backend web server
- **TensorFlow / Keras** - Model architecture and training pipeline
- **JavaScript / HTML / CSS** - Frontend interface and search filtering

## Setup & Running Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/lordDURGAPRASAD/VISHWAMRIT.git](https://github.com/lordDURGAPRASAD/VISHWAMRIT.git)
   cd VISHWAMRIT
