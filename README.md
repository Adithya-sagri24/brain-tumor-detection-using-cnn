# Brain Tumor Detection Using CNN

This project implements a Convolutional Neural Network (CNN) based application to detect and classify brain tumors from MRI scans. The model identifies multiple types of brain tumors with high accuracy, helping in early diagnosis and treatment planning.

---

## Features

- Detects 5 types of brain tumors from MRI images
- Uses CNN architecture implemented with TensorFlow/Keras
- Includes data augmentation for better generalization
- Provides accuracy and loss visualization during training
- Displays predicted tumor type and highlights tumor region on the MRI
- User-friendly interface for easy testing of MRI images

---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Flask or Tkinter (depending on your app interface)
- Git for version control

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Adithya-sagri24/brain-tumor-detection-using-cnn.git
   cd brain-tumor-detection-using-cnn

2.(Optional) Create a virtual environment:

      python -m venv venv
      source venv/bin/activate   # On Windows: venv\Scripts\activate

3.Install the required packages:

    pip install -r requirements.txt

Usage

-Run the main script to start the application:

    python main.py

Follow the on-screen instructions to upload MRI images and get tumor detection results.

Project Structure

    main.py - Main application script

    model.py - CNN model architecture and training

    data/ - Dataset folder (MRI images)

    utils/ - Helper functions for preprocessing and visualization

    requirements.txt - Python dependencies

Author

Adithya Sagri

