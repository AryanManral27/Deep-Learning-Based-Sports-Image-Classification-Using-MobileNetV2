# Deep Learning-Based Sports Image Classification Using MobileNetV2

## Overview

This project uses Deep Learning and Transfer Learning to classify sports images into 35 different
sports categories. The model is built using MobileNetV2 and TensorFlow/Keras and is deployed
through a Streamlit web application for real-time predictions.

Users can upload a sports image, and the model will instantly predict the sport category.

---

## Features

✔ Sports Image Classification Across 35 Categories 
✔ Transfer Learning Using MobileNetV2 Architecture  
✔ Image Preprocessing and Data Augmentation Techniques  
✔ Real-Time Prediction Through Streamlit Deployment  
✔ User-Friendly and Interactive Web Interface  
✔ Fast, Accurate, and Lightweight Deep Learning Model  

---

## Technologies Used

- **Python** – Programming language used for development.
- **TensorFlow** – Framework for building and training deep learning models.
- **Keras** – API used to create and manage neural networks.
- **NumPy** – Library for numerical computations and data processing.
- **Matplotlib** – Used for data visualization and performance graphs.
- **Scikit-learn** – Used for model evaluation and performance metrics.
- **Streamlit** – Framework for developing the web application.
- **MobileNetV2** – Pre-trained CNN model used for sports image classification.

---

## Dataset

The model is trained on the Kaggle Sports Classification Dataset.

**Dataset Link:** https://www.kaggle.com/datasets/gpiosenka/sports-classification

---

## Project Structure

```text
sports-image-classification/
│
├── sports-clf-mobilenet.py
├── sports-clf-final-predictions.py
├── utils.py
├── requirements.txt
├── README.md
│
├── streamlit/
│   ├── app.py
│   ├── utils.py
│   └── best_model.h5
│
└── Dataset/
    ├── train/
    ├── valid/
    └── test/
```

---

## Model Architecture

- MobileNetV2 Base Model (Pre-trained on ImageNet)
- Global Average Pooling Layer for Feature Extraction
- Dropout Layer (0.5) for Overfitting Prevention
- Dense Output Layer with Softmax Activation

**Input Image Size:** 224 × 224 × 3

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd sports-image-classification
```

### Create Virtual Environment

```bash
python -m venv venv
```

**Activate Environment:**

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script:

```bash
python sports-clf-mobilenet.py
```

The trained model will be saved as:

```text
best_model.h5
```

Copy it to the `streamlit/` folder before running the app.

---

## Model Evaluation

Run:

```bash
python sports-clf-final-predictions.py
```

The script evaluates the model using Accuracy, Precision, Recall, and F1-Score metrics.

---

## Running the Streamlit App

Start the application:

```bash
streamlit run streamlit/app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## How to Use

1. Upload a sports image in JPG, JPEG, or PNG format.
2. Click **Classify Sport** to start the prediction.
3. The model processes and analyzes the uploaded image.
4. Instantly view the predicted sport category.

---

## Results

✔ Achieved a Test Accuracy of 91%  
✔ Lightweight Model Size (~15 MB)  
✔ Fast Prediction Time (< 5 Seconds)  
✔ Real-Time Sports Image Classification  

---

#### Image 1: Upload a Sports Image for Classification

<p align="center">
  <img width="720" height="350"
       src="https://github.com/user-attachments/assets/4a828d20-bbc6-487e-b4b9-58c4b8b816ab"
       alt="Image">
</p>

---

#### Image 2: Use a Sample Sports Image for Prediction

<p align="center">
  <img width="720" height="350"
       src="https://github.com/user-attachments/assets/599fccda-36b1-4b25-8cbf-e1ecd0698d26" 
       alt="Image">
</p>


---

## Future Improvements

- Real-Time Video Classification Support
- Cloud-Based Deployment and Scalability  
- Support for Additional Sports Categories 
- Explainable AI visualizations (Grad-CAM)

---

## DEEP_LEARNING-BASED_SPORTS_IMAGE_CLASSIFICATION.ipynb

The DEEP_LEARNING-BASED_SPORTS_IMAGE_CLASSIFICATION.ipynb notebook contains the complete pipeline
for training and evaluating the sports image classification model. It includes dataset loading
image preprocessing, data augmentation, MobileNetV2 model development, training, and performance
evaluation. The notebook also provides visualization of test images with actual and predicted
labels, helping to analyze the model's classification accuracy and overall performance.
