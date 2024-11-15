# Handwritten-Digit-Recognition-ML-Model
This repository contains a Machine Learning project to recognize handwritten digits from the widely-used MNIST dataset. Using Python and popular ML libraries, this project demonstrates how a model can be trained to accurately predict handwritten digits.

Overview
Handwritten digit recognition is a classic problem in machine learning and computer vision. This project leverages supervised learning algorithms to classify digits (0-9) from images, based on pixel values. The project includes:

Data Preprocessing: Loading and normalizing image data
Model Building: Building and training ML models using scikit-learn and TensorFlow/Keras
Evaluation: Evaluating model performance and accuracy on test data
Dataset
The MNIST dataset is used, containing 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels.

Project Structure
data/ - Folder to store dataset (if not loaded directly from libraries)
notebooks/ - Jupyter notebooks for data exploration, preprocessing, and model training
src/ - Python scripts for loading data, building models, and making predictions
results/ - Saved models and evaluation results
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Preprocess and visualize the data:

bash
Copy code
python src/data_preprocessing.py
Train the model:

bash
Copy code
python src/train_model.py
Evaluate the model:

bash
Copy code
python src/evaluate_model.py
Make Predictions: You can load a saved model and use it to predict digits in new images.

Models Used
This project explores multiple models:

K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Convolutional Neural Network (CNN) using TensorFlow/Keras for higher accuracy
Results
Model	Accuracy
KNN	xx%
SVM	xx%
CNN	xx%
Sample Predictions:
<img src="C:\Users\nauti\OneDrive\Pictures\Screenshots\Screenshot 2024-11-15 130631.png" alt="Sample predictions of handwritten digits" width="500"/>
