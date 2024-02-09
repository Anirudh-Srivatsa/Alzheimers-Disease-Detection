# Alzheimer's Disease Classification using Convolutional Neural Networks
This project focuses on classifying brain MRI images into four stages of Alzheimer's disease: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. It utilizes Convolutional Neural Networks (CNNs), leveraging TensorFlow and Keras for model building and training. The project includes three different CNN architectures: a basic CNN model, an advanced CNN model with additional layers and regularization techniques, and a ResNet50-based model leveraging transfer learning.

# Installation
## Prerequisites
Python 3.6, 
TensorFlow, 
Keras, 
Scikit-learn, 
Matplotlib, 
Seaborn, 
Numpy. 

# Setup
1. Clone this repository or download the code.
2. Install the required dependencies:
pip install tensorflow keras scikit-learn matplotlib seaborn numpy

Make sure you have access to the dataset. This code assumes you have the Alzheimer's dataset placed in a Google Drive folder, accessible via Google Colab.

# Usage
1. Open the notebook in Google Colab or your preferred Jupyter environment.

2.  Mount your Google Drive to access the dataset:
from google.colab import drive
drive.mount('/content/drive')

3. Set the correct paths to your dataset directories:

  TRAIN_DIR = '/content/drive/MyDrive/Alzheimer_s Dataset/train'

  TEST_DIR = '/content/drive/MyDrive/Alzheimer_s Dataset/test'

4. Run the cells to build and train the models. There are sections for:
Loading and visualizing the data
Building and training the basic CNN model
Building and training the advanced CNN model
Building and training the ResNet50-based model
Evaluate the models using the provided evaluation scripts and visualize the confusion matrices.

# Models
### Basic CNN Model: 
A simple convolutional network designed as an introduction to image classification tasks.

### Advanced CNN Model: 
An enhanced version of the basic CNN with additional convolutional layers, dropout, and batch normalization to improve accuracy.

### ResNet50 Model: 
Leverages transfer learning from a pre-trained ResNet50 model, fine-tuned for the specific task of Alzheimer's disease stage classification.

# Dataset
The dataset used in this project consists of MRI scans of brains categorized into four classes based on the stage of Alzheimer's disease. 

The Kaggle Link - https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

# Acknowledgements
This project was inspired by the need for early and accurate diagnosis of Alzheimer's disease stages to aid in treatment planning and support research into the disease.
