# heart-disease-predication-webpage

This repository contains the final project of DATA 1200 with an integration of web services and machine learning for predicting heart disease. The project trains **Random Forest**, **SVM** (supervised), and **K-means** (unsupervised cluster) models on a heart disease dataset and hosts them via a RESTful API using Flask. Also included is a professional, interactive webpage where patients can enter patient data and receive predictions.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [References and Resources](#references-and-resources)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [License](#license)

---

## Overview

This project employs three different machine learning algorithms to forecast the likelihood of heart disease among patients:
- **Random Forest Classifier**
- **SVM Classifier**
- **K-means Clustering Model**

There exists a Jupyter Notebook (`train_models.ipynb`) that performs exploratory data analysis (EDA), data pre-processing (including one-hot encoding of categorical variables), models training, and model evaluation, and saves the trained models as pickle files.

The Flask API (`app.py`) uses these models and provides prediction endpoints. An interactive professionally presented front-end web page (`templates/index.html`) where end users enter patient information and receive a diagnosis—with clear output of a heart disease detection (binary result 1 or 0) and a text message.

---

## Project Structure

DATA-1200-02-Heart-Disease-Prediction-WebService/
├── app.py                   # Flask web app
├── train_models.ipynb       # Notebook for training models
├── feature_names.pkl        # .pkl model file
├──kmeans_model.pkl          # .pkl model file
├──random_forest_model.pkl   # .pkl model file
├──svm_model.pkl             # .pkl model file
├── templates/               # HTML templates for the web app
├──heart.csv                 # Dataset for the project
└── README.md                # Project documentation

---

## Features

- **Machine Learning Models:**
  - Supervised prediction with Random Forest and SVM classifiers.
  - Unsupervised analysis with K-means clustering (with mapping to binary diagnosis).
- **RESTful API:**
  Endpoints:
  - `/randomforest/evaluate` – returns binary prediction and diagnosis message.
  - `/svm/evaluate` – returns binary prediction and diagnosis message.
  - `/kmeans/evaluate` – converts cluster output to a binary prediction and a diagnosis message.
- **Interactive Web Interface:**
  A responsive Bootstrap-generated HTML page guides the user through placeholder proposals for input ranges and displays human-readable output (text and binary prediction).
- **Reproducible Environment:**
  All code is version-controlled, and the project folder structure and files fulfill the professor's requirements.

  ---

# Installation and Setup

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Git and VS Code (optional but recommended)
- Required Python packages:
  - Flask
  - NumPy
  - Pandas
  - scikit-learn
  - matplotlib

To install the required packages, run:

```bash
pip install flask numpy pandas scikit-learn matplotlib
```
# Setting Up the Project Locally
## 1. Clone the Repository

```bash
git clone https://github.com/Shriram002/DATA-1200-02-Heart-Disease-Prediction-WebService.git
```
```bash
cd DATA-1200-02-Heart-Disease-Prediction-WebService
```

## 2. Open in VS Code
   Launch the folder in Visual Studio Code or another IDE.

## 3. Train the Models
   Open and run all cells in train_models.ipynb to:
   - Load the dataset
   - Perform data preprocessing & EDA
   - Train Random Forest, SVM, and K-means models
   - Save them as .pkl files
## 4. Run the Flask Web App

```bash
python app.py
```

Then open http://127.0.0.1:5000/ in your browser to interact with the app.


---

# Usage

Follow the steps below to use the web interface:

1. **Start the Flask App**

   Run the command below from your terminal:

   ```bash
   python app.py 
    ```

2. **Open the Web Interface**
   Navigate to http://127.0.0.1:5000/ in your web browser.
   
3. **Enter Input Values**
   - Fill out the patient’s health details.
   - Each field provides a suggested value range.
   - Examples:
     - Age: 30–80
     - RestingBP: 90–200
     - Cholesterol: 100–600
     - Binary fields: 0 = No, 1 = Yes
    
4. **Choose a Model**
   - Select from:
       - Random Forest
       - Support Vector Machine (SVM)
       - K-means Clustering
         
5. **View Results**
   - The prediction result will be displayed clearly.
   - Output includes:
        - Binary value: 1 = Heart Disease, 0 = No Heart Disease
        - Text message (e.g., “Heart Disease Detected”)

# References and Resources

Below is a list of references and resources used in this project:

##  Libraries and Tools

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Bootstrap](https://getbootstrap.com/)

##  Dataset Source

- **Heart Disease Dataset**  
  Source: [Kaggle / UCI Heart Disease](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

##  Code Inspiration

- Official sklearn examples for model training and evaluation
- Tutorials on deploying ML models with Flask and pickle

# Authors and Acknowledgments

##  Team Members

- **Shriram**  
- **Anshul**  
- **Nikita**  
- **Tanya**

Each member contributed equally in areas of:
- Data preprocessing
- Model training
- Web API development
- Frontend design
- Testing and deployment

##  Special Thanks

We would like to thank **Professor Shanti Couvrette** for the DATA-1200-02 course for her continued guidance and support throughout the semester.

# License

This project is licensed under name of Shriram Yadav.

You are free to:
- Use, copy, modify, merge, publish, and distribute the software

Under the following conditions:
- Include the original copyright
- Include the license text in any copies
