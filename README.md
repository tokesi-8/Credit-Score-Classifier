# Credit Score Prediction

This repository contains an **end-to-end Machine Learning project** for classifying customer credit scores into three categories: **Poor, Standard, or Good**. The prediction is based on customer demographic, financial, and payment behavior data.

The trained model is also deployed as a **Streamlit web application** for real-time credit score prediction.

# Project Overview

This system is designed to make the credit risk assessment process faster and more consistent. By automatically predicting a customer's credit profile, the system can provide an early indication of potential credit risk.

Based on the **Macro F1-Score** evaluation during the Exploratory Data Analysis (EDA) stage, **LightGBM** was selected as the best classification model implemented in this project.

# Application Architecture

The project uses an end-to-end Machine Learning pipeline:

* **Data Ingestion & Preprocessing** — Loads the raw data, cleans invalid data, handles missing values, and performs feature engineering to prepare the data for the model.
* **Machine Learning Model** — Uses modular scripts for model training and experiment tracking with MLflow. The trained models are evaluated, and the selected LightGBM model is saved as a `.pkl` file for later use.
* **Streamlit Application** — Provides an interactive web interface where users can enter customer information and receive a credit score prediction directly.

## Repository Structure

```text
Credit-Score-Classifier/
├── EksplorasiDataDanModelling.ipynb  # EDA, model comparison, and hyperparameter tuning
├── data_ingestion.py                 # Loads and prepares the raw data
├── preprocessing.py                  # Data cleaning and feature engineering
├── training.py                        # Model training
├── evaluation.py                      # Model performance evaluation
├── inference.py                       # Model inference testing
├── pipeline.py                        # End-to-end ML pipeline
├── app_streamlit.py                   # Main Streamlit application
├── models/                            # Saved model files
├── mlruns/                            # MLflow experiment tracking
└── requirements.txt                   # Required Python libraries
```

# Running the Application

The application can be run locally. Make sure the model file `models/model_lightgbm.pkl` is available before starting the application.

```bash
streamlit run app_streamlit.py
```

Open the application in your browser:

```text
http://localhost:8501
```

# How the Prediction Works

In the Streamlit application, users enter customer information through several sections:

* **Income & Demographic Information**
* **Bank Account & Credit Card Information**
* **Loan & Payment History**

After the form is submitted, the input data is converted into a DataFrame and prepared using the same format used during model training.

The LightGBM model then performs the prediction and calculates the class probabilities. The final result is displayed as one of three credit score categories:

**Poor · Standard · Good**
