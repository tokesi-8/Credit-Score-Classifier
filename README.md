# Credit Score Prediction : End-To-End Machine Learning Deployment

🔗 **Live Dashboard:** [credit-score-prediction-tokesi-8.streamlit.app](https://credit-score-prediction-tokesi-8.streamlit.app/)

- Credit Score Classifier is an end-to-end Machine Learning project that classifies customers into three credit score categories: Poor, Standard, and Good.
- The project starts with EDA and model experimentation in a Jupyter Notebook, followed by the development of a reproducible end-to-end ML pipeline, local inference through Streamlit, and cloud deployment using AWS SageMaker and EC2.
- The final system separates the ML inference service from the user interface, with SageMaker serving the model and EC2 hosting the Streamlit application.


![Dashboard Overview](images/1.png)
![Dashboard Overview](images/2.png)




## Background Problem
- Financial institutions need to assess the credit performance of their customers efficiently and consistently.
- Manual credit assessment can be time-consuming, especially when dealing with a large number of customers. This project applies a data-driven classification approach to automatically assess customer credit profiles based on their financial, credit, and payment behavior.

- The prediction target, Credit_Score, consists of three classes:
  - Poor
  - Standard
  - Good
    
- Because the target classes are not perfectly balanced, Macro F1-Score is used as the primary evaluation metric to measure performance across all classes more evenly.
  
## Project Flow

**EDA & Modeling**
↓
**Model Comparison & Tuning**
↓
**Top 3 Model Selection**
↓
**End-to-End Local Pipeline**
↓
**Local Streamlit Inference**
↓
**AWS SageMaker Deployment**
↓
**SageMaker Endpoint** ← **Streamlit on EC2**
↓
**Final Prediction**

## Development Stages

| Stage                 | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| **EDA & Modeling**    | Explore, clean, transform, and analyze the dataset.             |
| **Model Selection**   | Compare and select the top 3 models based on Macro F1-Score.    |
| **Local Pipeline**    | Build a reproducible end-to-end ML pipeline.                    |
| **Local Inference**   | Integrate the trained model with Streamlit for prediction.      |
| **Cloud Deployment**  | Deploy the selected model through AWS SageMaker.                |
| **Cloud Application** | Host Streamlit on EC2 and connect it to the SageMaker endpoint. |


## Dataset

The dataset contains **24,998 customer records** with **21 features** covering demographic, financial, credit, loan, and payment information.

* **Target:** `Credit_Score`
* **Classes:** Poor, Standard, Good
* **Data split:** 80% training, 20% testing
* **Preprocessing:** Missing-value handling, feature transformation, encoding, scaling, and outlier treatment


## Repository Structure

```text
Credit-Score-Classifier/
│
├── AWS/
│   ├── app_streamlit.py                       # Streamlit application for AWS deployment
│   ├── data_ingestion.py                      # Load and prepare input data
│   ├── evaluation.py                          # Evaluate model performance
│   ├── inference.py                           # Handle model inference
│   ├── mlflow.db                              # MLflow tracking database
│   ├── preprocessing.py                       # Data preprocessing and feature engineering
│   ├── requirements.txt                       # AWS environment dependencies
│   ├── training.py                            # Model training
│   └── user-data.sh                           # EC2 instance initialization script
│
├── Local/
│   ├── models/                                # Trained model files
│   ├── app_streamlit.py                       # Local Streamlit application
│   ├── data_A.csv                             # Dataset
│   ├── data_ingestion.py                      # Load and prepare dataset
│   ├── evaluation.py                          # Evaluate model performance
│   ├── inference.py                           # Generate predictions
│   ├── pipeline.py                            # Run the end-to-end ML pipeline
│   ├── preprocessing.py                       # Data preprocessing and feature engineering
│   └── training.py                            # Model training
│
├── Notebook/
│   └── Data_Exploration_and_Modelling.ipynb    # EDA, model comparison, and tuning
├── images/                                     # README and project screenshots
│  
├── README.md                                   # Project documentation
└── requirements.txt                            # Project dependencies
```

## Running the Application Locally

The application can be run locally. Make sure the model file `models/model_lightgbm.pkl` is available before starting the application.

```bash
streamlit run app_streamlit.py
```

Open the application in your browser:

```text
http://localhost:8501
```

# Tech Stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · Random Forest · MLflow · Streamlit · AWS SageMaker · AWS EC2 · Amazon S3 · CloudWatch

