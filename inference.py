import json
import os
import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

CLASS_NAMES = {0: "Poor", 1: "Standard", 2: "Good"}
FEATURE_NAMES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
    "Num_Type_of_Loan", "Credit_History_Age_Months",
]

NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "Num_Type_of_Loan", "Credit_History_Age_Months",
]
 
CATEGORICAL_FEATURES = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]

def model_fn(model_dir: str):
    '''Load model dari hasil train pipeline yaitu model_lightgbm.joblib'''
    return joblib.load(os.path.join(model_dir, "model_lightgbm.joblib"))

def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    '''Mengambil request user berupa JSON dan diubah menjadi dataframe untuk prediksi'''
    
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body) #mengubah JSON string -> object python
        instances = payload["instances"] #mengambil isi dari instance
        df = pd.DataFrame(instances, columns = FEATURE_NAMES)

    elif request_content_type == CSV_CONTENT_TYPE:
        # Mengubah bytes -> string (jika diperlukan)
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
    
        # Mengubah CSV string -> list data
        rows = [
            line.split(",")
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        # Mengubah list data -> DataFrame
        df = pd.DataFrame(rows, columns = FEATURE_NAMES)
        
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
        
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
 
    return df
    

def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Menjalankan prediksi dan return hasilnya"""
    
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }

def output_fn(prediction: dict, accept_content_type: str):
    """Mengubah hasil prediksi menjadi respon dalam JSON."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")