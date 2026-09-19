from __future__ import annotations
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelTrainer:    
    # Hyperparameter optimal sesuai hasil RandomizedSearchCV pada notebook.
	MODEL_CONFIGS = {
        "lightgbm": {
            "estimator": LGBMClassifier,
            "params": {
                "num_leaves": 100,
                "n_estimators": 200,
                "max_depth": 9,
                "learning_rate": 0.2,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier,
            "params": {
                "n_estimators": 300,
                "min_samples_split": 5,
                "max_depth": 30,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
        },
        "xgboost": {
            "estimator": XGBClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 9,
                "learning_rate": 0.2,
                "random_state": 42,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "n_jobs": -1,
            },
        },
    }
    
	def __init__(self,
                model_name: str,
              	experiment_name: str = "credit_score_classification",
				artifact_path: str = "artifacts/models",
    			):
				
		if model_name not in self.MODEL_CONFIGS:
			raise ValueError(f"Model '{model_name}' tidak dikenal. Pilihan: {list(self.MODEL_CONFIGS)}")
		
		self.model_name = model_name
		self.experiment_name = experiment_name
		self.artifact_dir = Path(artifact_path) 
		
		# Memastikan artifact ada
		self.artifact_dir.mkdir(parents=True, exist_ok=True)
		mlflow.set_experiment(self.experiment_name)
    
	def run(self, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
			"""Melatih model: build pipeline, fit, lalu simpan ke MLflow dan sebagai file .pkl lewat joblib. """
			
			print(f"--- Step 3: Training Model [{self.model_name}] ---")
	
			config = self.MODEL_CONFIGS[self.model_name]

			# Membuat pipeline preprocessing dan modelling menjadi 1 pipeline
			model_pipeline = Pipeline([
				("preprocessing", preprocessor),
				("classifier", config["estimator"](**config["params"])),
			])
	
			with mlflow.start_run( run_name= self.model_name) as run:
				
				# Log hyperparameters model
				mlflow.log_param("model_name", self.model_name)
				mlflow.log_params(config["params"])
	
				# Training model
				model_pipeline.fit(X_train, y_train)
	
				# Save model local dan log ke MLflow
				model_file_path = self.artifact_dir / f"model_{self.model_name}.pkl"
				joblib.dump(model_pipeline, model_file_path)
				mlflow.sklearn.log_model(model_pipeline, name="model", serialization_format="pickle")
	
				print(f"✅ Model [{self.model_name}] trained & saved locally to {model_file_path}\n")
	
				return run.info.run_id, model_pipeline


    