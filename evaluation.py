from __future__ import annotations
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,f1_score,classification_report,precision_score, recall_score)


class ModelEvaluator:
    TARGET_NAMES = ["Poor", "Standard", "Good"]

    def __init__(self):
        #List akurasi Model
        self.results_: list[dict] = []

	# Method Evaluasi 1 Model
    def evaluate_model(self,model_name: str,fitted_model,X_test: pd.DataFrame,y_test: pd.Series,run_id: str = None,) -> dict:
        
        
        print(f"--- Step 4: Evaluasi Model [{model_name}] ---")

        y_pred = fitted_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average="macro")
        recall_macro = recall_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, target_names=self.TARGET_NAMES)
        print(f"\nClassification Report [{model_name}]")
        print(report)
        
        if macro_f1 < 0.65:
            print(f"❌ Model [{model_name}] DITOLAK ---- Macro F1 ({macro_f1:.4f}) < 0.65.\n")
            return None
    
        result = {
            "Model": model_name,
            "Accuracy": acc,
            "Macro F1": macro_f1,
        }
        self.results_.append(result)
        
        if run_id is not None:
               with mlflow.start_run(run_id = run_id):
                     mlflow.log_metric("test_accuracy", acc)
                     mlflow.log_metric("test_macro_f1", macro_f1)
                     mlflow.log_metric("test_precision_macro", precision_macro)
                     mlflow.log_metric("test_recall_macro", recall_macro)
                     mlflow.log_metric("test_f1_weighted", f1_weighted)
        print(f"✅ Evaluasi [{model_name}] selesai — Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}\n")
        return result

	# Method Evaluasi semua Model pada fitted_models
    def evaluate_all(self,fitted_models: dict,X_test: pd.DataFrame,y_test: pd.Series,) -> pd.DataFrame:
                
        self.results_ = []
        for model_name, (fitted_model, run_id) in fitted_models.items():
            self.evaluate_model(model_name, fitted_model, X_test, y_test, run_id)

        return self.get_comparison_table()
    
	# Method Untuk mendapatkan Tabel Komparasi Hasil
    def get_comparison_table(self) -> pd.DataFrame:
        results_df = pd.DataFrame(self.results_).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
        return results_df

	# Method Untuk mendapatkan model terbaik
    def get_best_model_name(self) -> str:
        comparison = self.get_comparison_table()
        if comparison.empty:
            raise RuntimeError("Belum ada model yang dievaluasi.")
        return comparison.iloc[0]["Model"]