from __future__ import annotations
from pathlib import Path
import joblib
import mlflow
from data_ingestion import DataIngestion
from preprocessing import DataPreprocessing
from training import ModelTrainer
from evaluation import ModelEvaluator


class CreditScorePipeline:
    MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]
    
    def __init__(
        self,
        input_path: str | Path = None, #path raw dataset
        output_dir: str | Path = None, #path hasil dari ingestion
        model_dir: str | Path = None, #folder untuk simpan model
        test_size: float = 0.2,
        random_state: int = 42,
        mlflow_tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "credit_score_classification", # nama eksperimen pada MLflow
    ):
		#Memastikan folder model simpan .pkl exist
        base_dir = Path(__file__).parent
        self.model_dir = Path(model_dir) if model_dir else base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

		#set MLflow agar bisa simpan hasil training
        mlflow.set_tracking_uri(mlflow_tracking_uri)

		#Define Object
        self.ingestion = DataIngestion(input_path=input_path, output_dir=output_dir)
        self.preprocessing = DataPreprocessing(test_size=test_size, random_state=random_state)
        self.evaluator = ModelEvaluator()

        self.experiment_name = experiment_name

    def run(self):
        
        # 1. Ingestion
        df = self.ingestion.ingest()

        # 2. Preprocessing
        X_train, X_test, y_train, y_test, preprocessor = self.preprocessing.run(df)

        # 3. Training (3 model dengan hyperparameter optimal, dicatat ke MLflow)
        fitted_models = {}
        
		# Membuat object trainer
        for model_name in self.MODEL_NAMES:
            trainer = ModelTrainer(
                model_name = model_name,
                experiment_name = self.experiment_name,
                artifact_path = str(self.model_dir),
            )
            # Menyimpan run_id dan model dari hasil training
            run_id, model_pipeline = trainer.run(preprocessor, X_train, y_train)
            fitted_models[model_name] = (model_pipeline, run_id)

        # 4. Evaluation & perbandingan model
        comparison_df = self.evaluator.evaluate_all(fitted_models, X_test, y_test)
        
        print("--- Step 5: Perbandingan Model ---")
        print(comparison_df.to_string(index=False))
        print(f"\n🏆 Model terbaik: {comparison_df.iloc[0]['Model']}")
        
		
def main():
	input_path = None #di assign None untuk di proses pada Ingestion
	pipeline = CreditScorePipeline(input_path = input_path)
	pipeline.run()


if __name__ == "__main__":
    main()