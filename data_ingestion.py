from __future__ import annotations
from pathlib import Path
import pandas as pd


class DataIngestion:
    
    def __init__(self, input_path = None, output_dir = None):
        base_dir = Path(__file__).parent
        self.input_file = Path(input_path or base_dir / "data_A.csv")
        self.output_dir = Path(output_dir or base_dir / "ingested")
        self.output_file = self.output_dir / self.input_file.name

    def ingest(self): 
        print("--- Step 1: Data Ingestion ---")
        
        # Cek apakah directory output sudah ada, jika belum maka akan dibuat
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read raw data
        df = pd.read_csv(self.input_file)

        # memastikan dataframe tidak kosong
        assert not df.empty, "Dataset Tidak ada!"

        # Save ingested data ke output file 
        df.to_csv(self.output_file, index=False) 
        print(f"✅ Data ingested dari {self.input_file} → {self.output_file}\n\n\n")
        
        return df