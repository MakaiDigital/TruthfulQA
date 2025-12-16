import logging
from pathlib import Path

import pandas as pd


class DataIngestion:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_dataset(self) -> pd.DataFrame:
        """Load TruthfulQA dataset"""

        paths_to_check = [
            self.data_dir / "v0" / "TruthfulQA.csv",
            self.data_dir / "v1" / "TruthfulQA.csv",
            self.data_dir / "TruthfulQA.csv",
        ]

        for path in paths_to_check:
            if path.exists():
                logging.info(f"Loading dataset from: {path}")
                df = pd.read_csv(path)
                logging.info(f"Dataset shape: {df.shape}")
                logging.info(f"Columns: {list(df.columns)}")
                return df

        for file_path in self.data_dir.rglob("TruthfulQA.csv"):
            logging.info(f"Found dataset at: {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Dataset shape: {df.shape}")
            return df

        raise FileNotFoundError(f"TruthfulQA.csv not found in {self.data_dir}")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist"""
        required = ["Question", "Best Answer", "Correct Answers"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            logging.error(f"Missing columns: {missing}")
            logging.error(f"Available columns: {list(df.columns)}")
            return False

        return True
