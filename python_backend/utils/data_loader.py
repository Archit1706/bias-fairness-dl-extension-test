"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict


class FairnessDataset(Dataset):
    """PyTorch Dataset for fairness analysis."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array
            y: Labels array
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """Preprocess CSV datasets for fairness analysis."""

    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.protected_feature_names = []

    def detect_sensitive_columns(self, columns: List[str]) -> List[str]:
        """
        Auto-detect sensitive/protected attributes.

        Args:
            columns: List of column names

        Returns:
            List of detected sensitive column names
        """
        sensitive_patterns = {
            "gender": r"(gender|sex|male|female)",
            "race": r"(race|ethnicity|ethnic|color)",
            "age": r"(age|birth|dob|year)",
            "nationality": r"(nationality|national|origin|country)",
            "religion": r"(religion|religious)",
            "disability": r"(disability|disabled|handicap)",
        }

        detected = []
        for col in columns:
            col_lower = col.lower()
            for attr_type, pattern in sensitive_patterns.items():
                import re

                if re.search(pattern, col_lower):
                    detected.append(col)
                    break

        return detected

    def load_and_preprocess(
        self,
        file_path: str,
        label_column: str,
        sensitive_features: List[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Dict:
        """
        Load CSV and preprocess for training.

        Args:
            file_path: Path to CSV file
            label_column: Name of label column
            sensitive_features: List of sensitive feature names (auto-detect if None)
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Dictionary with train/val/test loaders and metadata
        """
        # Load data
        df = pd.read_csv(file_path)

        # Auto-detect sensitive features if not provided
        if sensitive_features is None:
            sensitive_features = self.detect_sensitive_columns(df.columns.tolist())

        self.protected_feature_names = sensitive_features

        # Separate features and labels
        feature_columns = [c for c in df.columns if c != label_column]
        self.feature_names = feature_columns

        X = df[feature_columns].copy()
        y = df[label_column].copy()

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Handle missing values
        X = X.fillna(X.mean())

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["features"] = scaler

        # Encode labels if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders["label"] = le

        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size / (1 - test_size),  # Adjust for already split data
            random_state=42,
            stratify=y_train_val,
        )

        # Create datasets (convert Series to numpy arrays)
        train_dataset = FairnessDataset(
            X_train, y_train.values if hasattr(y_train, "values") else y_train
        )
        val_dataset = FairnessDataset(
            X_val, y_val.values if hasattr(y_val, "values") else y_val
        )
        test_dataset = FairnessDataset(
            X_test, y_test.values if hasattr(y_test, "values") else y_test
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Get protected feature indices
        protected_indices = [
            feature_columns.index(f) for f in sensitive_features if f in feature_columns
        ]

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "input_dim": X_scaled.shape[1],
            "num_classes": len(np.unique(y)),
            "protected_indices": protected_indices,
            "protected_features": sensitive_features,
            "feature_names": feature_columns,
            "scaler": scaler,
            "label_encoders": self.label_encoders,
        }


# Test script
if __name__ == "__main__":
    print("Testing DataPreprocessor...")

    # Create dummy CSV
    import tempfile
    import os

    dummy_data = """age,gender,income,education,label
25,male,50000,bachelors,0
30,female,60000,masters,1
22,male,35000,bachelors,0
45,female,80000,phd,1
35,male,55000,masters,0
28,female,45000,bachelors,1
40,male,70000,phd,1
33,female,52000,masters,0
26,male,38000,bachelors,0
42,female,75000,phd,1
29,male,48000,masters,1
37,female,65000,phd,0"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(dummy_data)
        temp_file = f.name

    try:
        preprocessor = DataPreprocessor()
        data = preprocessor.load_and_preprocess(
            temp_file, label_column="label", sensitive_features=None  # Auto-detect
        )

        print(f"✅ Input dim: {data['input_dim']}")
        print(f"✅ Protected features: {data['protected_features']}")
        print(f"✅ Protected indices: {data['protected_indices']}")
        print(f"✅ Train batches: {len(data['train_loader'])}")

    finally:
        os.unlink(temp_file)

    print("✅ All tests passed!")
