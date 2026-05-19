from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "breast_cancer_raw.csv"
PROCESSED_FILE = PROJECT_ROOT / "preprocessing" / "breast_cancer_preprocessed.csv"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_dataset() -> pd.DataFrame:

    if RAW_FILE.exists():
        return pd.read_csv(RAW_FILE)

    dataset = load_breast_cancer(as_frame=True)
    raw_data = dataset.frame.copy()
    raw_data.to_csv(RAW_FILE, index=False)
    return raw_data


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return basic inspection results for experimentation or logging."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": df.columns.tolist(),
    }


def preprocess_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Preprocess the dataset and return train/test splits ready for modeling."""
    cleaned = df.drop_duplicates().copy()

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single preprocessed dataset for export."""
    cleaned = df.drop_duplicates().copy()
    features = cleaned.drop(columns=[TARGET_COLUMN])
    target = cleaned[TARGET_COLUMN].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    processed = pd.DataFrame(scaled_features, columns=features.columns)
    processed[TARGET_COLUMN] = target.values
    return processed


def save_processed_dataset(
    df: pd.DataFrame, output_path: Path = PROCESSED_FILE
) -> Path:
    df.to_csv(output_path, index=False)
    return output_path


def run_preprocessing() -> Path:
    raw_dataset = load_raw_dataset()
    processed_dataset = build_processed_dataset(raw_dataset)
    return save_processed_dataset(processed_dataset)


def main() -> None:
    output_path = run_preprocessing()
    print(f"Preprocessed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
