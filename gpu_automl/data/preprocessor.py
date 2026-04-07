"""Basic GPU-native data preprocessing."""

from __future__ import annotations

from typing import Optional

import cudf
import cupy as cp
import numpy as np


class Preprocessor:
    """GPU-native preprocessor for tabular data.

    Handles missing values and categorical encoding using cuDF operations.
    Designed to run entirely on GPU memory.
    """

    def __init__(self):
        self._numeric_fill_values: dict[str, float] = {}
        self._categorical_mappings: dict[str, dict] = {}
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._is_fitted = False

    def fit(self, X: cudf.DataFrame) -> "Preprocessor":
        """Learn preprocessing parameters from training data."""
        self._numeric_cols = []
        self._categorical_cols = []

        for col in X.columns:
            if X[col].dtype in ("object", "str", "category"):
                self._categorical_cols.append(col)
            else:
                self._numeric_cols.append(col)

        # Numeric: store median for imputation
        for col in self._numeric_cols:
            series = X[col]
            if series.isna().any():
                median_val = float(series.median())
                self._numeric_fill_values[col] = median_val

        # Categorical: label encoding mappings
        for col in self._categorical_cols:
            categories = X[col].dropna().unique().to_pandas().tolist()
            self._categorical_mappings[col] = {
                cat: idx for idx, cat in enumerate(categories)
            }

        self._is_fitted = True
        return self

    def transform(self, X: cudf.DataFrame) -> cudf.DataFrame:
        """Apply preprocessing to data."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        X = X.copy()

        # Impute numeric columns
        for col, fill_val in self._numeric_fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_val)

        # Label-encode categorical columns
        for col, mapping in self._categorical_mappings.items():
            if col in X.columns:
                # Map known categories, unknowns become -1
                X[col] = X[col].map(mapping).fillna(-1).astype("int32")

        # Fill any remaining NaN in numeric cols with 0
        for col in self._numeric_cols:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(0)

        return X

    def fit_transform(self, X: cudf.DataFrame) -> cudf.DataFrame:
        return self.fit(X).transform(X)
