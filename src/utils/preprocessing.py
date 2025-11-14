from typing import Tuple

import numpy as np
import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "class_id",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa o DataFrame em X (features) e y (alvo).
    """
    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def minmax_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalização min-max simples por coluna.
    Mantém o DataFrame com os mesmos nomes de colunas.
    """
    X_norm = X.copy()
    for col in X_norm.columns:
        col_min = X_norm[col].min()
        col_max = X_norm[col].max()
        if col_max == col_min:
            X_norm[col] = 0.0
        else:
            X_norm[col] = (X_norm[col] - col_min) / (col_max - col_min)
    return X_norm