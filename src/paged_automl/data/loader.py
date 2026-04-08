"""Data loading utilities using cuDF for GPU-native I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cudf


def load_data(
    path: Union[str, Path],
    target_col: Optional[str] = None,
    usecols: Optional[list[str]] = None,
) -> Union[cudf.DataFrame, tuple[cudf.DataFrame, cudf.Series]]:
    """Load CSV or Parquet data into a cuDF DataFrame on GPU.

    Parameters
    ----------
    path : str or Path
        File path (.csv or .parquet).
    target_col : str, optional
        If provided, splits the DataFrame into (X, y).
    usecols : list[str], optional
        Columns to load.

    Returns
    -------
    cudf.DataFrame or (cudf.DataFrame, cudf.Series)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = cudf.read_csv(str(path), usecols=usecols)
    elif suffix in (".parquet", ".pq"):
        df = cudf.read_parquet(str(path), columns=usecols)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .parquet.")

    if target_col is not None:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in data.")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

    return df


def load_dask_data(
    path: Union[str, Path],
    target_col: Optional[str] = None,
) -> Union["dask_cudf.DataFrame", tuple["dask_cudf.DataFrame", "dask_cudf.Series"]]:
    """Load data using Dask-cuDF for out-of-VRAM datasets.

    Parameters
    ----------
    path : str or Path
        File path or glob pattern.
    target_col : str, optional
        If provided, splits into (X, y).
    """
    import dask_cudf

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv" or "*" in str(path):
        ddf = dask_cudf.read_csv(str(path))
    elif suffix in (".parquet", ".pq"):
        ddf = dask_cudf.read_parquet(str(path))
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if target_col is not None:
        y = ddf[target_col]
        X = ddf.drop(columns=[target_col])
        return X, y

    return ddf
