from __future__ import annotations
import hashlib
import mlflow
import pandas as pd
from typing import Dict, Any


def dataframe_hash(df: pd.DataFrame) -> str:
    # Stable hash of dataframe values and columns
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    cols = ",".join(df.columns)
    m = hashlib.sha256()
    m.update(content)
    m.update(cols.encode())
    return m.hexdigest()


def start_run(experiment: str, run_name: str | None = None, tags: Dict[str, Any] | None = None):
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name, tags=tags)