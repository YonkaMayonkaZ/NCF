"""I/O utilities for the NCF project."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary as JSON."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object as pickle."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict[str, Any], experiment_name: str, 
                 results_dir: Union[str, Path] = "results") -> None:
    """Save experiment results in multiple formats."""
    results_dir = Path(results_dir)
    
    # Save as JSON for easy reading
    json_path = results_dir / "reports" / f"{experiment_name}_results.json"
    save_json(results, json_path)
    
    # Save as pickle for exact reproduction
    pickle_path = results_dir / "reports" / f"{experiment_name}_results.pkl"
    save_pickle(results, pickle_path)
    
    # If results contain dataframes, save as CSV
    csv_dir = results_dir / "tables"
    ensure_dir(csv_dir)
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            csv_path = csv_dir / f"{experiment_name}_{key}.csv"
            value.to_csv(csv_path, index=False)