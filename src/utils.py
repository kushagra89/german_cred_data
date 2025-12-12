import os
import pickle
import json
import logging
import numpy as np
import pandas as pd
import random

# ---------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------

def ensure_dir(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def ensure_parent_dir(file_path: str):
    """Ensure parent directory exists for a given file."""
    directory = os.path.dirname(file_path)
    if directory:
        ensure_dir(directory)

# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------

def get_logger(name: str, level=logging.INFO):
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# ---------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------

def save_pickle(obj, file_path: str):
    """Save an object to pickle."""
    ensure_parent_dir(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(file_path: str):
    """Load object from pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_json(obj, file_path: str):
    """Save dictionary as JSON."""
    ensure_parent_dir(file_path)
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)

def load_json(file_path: str):
    """Load JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

# ---------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------

def set_seed(seed: int = 42):
    """Set seed for Python, NumPy, and Random."""
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------

def validate_columns(df: pd.DataFrame, required_cols: list):
    """Ensure input DataFrame contains all required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

# ---------------------------------------------------
# PRETTY PRINTING
# ---------------------------------------------------

def highlight(msg: str):
    """Highlight text for readability."""
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60 + "\n")