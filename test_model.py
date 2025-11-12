# test_model.py
"""
Model Testing & Evaluation Script
Loads trained model(s), tests on new dataset, generates predictions,
calculates metrics, saves results, and creates comparison visualizations.
"""

from __future__ import annotations
import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import constants and utilities from model.py
from model import (
    normalize_plant_name, normalize_column_names, FEATURE_BASE,
    TIME_COL, PLANT_COL, TARGET_COL, MODELS_FOLDER, METRICS_FOLDER,
    add_time_features, add_interaction_features
)

# Setup logging
logger = logging.getLogger("plant_health_testing")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh = logging.FileHandler(os.path.join("logs", "test_model.log"))
fh.setFormatter(fmt)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

RESULTS_FOLDER = "test_results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============== Data Loading & Preprocessing ==============
def load_test_data(csv_path: str) -> pd.DataFrame:
    """Load and validate test dataset."""
    logger.info(f"Loading test data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = normalize_column_names(df)
    
    # Validate required columns
    required = {TARGET_COL}.union(set(FEATURE_BASE))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in test data: {missing}")
    
    # Parse time if present
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    
    # Normalize plant names if present
    