from __future__ import annotations
import os
import glob
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

DATA_FOLDER = "datasets"
MODELS_FOLDER = "models"
LOGS_FOLDER = "logs"
METRICS_FOLDER = "models/metrics"
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(METRICS_FOLDER, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TRAIN_MODE = "per_plant"  # "per_plant" or "combined"
FEATURE_BASE = ["Temperature", "Humidity", "Soil_Moisture", "Rainfall", "Dew_Point"]
TIME_COL = "Time"
PLANT_COL = "Plant_Type"
TARGET_COL = "Health_Percentage"
USE_LIGHTGBM = HAS_LIGHTGBM  # default to available

# Default recommended ranges (configurable via JSON)
DEFAULT_RANGES = {
    "tomato": {"Temperature": [18, 32], "Soil_Moisture": [40, 70], "Humidity": [50, 80], "Rainfall": [0, 80]},
    "potato": {"Temperature": [12, 24], "Soil_Moisture": [45, 75], "Humidity": [60, 85], "Rainfall": [0, 80]},
    "wheat": {"Temperature": [5, 25], "Soil_Moisture": [30, 60], "Humidity": [40, 70], "Rainfall": [0, 50]},
    "generic": {"Temperature": [10, 30], "Soil_Moisture": [30, 60], "Humidity": [40, 80], "Rainfall": [0, 100]},
}


logger = logging.getLogger("plant_health")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh = logging.FileHandler(os.path.join(LOGS_FOLDER, "plant_health.log"))
fh.setFormatter(fmt)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# ---------------- Utilities ----------------
def normalize_plant_name(name: Any) -> str:
    """Normalize plant type strings (case-insensitive)."""
    if pd.isna(name):
        return "unknown"
    return str(name).strip().lower()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from common sensor formats to standard names.
    e.g., Temperature_C -> Temperature
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        if col.lower().startswith("temperature"):
            rename_map[col] = "Temperature"
        elif col.lower().startswith("humidity"):
            rename_map[col] = "Humidity"
        elif col.lower().startswith("soil_moisture"):
            rename_map[col] = "Soil_Moisture"
        elif col.lower().startswith("rainfall"):
            rename_map[col] = "Rainfall"
        elif col.lower().startswith("dew_point"):
            rename_map[col] = "Dew_Point"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.info(f"Normalized column names for a file: {rename_map}")
    return df


def discover_csv_files(data_dir: str = DATA_FOLDER) -> List[str]:
    """Find all CSV files under data_dir, recursively."""
    path_pattern = os.path.join(data_dir, "**", "*.csv")
    files = glob.glob(path_pattern, recursive=True)
    files = sorted(list(set(files)))
    logger.info(f"Discovered {len(files)} CSV files.")
    for f in files:
        logger.info(f"  - Found: {os.path.abspath(f)}")
    return files

# ---------------- Data Loading & Validation ----------------
def load_data(data_dir: str = DATA_FOLDER, sample: bool=False) -> pd.DataFrame:
    """
    Load all CSVs from data_dir, concat, validate columns.
    Drops duplicates and logs missing columns.
    """
    files = discover_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = normalize_column_names(df)
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
            logger.info(f"Loaded {f}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

        if sample and len(dfs) >= 1:
            break

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset rows (before dedup): {len(df)}")

    # required columns check
    required = {PLANT_COL, TIME_COL, TARGET_COL}.union(set(FEATURE_BASE))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    logger.info(f"Dropped {before-after} duplicate rows; {after} remain")

    # Normalize plant names
    df[PLANT_COL] = df[PLANT_COL].apply(normalize_plant_name)

    # Time parse
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        logger.warning("Some Time values could not be parsed and are NaT")

    return df.reset_index(drop=True)

# ---------------- Feature Engineering ----------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, dayofweek, month, is_daytime features derived from TIME_COL."""
    df = df.copy()
    df["hour"] = df[TIME_COL].dt.hour.fillna(12).astype(int)
    df["dayofweek"] = df[TIME_COL].dt.dayofweek.fillna(0).astype(int)
    df["month"] = df[TIME_COL].dt.month.fillna(1).astype(int)
    df["is_daytime"] = df["hour"].apply(lambda h: 1 if 6 <= h <= 18 else 0)
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple interactions likely useful."""
    df = df.copy()
    if "Temperature" in df.columns and "Humidity" in df.columns:
        df["temp_x_hum"] = df["Temperature"] * df["Humidity"]
    if "Soil_Moisture" in df.columns and "Rainfall" in df.columns:
        df["moisture_x_rain"] = df["Soil_Moisture"] * (1 + df["Rainfall"]/100.0)
    # dew gap
    if "Temperature" in df.columns and "Dew_Point" in df.columns:
        df["temp_minus_dew"] = df["Temperature"] - df["Dew_Point"]
    return df

def add_rolling_features(df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """
    Add rolling means over window_hours if dataset is time-series per plant.
    This function requires TIME_COL to be datetime and sorted per plant.
    """
    df = df.sort_values(TIME_COL).copy()
    if TIME_COL not in df.columns:
        return df
    # apply per-plant rolling if many timestamps present
    if df[TIME_COL].isnull().all():
        return df
    try:
        df = df.set_index(TIME_COL)
        rolling_window = f"{max(1, window_hours)}h"
        for col in ["Temperature", "Soil_Moisture", "Humidity", "Rainfall"]:
            if col in df.columns:
                df[f"{col}_roll_mean_{window_hours}h"] = df[col].rolling(rolling_window, min_periods=1).mean()
        df = df.reset_index()
    except Exception as e:
        logger.warning(f"Rolling features skipped due to: {e}")
    return df

# ---------------- Preprocessing & Outlier Detection ----------------
def impute_and_scale(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Impute numeric columns with median and scale with StandardScaler.
    Returns (transformed_array, scaler)
    """
    X_num = X.select_dtypes(include=[np.number]).copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
    else:
        X_scaled = scaler.transform(X_imp)
    return X_scaled, scaler

def flag_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Simple IQR-based outlier flagging: marks rows where column beyond 1.5*IQR.
    Adds column '<col>_outlier' boolean for each col.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[f"{c}_outlier"] = ((df[c] < low) | (df[c] > high)).astype(int)
    return df

# ---------------- Model training & selection ----------------
def build_regressor(use_lightgbm: bool = USE_LIGHTGBM):
    """Return a sklearn-compatible regressor and a parameter distribution for RandomizedSearchCV."""
    if use_lightgbm:
        reg = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 400],
            "num_leaves": [31, 50, 80],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_samples": [5, 10, 20],
        }
    else:
        reg = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    return reg, param_dist

def train_model_for_plant(df: pd.DataFrame, plant: str, out_dir: str = MODELS_FOLDER,
                          use_lightgbm: bool = USE_LIGHTGBM, tune: bool = True,
                          test_size: float = TEST_SIZE, time_aware: bool = True) -> Dict[str, Any]:
    """
    Train model for a single plant (DataFrame already filtered).
    Saves model pipeline and metrics JSON under out_dir.
    Returns metadata dict.
    """
    plant_norm = normalize_plant_name(plant)
    logger.info(f"Training for plant '{plant_norm}' ({len(df)} rows)")

    # optionally engineer time features if time present
    if time_aware:
        df = add_time_features(df)
        df = add_rolling_features(df, window_hours=24)

    df = add_interaction_features(df)
    df = flag_outliers_iqr(df, FEATURE_BASE)

    # form features list
    features = [c for c in df.columns if c not in [TARGET_COL, PLANT_COL, "_source_file"]]
    # prefer a subset
    use_feats = [c for c in ["Temperature", "Humidity", "Soil_Moisture", "Rainfall", "Dew_Point",
                             "temp_x_hum", "moisture_x_rain", "temp_minus_dew",
                             "hour", "dayofweek", "month", "is_daytime"] if c in df.columns]

    X = df[use_feats].copy()
    y = df[TARGET_COL].astype(float).copy()

    # Drop rows with missing target
    idx_keep = y.notnull()
    X = X.loc[idx_keep]
    y = y.loc[idx_keep]
    if len(y) < 10:
        logger.warning(f"Very few labeled rows ({len(y)}) for plant {plant_norm}. Consider global model.")

    # time-split if time present else random
    if TIME_COL in df.columns and not df[TIME_COL].isnull().all():
        df_sorted = df.loc[idx_keep].sort_values(TIME_COL)
        # use last test_size fraction as test set
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train = df_sorted[use_feats].iloc[:split_idx]
        y_train = df_sorted[TARGET_COL].iloc[:split_idx]
        X_test = df_sorted[use_feats].iloc[split_idx:]
        y_test = df_sorted[TARGET_COL].iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    # pipeline: imputer -> scaler -> regressor
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    reg, param_dist = build_regressor(use_lightgbm=use_lightgbm)

    # apply imputer and scaler to training data
    X_train_imp = imputer.fit_transform(X_train)
    scaler.fit(X_train_imp)
    X_train_scaled = scaler.transform(X_train_imp)

    # tuning
    if tune:
        rsearch = RandomizedSearchCV(reg, param_distributions=param_dist, n_iter=12,
                                     scoring="neg_mean_absolute_error", cv=3, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
        rsearch.fit(X_train_scaled, y_train)
        best_reg = rsearch.best_estimator_
        best_params = getattr(rsearch, "best_params_", {})
        logger.info(f"Tuning best params: {best_params}")
    else:
        best_reg = reg
        best_reg.fit(X_train_scaled, y_train)
        best_params = {}

    # final pipeline wrapper object to save: store imputer, scaler, features list, regressor
    pipeline = {
        "imputer": imputer,
        "scaler": scaler,
        "features": use_feats,
        "regressor": best_reg,
        "plant": plant_norm,
        "trained_on": int(len(X_train)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": best_params
    }

    # evaluate
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    preds = pipeline["regressor"].predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    within_10 = float(np.mean(np.abs(y_test - preds) <= 10) * 100.0)

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "within_10_pct": within_10, "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    # save model
    model_fname = os.path.join(out_dir, f"{plant_norm}_model.joblib")
    dump(pipeline, model_fname)
    metrics_fname = os.path.join(METRICS_FOLDER, f"{plant_norm}_metrics.json")
    with open(metrics_fname, "w") as fh:
        json.dump({"metrics": metrics, "pipeline_meta": {k: pipeline[k] for k in ["plant", "trained_on", "timestamp", "params"]}}, fh, default=str, indent=2)

    logger.info(f"Saved model -> {model_fname}")
    logger.info(f"Saved metrics -> {metrics_fname}")
    return {"model_path": model_fname, "metrics": metrics, "pipeline": pipeline}

def train_all(data_dir: str = DATA_FOLDER, out_dir: str = MODELS_FOLDER, train_mode: str = DEFAULT_TRAIN_MODE, tune: bool = True):
    """
    Entry function to train models. If train_mode == "per_plant", train one model per CSV file.
    If "combined", train a global model from all CSVs.
    """
    if train_mode == "combined":
        df = load_data(data_dir)
        df = add_time_features(df)
        df = add_interaction_features(df)
        # train a single model including plant type encoded
        df["plant_enc"] = df[PLANT_COL].astype("category").cat.codes
        # reuse lower-level training with small wrapper
        # Build features
        feats = [c for c in FEATURE_BASE + ["plant_enc", "hour", "dayofweek", "month", "is_daytime"] if c in df.columns]
        X = df[feats]
        y = df[TARGET_COL].astype(float)
        # simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        reg, param_dist = build_regressor(use_lightgbm=USE_LIGHTGBM)
        X_train_imp = imputer.fit_transform(X_train)
        scaler.fit(X_train_imp)
        X_train_scaled = scaler.transform(X_train_imp)
        if tune:
            search = RandomizedSearchCV(reg, param_distributions=param_dist, n_iter=12, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=RANDOM_STATE)
            search.fit(X_train_scaled, y_train)
            reg_best = search.best_estimator_
            params = getattr(search, "best_params_", {})
        else:
            reg.fit(X_train_scaled, y_train)
            reg_best = reg
            params = {}
        pipeline = {"imputer": imputer, "scaler": scaler, "features": feats, "regressor": reg_best, "plant": "combined", "timestamp": datetime.now(timezone.utc).isoformat(), "params": params}
        model_fname = os.path.join(out_dir, "combined_model.joblib")
        dump(pipeline, model_fname)
        # eval
        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)
        preds = reg_best.predict(X_test_scaled)
        metrics = {"mae": float(mean_absolute_error(y_test, preds)), "rmse": float(np.sqrt(mean_squared_error(y_test, preds))), "r2": float(r2_score(y_test, preds))}
        metrics_fname = os.path.join(METRICS_FOLDER, "combined_metrics.json")
        with open(metrics_fname, "w") as fh:
            json.dump({"metrics": metrics, "pipeline_meta": {"timestamp": pipeline["timestamp"], "params": params}}, fh, default=str, indent=2)
        logger.info(f"Saved combined model -> {model_fname} metrics -> {metrics}")
        return {"combined": {"model_path": model_fname, "metrics": metrics}}

    # per-plant: one model per CSV file
    files = discover_csv_files(data_dir)
    if not files:
        logger.error(f"No CSV files found in {data_dir} for training.")
        return {}

    results = {}
    for f in files:
        try:
            logger.info(f"Processing file for training: {os.path.abspath(f)}")
            df = pd.read_csv(f)
            df = normalize_column_names(df)
            plant_name = os.path.splitext(os.path.basename(f))[0]

            # Ensure time column is parsed
            if TIME_COL in df.columns:
                df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

            required = {TARGET_COL}.union(set(FEATURE_BASE))
            if not required.issubset(df.columns):
                logger.warning(f"Skipping {f}: missing required columns. Found {list(df.columns)}, require {list(required)}")
                continue

            res = train_model_for_plant(df, plant=plant_name, out_dir=out_dir, use_lightgbm=USE_LIGHTGBM, tune=tune, test_size=TEST_SIZE)
            results[plant_name] = res
        except Exception as e:
            logger.error(f"Failed to train model for {f}: {e}", exc_info=True)
    return results

# ---------------- Inference & Recommendations ----------------
def load_pipeline_for_plant(plant: str, train_mode: str = DEFAULT_TRAIN_MODE) -> Dict:
    """Load saved pipeline for a plant (joblib)."""
    plant_norm = normalize_plant_name(plant)
    if train_mode == "combined":
        path = os.path.join(MODELS_FOLDER, "combined_model.joblib")
    else:
        path = os.path.join(MODELS_FOLDER, f"{plant_norm}_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    pipeline = load(path)
    return pipeline

def recommend_from_thresholds(plant: str, inputs: Dict[str, float], ranges: Dict[str, Dict[str, List[float]]]=None) -> List[str]:
    """
    Deterministic recommendations based on ranges dict. Prioritize severity by deviation.
    """
    if ranges is None:
        ranges = DEFAULT_RANGES
    plant_norm = normalize_plant_name(plant)
    cfg = ranges.get(plant_norm, ranges.get("generic", {}))
    recs = []
    sev_list = []
    for feat in ["Soil_Moisture", "Temperature", "Humidity", "Rainfall"]:
        if feat not in inputs or feat not in cfg:
            continue
        val = float(inputs.get(feat))
        lo, hi = cfg[feat]
        if val < lo:
            diff = lo - val
            sev_list.append((diff, feat, "low", lo, hi))
        elif val > hi:
            diff = val - hi
            sev_list.append((diff, feat, "high", lo, hi))
    # sort by severity desc
    sev_list.sort(reverse=True, key=lambda x: x[0])
    for diff, feat, mode, lo, hi in sev_list:
        if mode == "low":
            if feat == "Soil_Moisture":
                recs.append(f"{feat} ({inputs[feat]:.2f}) below recommended min {lo}: Increase irrigation, check root uptake.")
            elif feat == "Temperature":
                recs.append(f"{feat} ({inputs[feat]:.2f}) below recommended min {lo}: Provide warmth/cover.")
            else:
                recs.append(f"{feat} ({inputs[feat]:.2f}) low: adjust environment.")
        else:
            # high
            if feat == "Soil_Moisture":
                recs.append(f"{feat} ({inputs[feat]:.2f}) above recommended max {hi}: Reduce watering, improve drainage - risk of root rot.")
            elif feat == "Temperature":
                recs.append(f"{feat} ({inputs[feat]:.2f}) above recommended max {hi}: Provide shade, reduce heat stress.")
            elif feat == "Humidity":
                recs.append(f"{feat} ({inputs[feat]:.2f}) high: risk of fungal disease - improve ventilation.")
            else:
                recs.append(f"{feat} ({inputs[feat]:.2f}) high: check conditions.")
    if not recs:
        recs.append("All measured parameters within expected ranges.")
    return recs

def predict_single(plant: str, inputs: Dict[str, float], time_str: Optional[str]=None, train_mode: str = DEFAULT_TRAIN_MODE) -> Dict[str, Any]:
    """
    Given plant and raw input dict (Temperature, Humidity, Soil_Moisture, Rainfall, Dew_Point),
    load pipeline, preprocess and predict health percentage and recommendations.
    """
    plant_norm = normalize_plant_name(plant)
    pipeline = load_pipeline_for_plant(plant_norm, train_mode=train_mode)
    features = pipeline["features"]
    # build DataFrame
    df_in = pd.DataFrame([inputs])
    # time features
    if time_str is not None:
        t = pd.to_datetime(time_str, errors="coerce")
    else:
        t = pd.Timestamp.now()
    df_in[TIME_COL] = t
    df_in = add_time_features(df_in)
    df_in = add_interaction_features(df_in)
    # select features in order
    X = df_in[features].copy()
    # impute & scale using saved objects
    X_imp = pipeline["imputer"].transform(X)
    X_scaled = pipeline["scaler"].transform(X_imp)
    pred = pipeline["regressor"].predict(X_scaled)[0]
    pred = float(np.clip(pred, 0.0, 100.0))

    # basic model variance estimate: if tree-based, use std of estimators' predictions
    var = None
    try:
        model = pipeline["regressor"]
        if hasattr(model, "estimators_"):
            # RandomForest: get std across trees
            preds_tree = np.array([est.predict(X_scaled) for est in model.estimators_])
            var = float(np.std(preds_tree))
        elif HAS_LIGHTGBM and isinstance(model, lgb.LGBMRegressor):
            # use built-in predictor variability approximation via n_estimators? fallback None
            var = None
    except Exception:
        var = None

    # recommendations & anomaly flags
    recs = recommend_from_thresholds(plant_norm, inputs)
    # quick outlier detection on input features
    flags = {}
    for feat in FEATURE_BASE:
        if feat in df_in.columns:
            col = df_in[feat].iloc[0]
            # IQR-based local flag: can't compute IQR here without historical data; use simple clamp checks
            if np.isnan(col):
                flags[feat] = "missing"
            else:
                # if outside reasonable absolute ranges, mark
                if feat == "Temperature" and not (-20 <= col <= 60):
                    flags[feat] = "unrealistic"
                if feat == "Soil_Moisture" and not (0 <= col <= 100):
                    flags[feat] = "unrealistic"

    # feature importance (simple): if regressor has feature_importances_
    feat_importances = {}
    try:
        model = pipeline["regressor"]
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            for f_name, imp in zip(features, imps):
                feat_importances[f_name] = float(imp)
            # sort
            feat_importances = dict(sorted(feat_importances.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        feat_importances = {}

    return {
        "Predicted_Health": round(pred, 2),
        "Variance_Estimate": var,
        "Recommendations": recs,
        "Flags": flags,
        "Feature_Importances": feat_importances,
        "Used_Model": pipeline.get("plant", "unknown")
    }

# ---------------- Interactive CLI & helpers ----------------
def manual_test_cli(train_mode: str = DEFAULT_TRAIN_MODE):
    """Interactive prompt-based testing in terminal."""
    print("Manual interactive test. Enter values (press enter to use default sample).")
    plant = input("Plant_Type (e.g., Tomato): ").strip() or "Tomato"
    def input_float(prompt, default):
        s = input(f"{prompt} [{default}]: ").strip()
        return float(s) if s else float(default)
    T = input_float("Temperature (°C)", 25.0)
    H = input_float("Humidity (%)", 60.0)
    SM = input_float("Soil_Moisture (%)", 45.0)
    R = input_float("Rainfall (mm)", 0.0)
    DP = input_float("Dew_Point (°C)", 16.0)
    ts = input("Time (YYYY-MM-DD HH:MM) [now]: ").strip() or None

    inputs = {"Temperature": T, "Humidity": H, "Soil_Moisture": SM, "Rainfall": R, "Dew_Point": DP}
    try:
        out = predict_single(plant, inputs, time_str=ts, train_mode=train_mode)
        print("\n--- Prediction Result ---")
        print(f"Predicted Health Percentage: {out['Predicted_Health']}%")
        if out["Variance_Estimate"] is not None:
            print(f"Model variance (approx std): {out['Variance_Estimate']:.3f}")
        print("\nFeature importances:")
        for f, v in list(out["Feature_Importances"].items())[:6]:
            print(f"  {f}: {v:.4f}")
        print("\nRecommendations:")
        for r in out["Recommendations"]:
            print(f"  - {r}")
        if out["Flags"]:
            print("\nFlags:")
            for k, v in out["Flags"].items():
                print(f"  {k}: {v}")
    except Exception as e:
        logger.exception(f"Interactive prediction failed: {e}")
        print("Prediction failed:", e)

# ---------------- CLI Entrypoint ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Plant Health ML pipeline (train/predict/interactive)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true", help="Train model for --plant")
    g.add_argument("--train-all", action="store_true", help="Train models for all discovered plants")
    g.add_argument("--predict", action="store_true", help="Run prediction (interactive or file input)")
    p.add_argument("--plant", type=str, default=None, help="Plant selector (e.g., Tomato). Used with --train or --predict to target specific plant")
    p.add_argument("--data-dir", type=str, default=DATA_FOLDER, help="Folder with CSV datasets")
    p.add_argument("--model-path", type=str, default=None, help="Model path for predict (optional)")
    p.add_argument("--input-file", type=str, default=None, help="CSV/JSON input file for batch predict")
    p.add_argument("--interactive", action="store_true", help="Interactive CLI input for prediction")

    # Direct prediction inputs
    p.add_argument("--temp", type=float, help="Temperature value for prediction")
    p.add_argument("--humidity", type=float, help="Humidity value for prediction")
    p.add_argument("--soil", type=float, help="Soil moisture value for prediction")
    p.add_argument("--rainfall", type=float, help="Rainfall value for prediction")
    p.add_argument("--dew", type=float, help="Dew point value for prediction")

    p.add_argument("--train-mode", type=str, choices=["per_plant", "combined"], default=DEFAULT_TRAIN_MODE, help="Training mode")
    p.add_argument("--tune", action="store_true", help="Enable hyperparameter randomized tuning")
    p.add_argument("--use-lightgbm", action="store_true", help="Use LightGBM if available")
    return p

def main():
    parser = parse_args()
    args = parser.parse_args()

    # If no action is specified, print help and exit
    if not (args.train or args.train_all or args.predict):
        parser.print_help()
        print("\nmodel.py: error: one of the arguments --train --train-all --predict is required")
        return

    global USE_LIGHTGBM
    if args.use_lightgbm and not HAS_LIGHTGBM:
        logger.warning("LightGBM requested but not installed. Falling back to RandomForest.")
        USE_LIGHTGBM = False
    else:
        USE_LIGHTGBM = args.use_lightgbm or USE_LIGHTGBM

    if args.train or args.train_all:
        if args.train:
            if not args.plant:
                raise ValueError("--train requires --plant PLANT_NAME")
            df_all = load_data(args.data_dir)
            plant_norm = normalize_plant_name(args.plant)
            sub = df_all[df_all[PLANT_COL] == plant_norm].copy()
            if sub.empty:
                raise ValueError(f"No rows found for plant '{plant_norm}' in {args.data_dir}")
            res = train_model_for_plant(sub, plant=args.plant, out_dir=MODELS_FOLDER, use_lightgbm=USE_LIGHTGBM, tune=args.tune)
            print(json.dumps({"result": res["metrics"], "model_path": res["model_path"]}, indent=2))
        elif args.train_all:
            results = train_all(data_dir=args.data_dir, out_dir=MODELS_FOLDER, train_mode=args.train_mode, tune=args.tune)
            print("Training finished.")
            print(json.dumps({k: v.get("metrics", {}) for k, v in results.items()}, indent=2))
    elif args.predict:
        if args.interactive:
            manual_test_cli(train_mode=args.train_mode)
        elif args.input_file:
            # batch predict, simply read CSV or JSON
            if not args.plant:
                raise ValueError("Batch predict requires --plant PLANT_NAME")
            ext = os.path.splitext(args.input_file)[1].lower()
            if ext == ".csv":
                df_in = pd.read_csv(args.input_file)
                # expect columns matching base features
                outputs = []
                for idx, row in df_in.iterrows():
                    inputs = {k: float(row[k]) for k in FEATURE_BASE if k in row}
                    time_str = row.get(TIME_COL, None)
                    out = predict_single(args.plant, inputs, time_str=time_str, train_mode=args.train_mode)
                    outputs.append(out)
                print(json.dumps(outputs, indent=2))
            elif ext in [".json", ".ndjson"]:
                data = json.load(open(args.input_file))
                outputs = []
                for rec in data:
                    inputs = {k: float(rec[k]) for k in FEATURE_BASE if k in rec}
                    time_str = rec.get(TIME_COL, None)
                    out = predict_single(args.plant, inputs, time_str=time_str, train_mode=args.train_mode)
                    outputs.append(out)
                print(json.dumps(outputs, indent=2))
            else:
                raise ValueError("Unsupported input-file type. Use CSV or JSON.")
        elif all(v is not None for v in [args.temp, args.humidity, args.soil, args.rainfall, args.dew]):
            if not args.plant:
                raise ValueError("Prediction with direct inputs requires --plant PLANT_NAME")
            inputs = {
                "Temperature": args.temp,
                "Humidity": args.humidity,
                "Soil_Moisture": args.soil,
                "Rainfall": args.rainfall,
                "Dew_Point": args.dew,
            }
            out = predict_single(args.plant, inputs, train_mode=args.train_mode)
            print(json.dumps(out, indent=2))
        else:
            raise ValueError("Prediction mode: use --interactive, --input-file, or provide all sensor values (--temp, --humidity, etc.)")
    else:
        raise RuntimeError("Unknown mode")

if __name__ == "__main__":
    main()

