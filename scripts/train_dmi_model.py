"""
train_dmi_model.py
==================
Trains the Pooled Expanded Extra Trees Regressor for Dry Matter Intake (DMI)
prediction using INRAE Dataset 2 (DataBase_Cows.csv).

Exactly reproduces the cohort-aware experiment setup from the uploaded
cohort_aware_dmi_results.zip:
  - Features : age, parity, DIM, BW, BCS, Milk, PC_Milk, Fat_Milk, Lac_Milk
  - Target   : DMI (kg/day, observed values only — rows where DMI is not NaN)
  - Groups   : Cow_ID  (grouped split/CV to prevent cow-level leakage)
  - Holdout  : GroupShuffleSplit(test_size=0.25, random_state=42)
  - CV       : GroupKFold(n_splits=5)
  - Imputer  : SimpleImputer(strategy='median') — baked into the pipeline
  - Regressor: ExtraTreesRegressor(n_estimators=200, random_state=42)

Usage
-----
  python scripts/train_dmi_model.py --data DataBase_Cows.csv

The script saves:
  models/dmi_extra_trees_pooled_expanded.joblib   <- fitted sklearn Pipeline
  models/dmi_model_metadata.json                  <- feature order, metrics, etc.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Constants (must match src/config.py exactly) ──────────────────────────────
FEATURE_NAMES = ["age", "parity", "DIM", "BW", "BCS", "Milk", "PC_Milk", "Fat_Milk", "Lac_Milk"]
TARGET_COL    = "DMI"
GROUP_COL     = "Cow_ID"

# Canonical column name variants the CSV might use
COLUMN_ALIASES = {
    "age"      : ["age", "Age", "AGE"],
    "parity"   : ["parity", "Parity", "PARITY", "lact", "Lact", "lactation"],
    "DIM"      : ["DIM", "dim", "days_in_milk", "Days_in_milk"],
    "BW"       : ["BW", "bw", "body_weight", "BodyWeight", "Body_Weight"],
    "BCS"      : ["BCS", "bcs", "body_condition_score", "BodyConditionScore"],
    "Milk"     : ["Milk", "milk", "milk_yield", "Milk_yield", "MilkYield", "Milk_Yield"],
    "PC_Milk"  : ["PC_Milk", "pc_milk", "PC_milk", "protein_pct", "ProteinPct"],
    "Fat_Milk" : ["Fat_Milk", "fat_milk", "Fat_milk", "fat_pct", "FatPct"],
    "Lac_Milk" : ["Lac_Milk", "lac_milk", "Lac_milk", "lactose_pct", "LactosePct"],
    "DMI"      : ["DMI", "dmi", "dry_matter_intake", "DryMatterIntake", "DMI_kg"],
    "Cow_ID"   : ["Cow_ID", "cow_id", "CowID", "ID", "id", "animal_id"],
}

MODEL_DIR    = Path("models")
MODEL_PATH   = MODEL_DIR / "dmi_extra_trees_pooled_expanded.joblib"
METADATA_PATH = MODEL_DIR / "dmi_model_metadata.json"

# Known holdout split from experiment ZIP (reproduced for exact consistency)
HOLDOUT_TEST_COWS = {
    "DB_01_cow_04", "DB_01_cow_05", "DB_01_cow_07", "DB_01_cow_09", "DB_01_cow_13",
    "DB_02_cow_07", "DB_02_cow_08", "DB_02_cow_17", "DB_02_cow_19", "DB_02_cow_20",
    "DB_02_cow_22",
}


def resolve_column(df: pd.DataFrame, canonical: str) -> str:
    """Return the actual column name in df matching the canonical feature name."""
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return canonical  # return as-is; will fail with a clear KeyError later


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load CSV, resolve column aliases, filter to rows with observed DMI."""
    print(f"\n📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Raw shape : {df.shape}")
    print(f"   Columns   : {list(df.columns[:20])} {'...' if len(df.columns) > 20 else ''}")

    # Resolve column names
    col_map = {}
    for canonical in FEATURE_NAMES + [TARGET_COL, GROUP_COL]:
        actual = resolve_column(df, canonical)
        if actual not in df.columns:
            raise ValueError(
                f"Required column '{canonical}' not found in {csv_path.name}.\n"
                f"Tried aliases: {COLUMN_ALIASES.get(canonical, [canonical])}\n"
                f"Available columns: {list(df.columns)}"
            )
        col_map[canonical] = actual

    print(f"\n   Column mapping resolved:")
    for k, v in col_map.items():
        marker = "" if k == v else f"  (alias → '{v}')"
        print(f"     {k:<12}{marker}")

    # Rename to canonical names for consistency
    reverse_map = {v: k for k, v in col_map.items()}
    df = df.rename(columns=reverse_map)

    # Filter to rows with observed target
    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    print(f"\n   Rows with observed DMI : {len(df):,}  (dropped {before - len(df):,} rows with NaN target)")

    if len(df) < 100:
        raise ValueError(f"Too few rows with observed DMI ({len(df)}). Check the dataset and target column.")

    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COL].astype(float)
    groups = df[GROUP_COL].astype(str)

    print(f"   Unique cows            : {groups.nunique()}")
    print(f"   DMI range              : {y.min():.2f} – {y.max():.2f} kg/day  (mean {y.mean():.2f})")

    return X, y, groups


def build_pipeline() -> Pipeline:
    """Median imputation + Extra Trees Regressor (200 trees)."""
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("regressor",  ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def evaluate_metrics(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"   {label:<30}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"MAE_kg_day": mae, "RMSE_kg_day": rmse, "R2": r2, "n_rows": int(len(y_true))}


def run_grouped_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """5-fold GroupKFold CV — cows never split across train/val."""
    print("\n📊 Running 5-fold GroupKFold cross-validation …")
    gkf = GroupKFold(n_splits=5)
    fold_mae, fold_rmse, fold_r2 = [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        pipe = build_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        mae = mean_absolute_error(y.iloc[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], preds))
        r2  = r2_score(y.iloc[val_idx], preds)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"   Fold {fold_i + 1}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    cv_results = {
        "CV_MAE_mean"  : float(np.mean(fold_mae)),
        "CV_MAE_std"   : float(np.std(fold_mae)),
        "CV_RMSE_mean" : float(np.mean(fold_rmse)),
        "CV_RMSE_std"  : float(np.std(fold_rmse)),
        "CV_R2_mean"   : float(np.mean(fold_r2)),
        "CV_R2_std"    : float(np.std(fold_r2)),
    }
    print(f"\n   CV summary: MAE {cv_results['CV_MAE_mean']:.4f} ± {cv_results['CV_MAE_std']:.4f}"
          f"   RMSE {cv_results['CV_RMSE_mean']:.4f} ± {cv_results['CV_RMSE_std']:.4f}"
          f"   R² {cv_results['CV_R2_mean']:.4f} ± {cv_results['CV_R2_std']:.4f}")
    return cv_results


def run_holdout(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """
    Cow-grouped holdout: reproduce the exact split from the experiment ZIP
    (HOLDOUT_TEST_COWS), or fall back to GroupShuffleSplit if those IDs are absent.
    """
    print("\n🔀 Running cow-grouped holdout evaluation …")
    available_test_cows = set(groups.unique()) & HOLDOUT_TEST_COWS

    if len(available_test_cows) >= 3:
        print(f"   Using exact experiment holdout split ({len(available_test_cows)} test cows matched)")
        test_mask  = groups.isin(available_test_cows)
        train_mask = ~test_mask
    else:
        print("   ⚠️  Holdout cow IDs from experiment not fully matched — using GroupShuffleSplit(0.25, seed=42)")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        train_mask = pd.Series(False, index=X.index)
        test_mask  = pd.Series(False, index=X.index)
        train_mask.iloc[train_idx] = True
        test_mask.iloc[test_idx]   = True

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"   Train: {len(X_train):,} rows ({groups[train_mask].nunique()} cows)   "
          f"Test: {len(X_test):,} rows ({groups[test_mask].nunique()} cows)")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    holdout_metrics = evaluate_metrics(y_test, preds, "Holdout (cow-grouped)")
    return holdout_metrics


def train_final_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train on ALL rows with observed DMI (full dataset)."""
    print("\n🚂 Training final pipeline on full dataset …")
    pipe = build_pipeline()
    pipe.fit(X, y)
    print("   ✅ Pipeline fitted.")
    return pipe


def save_artifacts(pipeline: Pipeline, metadata: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n💾 Model saved   → {MODEL_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved → {METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train DMI Extra Trees pipeline")
    parser.add_argument(
        "--data",
        default="DataBase_Cows.csv",
        help="Path to DataBase_Cows.csv  (default: project root)",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"\n❌  Dataset not found: {csv_path.resolve()}")
        print("\n   Please place DataBase_Cows.csv at one of these locations:")
        print(f"      {Path('DataBase_Cows.csv').resolve()}   ← default")
        print(f"      <any path>  and pass it as:  --data <path>")
        print("\n   The file must contain at minimum these columns (case-insensitive):")
        for feat in FEATURE_NAMES + [TARGET_COL, GROUP_COL]:
            print(f"      {feat}")
        sys.exit(1)

    print("=" * 70)
    print("  DMI Extra Trees Regressor — Training Script")
    print("  INRAE Dataset 2 | Pooled Expanded Features (9 features)")
    print("=" * 70)

    # 1. Load data
    X, y, groups = load_and_prepare(csv_path)

    # 2. Grouped CV
    cv_metrics = run_grouped_cv(X, y, groups)

    # 3. Holdout
    holdout_metrics = run_holdout(X, y, groups)

    # 4. Train final model on full data
    final_pipeline = train_final_model(X, y)

    # 5. Quick sanity check on training data (should be very tight — overfit expected for ET)
    print("\n🔍 In-sample sanity check (should be near-zero for Extra Trees):")
    in_sample_preds = final_pipeline.predict(X)
    evaluate_metrics(y, in_sample_preds, "In-sample (full dataset)")

    # 6. Build metadata
    metadata = {
        "model_name"        : "Pooled Expanded Features Extra Trees Regressor",
        "training_dataset"  : "INRAE Dataset 2 — DataBase_Cows.csv",
        "target"            : "DMI",
        "target_units"      : "kg/day",
        "feature_names"     : FEATURE_NAMES,
        "n_features"        : len(FEATURE_NAMES),
        "n_rows_trained_on" : int(len(X)),
        "n_unique_cows"     : int(groups.nunique()),
        "group_column"      : GROUP_COL,
        "model_configuration": {
            "imputer": "SimpleImputer(strategy=median)",
            "regressor": "ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)",
        },
        "holdout_strategy"  : "GroupShuffleSplit by Cow_ID, test_size=0.25, random_state=42",
        "cv_strategy"       : "GroupKFold(n_splits=5) by Cow_ID",
        "holdout_metrics"   : holdout_metrics,
        "cv_metrics"        : cv_metrics,
        "artifact_path"     : str(MODEL_PATH),
        "caveats" : [
            "Research-stage model trained on INRAE Dataset 2 only.",
            "Not validated on external farms, breeds, or management systems.",
            "Predictions are AI estimates, not prescribed feed rations.",
            "DMI may vary substantially with diet formulation, health status, and season.",
            "Consult a qualified nutritionist before adjusting feed rations.",
        ],
    }

    # 7. Save
    save_artifacts(final_pipeline, metadata)

    # 8. Print summary
    print("\n" + "=" * 70)
    print("  ✅  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Grouped CV  (5-fold):  MAE {cv_metrics['CV_MAE_mean']:.3f} ± {cv_metrics['CV_MAE_std']:.3f} kg/day"
          f"   R² {cv_metrics['CV_R2_mean']:.3f} ± {cv_metrics['CV_R2_std']:.3f}")
    print(f"  Holdout     (cow-grp): MAE {holdout_metrics['MAE_kg_day']:.3f} kg/day"
          f"   RMSE {holdout_metrics['RMSE_kg_day']:.3f}   R² {holdout_metrics['R2']:.3f}")
    print(f"\n  Artifact : {MODEL_PATH.resolve()}")
    print(f"  Metadata : {METADATA_PATH.resolve()}")
    print()
    print("  Next step:")
    print("    python -m streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
