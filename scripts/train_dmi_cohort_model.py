"""Compare tree regressors for DMI on a cow-grouped holdout.

This is an experimental cohort-data comparison, not a production training
workflow. It uses only complete-case rows and does not remove outliers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


SEED = 42
TEST_SIZE = 0.25
FEATURE_NAMES = [
    "age",
    "parity",
    "DIM",
    "BW",
    "BCS",
    "Milk",
    "PC_Milk",
    "Fat_Milk",
    "Lac_Milk",
]
TARGET_NAME = "DMI"
GROUP_NAME = "Cow_ID"
REQUIRED_COLUMNS = [GROUP_NAME, TARGET_NAME, *FEATURE_NAMES]

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_DIR / "DataBase_Cows.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "dmi_cohort_best_tree.joblib"
FEATURES_PATH = MODEL_DIR / "dmi_cohort_feature_names.json"
REPORT_PATH = MODEL_DIR / "dmi_cohort_experiment_report.json"


def build_pipeline(regressor: Any) -> Pipeline:
    """Keep preprocessing fitted inside each model's training partition."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor),
        ]
    )


def load_complete_cases(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path, low_memory=False)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    for column in [TARGET_NAME, *FEATURE_NAMES]:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' contains non-numeric values: {exc}") from exc

    complete_mask = frame[[TARGET_NAME, *FEATURE_NAMES]].notna().all(axis=1)
    complete = frame.loc[complete_mask].copy()
    if complete[GROUP_NAME].isna().any():
        missing_group_rows = int(complete[GROUP_NAME].isna().sum())
        raise ValueError(
            f"{missing_group_rows} complete-case rows have missing {GROUP_NAME}; "
            "cow-grouped splitting cannot be performed safely."
        )
    if complete.empty:
        raise ValueError("No rows have non-missing DMI and all nine input features.")

    dataset_counts = {
        "dataset_path": str(csv_path.resolve()),
        "total_rows": int(len(frame)),
        "rows_with_missing_target_or_features_excluded": int(len(frame) - len(complete)),
        "complete_case_rows": int(len(complete)),
        "complete_case_unique_cows": int(complete[GROUP_NAME].nunique()),
        "target_missing_rows": int(frame[TARGET_NAME].isna().sum()),
    }
    return (
        complete[FEATURE_NAMES],
        complete[TARGET_NAME].astype(float),
        complete[GROUP_NAME].astype(str),
        dataset_counts,
    )


def score_model(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    """Fit and score one model, preserving per-model errors in the report."""
    try:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": float(r2_score(y_test, predictions)),
            "test_rows": int(len(y_test)),
        }
        print(
            f"{name}: MAE={metrics['MAE']:.4f} kg/day, "
            f"RMSE={metrics['RMSE']:.4f} kg/day, R²={metrics['R2']:.4f}"
        )
        return {"status": "ok", "metrics": metrics, "pipeline": pipeline}
    except Exception as exc:
        print(f"{name}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare DMI regressors using a cow-grouped holdout."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV dataset path (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()

    try:
        x, y, groups, dataset_counts = load_complete_cases(args.data)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("DMI cohort model comparison (experimental; no outliers removed)")
    print(f"Complete-case rows: {len(x):,}")
    print(f"Unique cows: {groups.nunique():,}")
    if groups.nunique() < 2:
        print("ERROR: At least two cows are required for a grouped holdout.", file=sys.stderr)
        return 1

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_indices, test_indices = next(splitter.split(x, y, groups))
    train_cows = set(groups.iloc[train_indices])
    test_cows = set(groups.iloc[test_indices])
    overlap = train_cows & test_cows
    if overlap:
        print(f"ERROR: Cow leakage detected: {sorted(overlap)}", file=sys.stderr)
        return 1

    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    print(
        f"Training split: {len(train_indices):,} rows, {len(train_cows):,} cows; "
        f"testing split: {len(test_indices):,} rows, {len(test_cows):,} cows"
    )
    print(f"Split: GroupShuffleSplit(test_size={TEST_SIZE}, random_state={SEED}) by {GROUP_NAME}")

    candidates = {
        "DummyRegressor(mean)": build_pipeline(DummyRegressor(strategy="mean")),
        "RandomForestRegressor": build_pipeline(
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        ),
        "ExtraTreesRegressor": build_pipeline(
            ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        ),
    }
    results = {
        name: score_model(name, pipeline, x_train, x_test, y_train, y_test)
        for name, pipeline in candidates.items()
    }

    tree_names = ("RandomForestRegressor", "ExtraTreesRegressor")
    successful_trees = [
        (name, results[name])
        for name in tree_names
        if results[name]["status"] == "ok"
    ]
    report = {
        "experiment": "DMI complete-case tree model comparison",
        "training_dataset": "INRAE Dataset 2 — DataBase_Cows.csv",
        "target": TARGET_NAME,
        "target_units": "kg/day",
        "feature_names": FEATURE_NAMES,
        "group_column": GROUP_NAME,
        "seed": SEED,
        "preprocessing": "SimpleImputer(strategy='median') inside each model pipeline",
        "outlier_handling": "No outliers removed",
        "dataset_counts": dataset_counts,
        "split": {
            "method": "GroupShuffleSplit",
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "group_column": GROUP_NAME,
            "train_rows": int(len(train_indices)),
            "train_cows": int(len(train_cows)),
            "test_rows": int(len(test_indices)),
            "test_cows": int(len(test_cows)),
            "cow_overlap": sorted(overlap),
        },
        "models": {
            name: {key: value for key, value in result.items() if key != "pipeline"}
            for name, result in results.items()
        },
        "selected_model": None,
        "model_artifact": None,
        "feature_list_artifact": str(FEATURES_PATH),
        "model_status": "no tree model completed successfully",
        "limitations": [
            "Experimental result; not production-ready or externally validated.",
            "Holdout metrics apply only to this grouped split and dataset.",
        ],
    }

    if successful_trees:
        selected_name, selected_result = min(
            successful_trees,
            key=lambda item: (
                item[1]["metrics"]["MAE"],
                item[1]["metrics"]["RMSE"],
            ),
        )
        final_pipeline = selected_result["pipeline"]
        try:
            # Keep the reported metrics from the held-out evaluation, then fit
            # the selected artifact on all complete cases for later experiments.
            final_pipeline.fit(x, y)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_pipeline, MODEL_PATH)
            FEATURES_PATH.write_text(
                json.dumps(FEATURE_NAMES, indent=2) + "\n", encoding="utf-8"
            )
            report["selected_model"] = selected_name
            report["model_artifact"] = str(MODEL_PATH)
            report["model_status"] = "saved; refitted on all complete-case rows after holdout evaluation"
            print(f"Selected tree model: {selected_name} (lowest holdout MAE)")
            print(f"Saved model: {MODEL_PATH}")
            print(f"Saved ordered features: {FEATURES_PATH}")
        except Exception as exc:
            report["model_status"] = f"artifact save/refit failed: {type(exc).__name__}: {exc}"
            print(f"ERROR saving selected model: {report['model_status']}", file=sys.stderr)

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved experiment report: {REPORT_PATH}")
    except Exception as exc:
        print(f"ERROR saving experiment report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0 if successful_trees and report["model_artifact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
