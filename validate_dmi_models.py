import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA = Path("dataset_dmi_cownflow/CowNflow_6_Cow_measurements.xlsx")
OUT = Path("outputs/cownflow_dmi_experiment")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)
df.columns = df.columns.str.strip()

target = "4.-DM-intake-(kg/day)"
group = "1.-Cow-ID"

numeric = [
    "2.-Cow-age-(month)", "2.-Body-weight-(kg)",
    "2.-Lactation-week", "2.-Gestation-week"
]
categorical = [
    "2.-Physiological-status-", "2.-Gestation-status", "3.-Diet-type"
]
features = [c for c in numeric + categorical if c in df.columns]

data = df[features + [target, group]].copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, group]).reset_index(drop=True)

X = data[features]
y = data[target]
groups = data[group]

# Keep validation grouped by cow to prevent repeated measurements
# from the same cow appearing in both train and validation folds.
cv = GroupKFold(n_splits=5)

def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), 
         [c for c in numeric if c in features]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), [c for c in categorical if c in features])
    ])
    return Pipeline([("prep", prep), ("model", model)])

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2,
        random_state=42, n_jobs=1
    ),
    "Extra Trees (untuned)": ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2,
        random_state=42, n_jobs=1
    ),
    "Extra Trees (Optuna tuned)": ExtraTreesRegressor(
        n_estimators=150, max_depth=6,
        min_samples_leaf=1, max_features=0.8543596182727795,
        random_state=42, n_jobs=1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=1
    )
}

fold_rows = []
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_cows = set(groups.iloc[train_idx])
    val_cows = set(groups.iloc[val_idx])
    assert train_cows.isdisjoint(val_cows), "Cow leakage detected!"

    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)

        fold_rows.append({
            "model": name,
            "fold": fold,
            "MAE_kg_day": mean_absolute_error(y_val, pred),
            "RMSE_kg_day": np.sqrt(mean_squared_error(y_val, pred)),
            "R2": r2_score(y_val, pred),
            "train_cows": len(train_cows),
            "validation_cows": len(val_cows),
        })

fold_df = pd.DataFrame(fold_rows)
fold_path = OUT / "grouped_cv_fold_results.csv"
fold_df.to_csv(fold_path, index=False)

summary = fold_df.groupby("model").agg(
    MAE_mean=("MAE_kg_day", "mean"),
    MAE_std=("MAE_kg_day", "std"),
    RMSE_mean=("RMSE_kg_day", "mean"),
    RMSE_std=("RMSE_kg_day", "std"),
    R2_mean=("R2", "mean"),
    R2_std=("R2", "std"),
).reset_index().sort_values("MAE_mean")

summary_path = OUT / "grouped_cv_summary.csv"
summary.to_csv(summary_path, index=False)

report = {
    "dataset_rows": len(data),
    "unique_cows": int(groups.nunique()),
    "folds": 5,
    "split_method": "GroupKFold by Cow ID",
    "features": features,
    "summary": summary.replace({np.nan: None}).to_dict(orient="records")
}
with open(OUT / "grouped_cv_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nGROUPED 5-FOLD CROSS-VALIDATION")
print("Rows:", len(data), "| Unique cows:", groups.nunique())
print(summary.round(4).to_string(index=False))
print("\nSaved fold results:", fold_path)
print("Saved summary:", summary_path)
print("Saved report:", OUT / "grouped_cv_report.json")
