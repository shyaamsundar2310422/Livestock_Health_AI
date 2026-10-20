import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

data_path = Path("dataset_dmi_cownflow/CowNflow_6_Cow_measurements.xlsx")
out_dir = Path("outputs/cownflow_dmi_experiment")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(data_path)
df.columns = df.columns.str.strip()

target = "4.-DM-intake-(kg/day)"
group = "1.-Cow-ID"

numeric = [
    "2.-Cow-age-(month)",
    "2.-Body-weight-(kg)",
    "2.-Lactation-week",
    "2.-Gestation-week",
]
categorical = [
    "2.-Physiological-status-",
    "2.-Gestation-status",
    "3.-Diet-type",
]
features = [c for c in numeric + categorical if c in df.columns]

if target not in df.columns or group not in df.columns or not features:
    raise ValueError("Target, cow ID, or feature columns do not match the dataset.")

data = df[features + [target, group]].copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, group])

X, y, groups = data[features], data[target], data[group]
numeric_used = [c for c in numeric if c in features]
categorical_used = [c for c in categorical if c in features]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_used),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_used),
])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    ),
}

results = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results.append({
        "model": name,
        "MAE_kg_day": mean_absolute_error(y_test, pred),
        "RMSE_kg_day": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "split": "GroupShuffleSplit_25pct_random_state_42",
        "features": ", ".join(features),
    })

results_df = pd.DataFrame(results)
results_path = out_dir / "model_comparison_v2.csv"

if results_path.exists():
    previous = pd.read_csv(results_path)
    results_df = pd.concat([previous, results_df], ignore_index=True)
results_df.to_csv(results_path, index=False)

print("Rows:", len(data), "| Unique cows:", groups.nunique())
print("Training rows:", len(train_idx), "| Testing rows:", len(test_idx))
print("\nMODEL COMPARISON")
print(results_df[["model", "MAE_kg_day", "RMSE_kg_day", "R2"]].to_string(index=False))
print("\nComparison saved to:", results_path)
