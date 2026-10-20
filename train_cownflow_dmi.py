import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print("Matched features:", features)
print("Missing feature names:", [c for c in numeric + categorical if c not in df.columns])

if target not in df.columns or group not in df.columns:
    raise ValueError(f"Missing target/group. Target candidates: {[c for c in df.columns if 'DM-intake' in c]}, group candidates: {[c for c in df.columns if 'Cow-ID' in c]}")
if not features:
    raise ValueError("No features matched.")

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

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
}

results = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "model": name,
        "MAE_kg_day": round(mean_absolute_error(y_test, pred), 4),
        "RMSE_kg_day": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "R2": round(r2_score(y_test, pred), 4),
    })

results_df = pd.DataFrame(results).sort_values("MAE_kg_day")
results_df.to_csv(out_dir / "metrics.csv", index=False)

print("\nRows:", len(data), "| Unique cows:", groups.nunique())
print("Train rows:", len(train_idx), "| Test rows:", len(test_idx))
print("\nRESULTS:")
print(results_df.to_string(index=False))
print("\nSaved to:", out_dir / "metrics.csv")
