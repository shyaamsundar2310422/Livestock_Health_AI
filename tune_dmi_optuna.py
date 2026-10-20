
import json
import numpy as np
import pandas as pd
import optuna

from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = Path("dataset_dmi_cownflow/CowNflow_6_Cow_measurements.xlsx")
OUT = Path("outputs/cownflow_dmi_experiment")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)
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
    raise ValueError(
        f"Column mismatch. Target exists: {target in df.columns}; "
        f"Cow ID exists: {group in df.columns}; features: {features}"
    )

data = df[features + [target, group]].copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, group]).reset_index(drop=True)

X = data[features]
y = data[target]
groups = data[group]

num_cols = [c for c in numeric if c in features]
cat_cols = [c for c in categorical if c in features]

# Fixed held-out cows; do not use them during tuning.
outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(outer.split(X, y, groups))

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
g_train = groups.iloc[train_idx].reset_index(drop=True)

X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

cv = GroupKFold(n_splits=min(5, g_train.nunique()))

def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])

def objective(trial):
    model_name = trial.suggest_categorical(
        "model", ["Extra Trees", "Random Forest", "XGBoost"]
    )

    if model_name in ("Extra Trees", "Random Forest"):
        params = {
            "n_estimators": trial.suggest_int(
                "tree_n_estimators", 150, 400, step=50
            ),
            "max_depth": trial.suggest_categorical(
                "tree_max_depth", ["none", "4", "6", "8", "12"]
            ),
            "min_samples_leaf": trial.suggest_int(
                "tree_min_samples_leaf", 1, 6
            ),
            "max_features": trial.suggest_float(
                "tree_max_features", 0.6, 1.0
            ),
        }
        params["max_depth"] = (
            None if params["max_depth"] == "none"
            else int(params["max_depth"])
        )
        cls = (
            ExtraTreesRegressor
            if model_name == "Extra Trees"
            else RandomForestRegressor
        )
        model = cls(**params, random_state=42, n_jobs=-1)

    else:
        model = XGBRegressor(
            n_estimators=trial.suggest_int(
                "xgb_n_estimators", 100, 400, step=50
            ),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 5),
            learning_rate=trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.15, log=True
            ),
            subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "xgb_colsample_bytree", 0.7, 1.0
            ),
            reg_lambda=trial.suggest_float(
                "xgb_reg_lambda", 0.1, 10.0, log=True
            ),
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )

    scores = []
    for tr, va in cv.split(X_train, y_train, g_train):
        pipe = make_pipeline(model)
        pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = pipe.predict(X_train.iloc[va])
        scores.append(mean_absolute_error(y_train.iloc[va], pred))

    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("fold_mae", [float(s) for s in scores])
    return float(np.mean(scores))

# A new study avoids mixing results from the previous broken search.
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

# Catch a failed trial without terminating all remaining trials.
study.optimize(objective, n_trials=30, catch=(Exception,))

trial_rows = []
for t in study.trials:
    trial_rows.append({
        "trial": t.number,
        "state": t.state.name,
        "model": t.user_attrs.get("model_name"),
        "cv_MAE_mean": t.value,
        "fold_MAE": json.dumps(t.user_attrs.get("fold_mae", [])),
        "params": json.dumps(t.params),
    })

trials_path = OUT / "optuna_trials.csv"
pd.DataFrame(trial_rows).to_csv(trials_path, index=False)

if study.best_trial is None or study.best_trial.value is None:
    raise RuntimeError("All Optuna trials failed. Check the errors above.")

best = study.best_trial
best_name = best.user_attrs["model_name"]
p = best.params

# Convert Optuna search names back to estimator parameter names.
if best_name == "Extra Trees":
    final_model = ExtraTreesRegressor(
        n_estimators=p["tree_n_estimators"],
        max_depth=None if p["tree_max_depth"] == "none"
        else int(p["tree_max_depth"]),
        min_samples_leaf=p["tree_min_samples_leaf"],
        max_features=p["tree_max_features"],
        random_state=42, n_jobs=-1,
    )
elif best_name == "Random Forest":
    final_model = RandomForestRegressor(
        n_estimators=p["tree_n_estimators"],
        max_depth=None if p["tree_max_depth"] == "none"
        else int(p["tree_max_depth"]),
        min_samples_leaf=p["tree_min_samples_leaf"],
        max_features=p["tree_max_features"],
        random_state=42, n_jobs=-1,
    )
else:
    final_model = XGBRegressor(
        n_estimators=p["xgb_n_estimators"],
        max_depth=p["xgb_max_depth"],
        learning_rate=p["xgb_learning_rate"],
        subsample=p["xgb_subsample"],
        colsample_bytree=p["xgb_colsample_bytree"],
        reg_lambda=p["xgb_reg_lambda"],
        objective="reg:squarederror",
        random_state=42, n_jobs=4,
    )

best_pipe = make_pipeline(final_model)
best_pipe.fit(X_train, y_train)
pred = best_pipe.predict(X_test)

metrics = {
    "model": best_name,
    "cv_MAE_mean": float(best.value),
    "test_MAE": float(mean_absolute_error(y_test, pred)),
    "test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "test_R2": float(r2_score(y_test, pred)),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "train_cows": int(g_train.nunique()),
    "test_cows": int(groups.iloc[test_idx].nunique()),
    "features": features,
    "best_params": best.params,
}

with open(OUT / "optuna_best_result.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Append the tuned model to the historical comparison without deleting prior results.
comparison_path = OUT / "model_comparison_v2.csv"
new_result = pd.DataFrame([{
    "model": f"{best_name} (Optuna tuned)",
    "MAE_kg_day": metrics["test_MAE"],
    "RMSE_kg_day": metrics["test_RMSE"],
    "R2": metrics["test_R2"],
    "split": "GroupShuffleSplit_25pct_random_state_42",
    "features": ", ".join(features),
}])

if comparison_path.exists():
    old = pd.read_csv(comparison_path)
    old = old[old["model"] != f"{best_name} (Optuna tuned)"]
    new_result = pd.concat([old, new_result], ignore_index=True)

new_result.to_csv(comparison_path, index=False)

print("\nBEST TUNED MODEL:", best_name)
print("Cross-validation MAE:", round(best.value, 4))
print("Held-out test MAE:", round(metrics["test_MAE"], 4))
print("Held-out test RMSE:", round(metrics["test_RMSE"], 4))
print("Held-out test R2:", round(metrics["test_R2"], 4))
print("\nBest parameters:")
print(json.dumps(best.params, indent=2))
print("\nSaved trials:", trials_path)
print("Saved best result:", OUT / "optuna_best_result.json")
print("Updated comparison:", comparison_path)
