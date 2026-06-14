pip install -r requirements.txt


import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import KFold, GridSearchCV 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from IPython.display import display
import warnings
import optuna
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt 
from iapws import IAPWS97
from SALib.sample import saltelli 
from SALib.analyze import sobol


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "Results"
FIGURES_DIR = BASE_DIR / "Figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------
DATA_PATH = "Nuclear Data.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8", na_values=["NA", "NaN", "", " "])
df.columns = [c.strip() for c in df.columns]

# ---------------------------------------------------------------
# Define Reactor_Group 
# ---------------------------------------------------------------

def assign_reactor_group(row):
    rt = row["Reactor_Type"]
    p = row["Thermal_Power"]

    if pd.isna(rt) or pd.isna(p):
        return "Unknown"

    if rt == "PWR":
        if p < 2785:
            return "PWR_Low"
        elif p <= 3020:
            return "PWR_Mid"
        else:
            return "PWR_High"

    elif rt == "BWR":
        if p < 3200:
            return "BWR_Mid" 
        else: 
            return "BWR_High"

    elif rt == "PHWR":
        if p < 1000:
            return "PHWR_Small"
        elif p <= 2200:
            return "PHWR_Standard"
        else:
            return "PHWR_Large"

    else:
        return rt   # GCR, FBR, HTGR etc.

df["Reactor_Group"] = df.apply(assign_reactor_group, axis=1)


expected_cols = [
    "Reactor_Type","Fuel_Enrichment","Core_Diameter","Core_Height",
    "Number_of_Fuel_Assemblies","Fuel_Linear_Heat_Generation_Rate",
    "Control_Rod_Assemblies","Coolant_Pressure","Outlet_Temperature",
    "Thermal_Power","Gross_Electrical_Power"
]

col_map = {}
for ec in expected_cols:
    for c in df.columns:
        if c.lower().replace(" ","_") == ec.lower().replace(" ","_"):
            col_map[c] = ec
df = df.rename(columns=col_map)

y_cols = ["Outlet_Temperature", "Thermal_Power", "Gross_Electrical_Power"]
X_cols = [c for c in expected_cols if c not in y_cols and c != "Reactor_Type"]

for c in X_cols + y_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Reactor_Type"] = df["Reactor_Type"].fillna("Unknown")
df = df.dropna(subset=y_cols, how="all").reset_index(drop=True)

counts = df["Reactor_Group"].value_counts()
valid_groups = counts[counts >= 15].index
dropped_groups = counts[counts < 15].index.tolist()
df = df[df["Reactor_Group"].isin(valid_groups)].reset_index(drop=True)

# ---------------------------------------------------------------
# Custom MLP
# ---------------------------------------------------------------
class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}

# ---------------------------------------------------------------
# Helper: MAPE
# ---------------------------------------------------------------
def mape_numpy(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# ---------------------------------------------------------------
# Optuna search spaces (replicating prior param grids as suggestions)
# ---------------------------------------------------------------

SEARCH_SPACE = {
    "GradientBoosting": {
        "n_estimators": lambda trial: trial.suggest_categorical("n_estimators", [100, 200]),
        "learning_rate": lambda trial: trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "max_depth": lambda trial: trial.suggest_categorical("max_depth", [3, 4, 5]),
        "subsample": lambda trial: trial.suggest_categorical("subsample", [0.8, 1.0])
    },
    "SVM": {
        "C": lambda trial: trial.suggest_categorical("C", [1.0, 10.0, 100.0]),
        "epsilon": lambda trial: trial.suggest_categorical("epsilon", [0.01, 0.1]),
        "kernel": lambda trial: trial.suggest_categorical("kernel", ["linear", "rbf"]) 
    },
    "KNN": {
        "n_neighbors": lambda trial: trial.suggest_categorical("n_neighbors", [3,5,7]),
        "weights": lambda trial: trial.suggest_categorical("weights", ["uniform","distance"]),
        "p": lambda trial: trial.suggest_categorical("p", [1,2])
    },
    "RandomForest": {
        "n_estimators": lambda trial: trial.suggest_categorical("n_estimators", [100,200]),
        "max_depth": lambda trial: trial.suggest_categorical("max_depth", [None, 10, 20]),
        "min_samples_split": lambda trial: trial.suggest_categorical("min_samples_split", [2,5])
    },
    "DecisionTree": {
        "max_depth": lambda trial: trial.suggest_categorical("max_depth", [None, 10, 20]),
        "min_samples_split": lambda trial: trial.suggest_categorical("min_samples_split", [2,5])
    }
}

# ===============================================================
# LOAD SAVED HYPERPARAMETERS IF THEY EXIST
# ===============================================================

HYPERPARAM_FILE = RESULTS_DIR / "Best_Hyperparameters_Optuna.json"

if os.path.exists(HYPERPARAM_FILE):
    print("Loading saved Optuna hyperparameters...")
    with open(HYPERPARAM_FILE, "r") as f:
        best_params_all = json.load(f)

    print("Hyperparameters loaded successfully. Skipping Optuna tuning.")
    RUN_OPTUNA = False
else:
    print("No saved hyperparameters found. Running Optuna tuning...")
    best_params_all = {}
    RUN_OPTUNA = True

# ===============================================================
#  STAGE 1: OPTUNA HYPERPARAMETER TUNING
# ===============================================================
if RUN_OPTUNA:
    best_params_all = {}
    N_TRIALS = 40  # reasonable default; adjust if you want heavier search
    CV_FOLDS = 4

    for reactor, df_sub in df.groupby("Reactor_Group"):
        print(f"Optuna Optimization for Reactor_Group: {reactor} =====")

        X = df_sub[X_cols].copy()
        y = df_sub[y_cols].copy()
        imp_X = SimpleImputer(strategy="mean")
        imp_y = SimpleImputer(strategy="mean")
        X_full_imp = imp_X.fit_transform(X)
        y_full_imp = pd.DataFrame(imp_y.fit_transform(y), columns=y_cols)

        scaler = StandardScaler()

        best_params_all[reactor] = {}

        for target in y_cols:
            print(f" Target: {target}")
            y_target = y_full_imp[target].values
            best_params_all[reactor][target] = {}

            # For each model, create an Optuna study
            for model_name in SEARCH_SPACE.keys():
                print(f"Tuning {model_name} with Optuna...")

                def make_objective(model_name_local):
                    def objective(trial):
                        # sample params using SEARCH_SPACE definitions
                        params = {}
                        for k, sampler in SEARCH_SPACE[model_name_local].items():
                            params[k] = sampler(trial)

                        # build the sklearn model with sampled params
                        if model_name_local == "GradientBoosting":
                            model = GradientBoostingRegressor(random_state=42, **params)
                        elif model_name_local == "SVM":
                            # SVR does not accept None for kernel etc.
                            model = SVR(**params)
                        elif model_name_local == "KNN":
                            model = KNeighborsRegressor(**params)
                        elif model_name_local == "RandomForest":
                            # sklearn doesn't accept None as Python None via kwargs if it's in params already; OK
                            model = RandomForestRegressor(random_state=42, **params)
                        elif model_name_local == "DecisionTree":
                            model = DecisionTreeRegressor(random_state=42, **params)
                        else:
                            raise ValueError("Unknown model")

                        # manual CV to compute MAPE
                        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                        mapes = []

                        for train_idx, test_idx in kf.split(X_full_imp):
                            X_train_raw, X_test_raw = X_full_imp[train_idx], X_full_imp[test_idx]
                            y_train, y_test = y_target[train_idx], y_target[test_idx]

                            # Fit scaler ONLY on training data
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train_raw)
                            X_test = scaler.transform(X_test_raw)

                            m = clone(model)
                            m.fit(X_train, y_train)
                            y_pred = m.predict(X_test)

                            mapes.append(mape_numpy(y_test, y_pred))
                        return float(np.mean(mapes))
                    return objective

                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(make_objective(model_name), n_trials=N_TRIALS, show_progress_bar=False)

                best_params = study.best_params
                best_value = study.best_value

                # store
                best_params_all[reactor][target][model_name] = {
                    "Best_MAPE(%)": round(float(best_value), 6),
                    "Best_Params": best_params
                }

                print(f"      -> Best MAPE%: {best_value:.4f} | Params: {best_params}")

            print(f"Completed tuning for {target}")

        print(f" Done tuning {reactor}")

    # Save tuned hyperparameters
    os.makedirs("./", exist_ok=True)
    with open(RESULTS_DIR / "Best_Hyperparameters_Optuna.json", "w") as f:
        json.dump(best_params_all, f, indent=2)
    print("Saved best hyperparameters to 'Best_Hyperparameters_Optuna.json'")

# ===============================================================
# STAGE 2: MODEL TRAINING & EVALUATION (with Optuna-tuned params)
# ===============================================================

KFOLDS = 4
all_results = {}

for reactor, df_sub in df.groupby("Reactor_Group"):
    model_results= {}
    print(f"Training Models for Reactor_Group: {reactor} =====")

    X = df_sub[X_cols].copy()
    y = df_sub[y_cols].copy()
    imp_X = SimpleImputer(strategy="mean")
    imp_y = SimpleImputer(strategy="mean")
    X_full_imp = imp_X.fit_transform(X)
    y_full_imp = pd.DataFrame(imp_y.fit_transform(y), columns=y_cols)

    # -----------------------------------------------------------
    # Initialize tuned models 
    # -----------------------------------------------------------
    
    def get_params_safe(reactor_local, target_local, model_local):
        try:
            return best_params_all[reactor_local][target_local][model_local]["Best_Params"]
        except Exception:
            return {}

    for model_name in ["DecisionTree", "RandomForest", "GradientBoosting", "KNN", "SVM"]:
        print(f"{model_name} (10 runs avg, {KFOLDS}-fold CV) ---")
        target_results = {}

        for target in y_cols:
            params = get_params_safe(reactor, target, model_name)

            if model_name == "DecisionTree":
                base_model = DecisionTreeRegressor(**params)
            elif model_name == "RandomForest":
                base_model = RandomForestRegressor(**params, random_state=42)
            elif model_name == "GradientBoosting":
                base_model = GradientBoostingRegressor(**params, random_state=42)
            elif model_name == "KNN":
                base_model = KNeighborsRegressor(**params)
            elif model_name == "SVM":
                base_model = SVR(**params)

            metrics_all = []

            for run in range(10):
                kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42 + run)
                fold_metrics = []

                for train_idx, test_idx in kf.split(X_full_imp):
                    X_train_raw, X_test_raw = X_full_imp[train_idx], X_full_imp[test_idx]
                    y_train = y_full_imp.iloc[train_idx][target]
                    y_test  = y_full_imp.iloc[test_idx][target]

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train_raw)
                    X_test = scaler.transform(X_test_raw)

                    m = clone(base_model)
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)

                    fold_metrics.append(list(evaluate(y_test, y_pred).values()))

                metrics_all.append(np.mean(fold_metrics, axis=0))

            metrics_array = np.array(metrics_all)

            mean_vals = np.mean(metrics_array, axis=0)
            std_vals  = np.std(metrics_array, axis=0)

            target_results[target] = {
                "R2": mean_vals[0],
                "R2_std": std_vals[0],
                "RMSE": mean_vals[1],
                "RMSE_std": std_vals[1],
                "MAE": mean_vals[2],
                "MAE_std": std_vals[2],
                "MAPE(%)": mean_vals[3],
                "MAPE_std": std_vals[3],
            }

        model_results[model_name] = pd.DataFrame(target_results).T

    # -----------------------------------------------------------
    # Custom MLP
    # -----------------------------------------------------------
    print(" CustomMLP (100 epochs, 4-fold CV) ---")
    target_results = {}
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    scaler_X = StandardScaler()

    for target in y_cols:
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_full_imp)):
            
            X_train_raw = X_full_imp[train_idx]
            X_test_raw  = X_full_imp[test_idx]
            X_train = scaler_X.fit_transform(X_train_raw)
            X_test  = scaler_X.transform(X_test_raw)
            y_train, y_test = y_full_imp.iloc[train_idx][target], y_full_imp.iloc[test_idx][target]

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1))
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1,1))

            input_size = X_train.shape[1]
            output_size = 1
            model = CustomMLP(input_size, output_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
            X_test_t  = torch.tensor(X_test, dtype=torch.float32)

            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_t).numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            fold_metrics.append(list(evaluate(y_test.values, y_pred.ravel()).values()))

        metrics_array = np.array(fold_metrics)

        mean_vals = np.mean(metrics_array, axis=0)
        std_vals  = np.std(metrics_array, axis=0)

        target_results[target] = {
            "R2": mean_vals[0],
            "R2_std": std_vals[0],
            "RMSE": mean_vals[1],
            "RMSE_std": std_vals[1],
            "MAE": mean_vals[2],
            "MAE_std": std_vals[2],
            "MAPE(%)": mean_vals[3],
            "MAPE_std": std_vals[3],
        }

    model_results["CustomMLP"] = pd.DataFrame(target_results).T

    # -----------------------------------------------------------
    # Save per-reactor results
    # -----------------------------------------------------------
    reactor_filename = f"Results_{reactor.replace('/', '_')}_Optuna.xlsx"
    with pd.ExcelWriter(reactor_filename) as writer:
        for name, df_res in model_results.items():
            df_res.to_excel(writer, sheet_name=name[:31])
    print(f"Saved results for {reactor} → {reactor_filename}")

    all_results[reactor] = model_results

# ---------------------------------------------------------------
# Display and save best hyperparameters 
# ---------------------------------------------------------------
print("BEST HYPERPARAMETERS SUMMARY (Optuna) =====")
summary_rows = []
for reactor, targets in best_params_all.items():
    for target, models in targets.items():
        for model_name, info in models.items():
            summary_rows.append({
                "Reactor": reactor,
                "Target": target,
                "Model": model_name,
                "Best_MAPE(%)": info["Best_MAPE(%)"],
                "Best_Params": info["Best_Params"]
            })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)
summary_df.to_csv(RESULTS_DIR / "Best_Hyperparameter_Summary_Optuna.csv", index=False)
with open(RESULTS_DIR / "Best_Hyperparameters_Optuna.json", "w") as f:
    json.dump(best_params_all, f, indent=2)

print("Saved summary to 'Best_Hyperparameter_Summary_Optuna.csv' and 'Best_Hyperparameters_Optuna.json'")

print("All processing, Optuna tuning, and evaluation complete!")

# ---------------------------------------------------------------
# Hybrid Stacking Model 
# ---------------------------------------------------------------


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Cache file
# ---------------------------------------------------------------
STACKING_FILE = RESULTS_DIR / "Best_Stacking_Results.json"

if os.path.exists(STACKING_FILE):
    print("Loading saved stacking results...")
    with open(STACKING_FILE, "r") as f:
        saved_data = json.load(f)

    summary_rows = saved_data["summary"]
    details = saved_data.get("details", {})
    RUN_STACKING = False

else:
    print("No saved stacking results found. Running stacking...")
    summary_rows = []
    details = {}
    RUN_STACKING = True


# ---------------------------------------------------------------
# Load dataset 
# ---------------------------------------------------------------
DATA_PATH = "Nuclear Data.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8", na_values=["NA", "NaN", "", " "])
df.columns = [c.strip() for c in df.columns]


# ---------------------------------------------------------------
# Reactor group 
# ---------------------------------------------------------------
def assign_reactor_group(row):
    rt = row["Reactor_Type"]
    p = row["Thermal_Power"]

    if pd.isna(rt) or pd.isna(p):
        return "Unknown"

    if rt == "PWR":
        if p < 2785:
            return "PWR_Low"
        elif p <= 3020:
            return "PWR_Mid"
        else:
            return "PWR_High"

    elif rt == "BWR":
        if p < 3200:
            return "BWR_Mid"
        else:
            return "BWR_High"

    elif rt == "PHWR":
        if p < 1000:
            return "PHWR_Small"
        elif p <= 2200:
            return "PHWR_Standard"
        else:
            return "PHWR_Large"

    else:
        return rt


df["Reactor_Group"] = df.apply(assign_reactor_group, axis=1)


# ---------------------------------------------------------------
# Feature / target setup 
# ---------------------------------------------------------------
X_cols = [
    "Fuel_Enrichment","Core_Diameter","Core_Height",
    "Number_of_Fuel_Assemblies","Fuel_Linear_Heat_Generation_Rate",
    "Control_Rod_Assemblies","Coolant_Pressure"
]

targets = ["Outlet_Temperature","Thermal_Power","Gross_Electrical_Power"]

df = df.dropna(subset=["Reactor_Group"]).reset_index(drop=True)

valid = df["Reactor_Group"].value_counts()
df = df[df["Reactor_Group"].isin(valid[valid >= 15].index)].reset_index(drop=True)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return r2, rmse, mae, mape


# ---------------------------------------------------------------
# Base models using BEST_PARAMS
# ---------------------------------------------------------------
def build_base_models(reactor, target, best_params_all):

    return [
        ("DecisionTree",
         DecisionTreeRegressor(**best_params_all[reactor][target]["DecisionTree"]["Best_Params"])),

        ("RandomForest",
         RandomForestRegressor(**best_params_all[reactor][target]["RandomForest"]["Best_Params"],
                               random_state=42)),

        ("GradientBoosting",
         GradientBoostingRegressor(**best_params_all[reactor][target]["GradientBoosting"]["Best_Params"],
                                   random_state=42)),

        ("KNN",
         KNeighborsRegressor(**best_params_all[reactor][target]["KNN"]["Best_Params"])),

        ("SVM",
         SVR(**best_params_all[reactor][target]["SVM"]["Best_Params"]))
    ]


# ---------------------------------------------------------------
# Core stacking evaluation
# ---------------------------------------------------------------
def run_stacking(df, reactor, target, best_params_all):

    sub = df[df["Reactor_Group"] == reactor].copy()

    X = sub[X_cols].copy()
    y = sub[target].copy()

    imp_X = SimpleImputer(strategy="mean")
    imp_y = SimpleImputer(strategy="mean")

    X = imp_X.fit_transform(X)
    y = imp_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    base_models = build_base_models(reactor, target, best_params_all)

    meta_alphas = [0.01, 0.1, 1, 10]

    run_metrics = []

    for run in range(10):

        fold_scores = []

        for train_idx, test_idx in kf.split(X):

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            estimators = [(n, clone(m)) for n, m in base_models]

            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=4,
                n_jobs=-1
            )

            gs = GridSearchCV(
                stack,
                param_grid={"final_estimator__alpha": meta_alphas},
                scoring="r2",
                cv=3,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            pred = best_model.predict(X_test)

            fold_scores.append(evaluate(y_test, pred))

        run_metrics.append(np.mean(fold_scores, axis=0))

    run_metrics = np.array(run_metrics)

    mean_vals = run_metrics.mean(axis=0)
    std_vals = run_metrics.std(axis=0)

    return {
        "R2": mean_vals[0],
        "R2_std": std_vals[0],
        "RMSE": mean_vals[1],
        "RMSE_std": std_vals[1],
        "MAE": mean_vals[2],
        "MAE_std": std_vals[2],
        "MAPE(%)": mean_vals[3],
        "MAPE_std": std_vals[3]
    }


# ---------------------------------------------------------------
# RUN STACKING
# ---------------------------------------------------------------
if RUN_STACKING:

    for reactor in df["Reactor_Group"].unique():

        for target in targets:

            try:
                res = run_stacking(df, reactor, target, best_params_all)

                summary_rows.append({
                    "Reactor": reactor,
                    "Target": target,
                    **res
                })

                print(f"✔ {reactor} - {target} done | R2={res['R2']:.3f}")

            except Exception as e:
                print(f"⚠ Skipped {reactor}/{target}: {e}")


    # -----------------------------------------------------------
    # Save results 
    # -----------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("Best_Stacking_Summary.csv", index=False)

    with open(STACKING_FILE, "w") as f:
        json.dump({
            "summary": summary_rows,
            "details": details
        }, f, indent=2)

    print("Stacking results saved.")

else:
    summary_df = pd.DataFrame(summary_rows)


# ---------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------
display(summary_df)

# =========================================================
# UNIFIED MODEL RESULTS LOGGER (BASE + MLP + STACKING)
# =========================================================

ALL_RESULTS_FILE = RESULTS_DIR / "All_Model_Results.json"
ALL_RESULTS_CSV  = RESULTS_DIR / "All_Model_Results.csv"

# ---------------------------------------------------------
# Initialize / Load existing unified results
# ---------------------------------------------------------
if os.path.exists(ALL_RESULTS_FILE):
    with open(ALL_RESULTS_FILE, "r") as f:
        all_results_list = json.load(f)
    print("Loaded existing unified results...")
else:
    all_results_list = []
    print("Creating new unified results log...")

# ---------------------------------------------------------
# Helper to safely append (avoid duplicates)
# ---------------------------------------------------------
def add_result(entry, results_list):
    key = (entry["Reactor"], entry["Target"], entry["Model"])

    existing_keys = {
        (r["Reactor"], r["Target"], r["Model"])
        for r in results_list
    }

    if key not in existing_keys:
        results_list.append(entry)


# =========================================================
# BASE MODELS + MLP (from all_results variable)
# =========================================================

for reactor, model_dict in all_results.items():

    for model_name, df_res in model_dict.items():

        for target in df_res.index:

            row = df_res.loc[target]

            add_result({
                "Reactor": reactor,
                "Target": target,
                "Model": model_name,
                "R2": float(row["R2"]),
                "R2_std": float(row.get("R2_std", np.nan)),
                "RMSE": float(row["RMSE"]),
                "RMSE_std": float(row.get("RMSE_std", np.nan)),
                "MAE": float(row["MAE"]),
                "MAE_std": float(row.get("MAE_std", np.nan)),
                "MAPE": float(row["MAPE(%)"]),
                "MAPE_std": float(row.get("MAPE_std", np.nan))
            }, all_results_list)


# =========================================================
# STACKING RESULTS (from summary_df)
# =========================================================

for _, row in summary_df.iterrows():

    add_result({
        "Reactor": row["Reactor"],
        "Target": row["Target"],
        "Model": "Stacking",
        "R2": float(row["R2"]),
        "R2_std": float(row.get("R2_std", np.nan)),
        "RMSE": float(row["RMSE"]),
        "RMSE_std": float(row.get("RMSE_std", np.nan)),
        "MAE": float(row["MAE"]),
        "MAE_std": float(row.get("MAE_std", np.nan)),
        "MAPE": float(row["MAPE(%)"]),
        "MAPE_std": float(row.get("MAPE_std", np.nan))
    }, all_results_list)


# =========================================================
# SAVE UNIFIED RESULTS
# =========================================================

# Save JSON
with open(ALL_RESULTS_FILE, "w") as f:
    json.dump(all_results_list, f, indent=2)

# Save CSV
df_all_results = pd.DataFrame(all_results_list)
df_all_results.to_csv(ALL_RESULTS_CSV, index=False)

print("✅ Unified model results saved:")
print(f"   → {ALL_RESULTS_FILE}")
print(f"   → {ALL_RESULTS_CSV}")

# ---------------------------------------------------------
# Optional: Display preview
# ---------------------------------------------------------
display(df_all_results.head())

# ==============================================================
# 📁 STACKING INTERPRETABILITY EXPORT 
# ==============================================================

OUTPUT_DIR = RESULTS_DIR / "Stacking_Interpretability"
OUTPUT_DIR.mkdir(exist_ok=True)

meta_csv_path = OUTPUT_DIR / "Meta_Learner_Coefficients_Optuna.csv"
meta_json_path = OUTPUT_DIR / "Meta_Learner_Coefficients_Optuna.json"
best_stack_json = OUTPUT_DIR / "Best_Stacking_Models.json"

if (
    meta_csv_path.exists()
    and meta_json_path.exists()
    and best_stack_json.exists()
):
    print("✅ Interpretability outputs already exist. Skipping recomputation.")
    print(f"📁 Folder: {OUTPUT_DIR}")
else:
    print("⚙️ Running stacking interpretability export...")


    meta_results = []

    meta_alphas = [0.01, 0.1, 1.0, 10.0]

    for _, row in summary_df.iterrows():

        reactor = row["Reactor"]
        target = row["Target"]

        # ---------------------------------------------------------
        # Compute BEST META ALPHA 
        # ---------------------------------------------------------
        sub = df[df["Reactor_Group"] == reactor].copy()

        X = sub[X_cols].copy()
        y = sub[target].copy()

        imp_X = SimpleImputer(strategy="mean")
        imp_y = SimpleImputer(strategy="mean")

        X = imp_X.fit_transform(X)
        y = imp_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        base_models = build_base_models(reactor, target, best_params_all)

        best_alpha = None
        best_score = -np.inf

        for alpha in meta_alphas:

            scores = []

            for train_idx, test_idx in kf.split(X):

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test  = scaler.transform(X[test_idx])

                y_train, y_test = y[train_idx], y[test_idx]

                estimators = [(n, clone(m)) for n, m in base_models]

                stack = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(alpha=alpha),
                    cv=4,
                    n_jobs=-1
                )

                stack.fit(X_train, y_train)
                pred = stack.predict(X_test)

                scores.append(r2_score(y_test, pred))

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # ---------------------------------------------------------
        # Train FINAL stacking model on FULL DATA
        # ---------------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        estimators = [(n, clone(m)) for n, m in base_models]

        final_stack = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=best_alpha),
            cv=4,
            n_jobs=-1
        )

        final_stack.fit(X_scaled, y)

        meta_model = final_stack.final_estimator_
        coefs = getattr(meta_model, "coef_", None)

        if coefs is None:
            continue

        # ---------------------------------------------------------
        # Save per-model coefficients
        # ---------------------------------------------------------
        for name, coef in zip([n for n, _ in base_models], coefs.ravel()):

            meta_results.append({
                "Reactor": reactor,
                "Target": target,
                "Base_Model": name,
                "Meta_Coefficient": float(coef),
                "Best_Meta_Alpha": float(best_alpha),
                "Stacking_R2": float(row["R2"]),
                "Stacking_RMSE": float(row["RMSE"]),
                "Stacking_MAE": float(row["MAE"]),
                "Stacking_MAPE": float(row["MAPE(%)"])
            })

    # ==============================================================
    #  SAVE FILES
    # ==============================================================

    meta_df = pd.DataFrame(meta_results)

    # CSV (required)
    meta_df.to_csv(meta_csv_path, index=False)

    # JSON (NEW for PINN / reuse)
    with open(meta_json_path, "w") as f:
        json.dump(meta_results, f, indent=2)

    # BEST STACKING MODEL JSON (NEW)

    best_models = {}
    for r in meta_results:
        key = (r["Reactor"], r["Target"])
        best_models.setdefault(str(key), {
            "Best_Meta_Alpha": r["Best_Meta_Alpha"],
            "Model_Weights": {}
        })
        best_models[str(key)]["Model_Weights"][r["Base_Model"]] = r["Meta_Coefficient"]

    with open(best_stack_json, "w") as f:
        json.dump(best_models, f, indent=2)

    print("\n✅ CLEAN stacking interpretability export complete!")
    print(f"📁 Folder: {OUTPUT_DIR}")
    print("- Meta_Learner_Coefficients_Optuna.csv")
    print("- Meta_Learner_Coefficients_Optuna.json")
    print("- Best_Stacking_Models.json")

    
HEATMAP_DIR = FIGURES_DIR / "Heatmaps"
HEATMAP_DIR.mkdir(exist_ok=True)

# Folder to save heatmaps (UNCHANGED)
output_folder = HEATMAP_DIR
os.makedirs(output_folder, exist_ok=True)

# ===============================================================
# LOAD NEW UNIFIED FILES
# ===============================================================

# Performance results (BASE + MLP + STACKING)
performance_all = pd.read_csv("All_Model_Results.csv")

# Meta-learner coefficients
with open("Stacking_Interpretability/Meta_Learner_Coefficients_Optuna.json", "r") as f:
    meta_results = pd.DataFrame(json.load(f))

# ===============================================================
# META-LEARNER COEFFICIENT HEATMAP
# ===============================================================

coef_heatmap = meta_results.pivot_table(
    index="Base_Model",
    columns="Reactor",
    values="Meta_Coefficient",
    aggfunc="mean"
)

# Sort by importance (same logic as before)
coef_order = coef_heatmap.abs().mean(axis=1).sort_values(ascending=False).index
coef_heatmap = coef_heatmap.loc[coef_order]

plt.figure(figsize=(12, 5))
sns.heatmap(coef_heatmap, annot=True, fmt=".2f", cmap="coolwarm", center=0)

plt.title("Meta-Learner Ridge Coefficients per Reactor Group", fontsize=14)
plt.xlabel("Reactor Group")
plt.ylabel("Base Model")

plt.tight_layout()
plt.savefig(
    HEATMAP_DIR / "MetaLearner_Coefficients.png",
    dpi=300
)
plt.close()

print("✅ Meta-Learner coefficients heatmap saved.")
meta_df = pd.DataFrame(meta_results)

# ===============================================================
# PERFORMANCE HEATMAPS (BASE + MLP + STACKING)
# ===============================================================

metrics = ["R2", "RMSE", "MAE", "MAPE"]

model_order = performance_all["Model"].unique()
reactor_order = performance_all["Reactor"].unique()

for metric in metrics:

    perf_heatmap = performance_all.pivot_table(
        index="Model",
        columns="Reactor",
        values=metric,
        aggfunc="mean"
    )

    # align ordering (important for consistency across plots)
    perf_heatmap = perf_heatmap.reindex(index=model_order, columns=reactor_order)

    plt.figure(figsize=(12, 5))

    # same color logic as your original code
    cmap_choice = "YlGnBu" if metric != "R2" else "YlOrRd"

    sns.heatmap(perf_heatmap, annot=True, fmt=".3f", cmap=cmap_choice)

    plt.title(f"Performance Metric: {metric} per Reactor Group", fontsize=14)
    plt.xlabel("Reactor Group")
    plt.ylabel("Model")

    plt.tight_layout()
    plt.savefig(
        HEATMAP_DIR / f"Performance_{metric}.png",
        dpi=300
    )
    plt.close()

    print(f"✅ Performance heatmap for {metric} saved.")

    import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Folder to save heatmaps
output_folder = HEATMAP_DIR
os.makedirs(output_folder, exist_ok=True)

# ===============================================================
# META-LAYER COEFFICIENT HEATMAP
# ===============================================================

coef_heatmap = meta_df.pivot_table(
    index="Base_Model",
    columns="Reactor",
    values="Meta_Coefficient",
    aggfunc="mean"
)

coef_heatmap = coef_heatmap.fillna(0)

coef_order = coef_heatmap.abs().mean(axis=1).sort_values(ascending=False).index
coef_heatmap = coef_heatmap.loc[coef_order]

plt.figure(figsize=(12, 5))
sns.heatmap(coef_heatmap, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Meta-Learner Ridge Coefficients per Reactor Group", fontsize=14)
plt.xlabel("Reactor Group")
plt.ylabel("Base Model")
plt.tight_layout()
plt.savefig(
    HEATMAP_DIR / "MetaLearner_Coefficients.png",
    dpi=300
)
plt.close()

print("✅ Meta-Learner coefficients heatmap saved.")


# ===============================================================
# PERFORMANCE HEATMAPS 
# ===============================================================

metrics = ["R2", "RMSE", "MAE", "MAPE"]

model_order = df_all_results["Model"].unique()
reactor_order = df_all_results["Reactor"].unique()

for metric in metrics:

    perf_heatmap = df_all_results.pivot_table(
        index="Model",
        columns="Reactor",
        values=metric,
        aggfunc="mean"
    )

    perf_heatmap = perf_heatmap.reindex(index=model_order, columns=reactor_order)

    plt.figure(figsize=(12, 5))
    cmap_choice = "YlGnBu" if metric != "R2" else "YlOrRd"

    sns.heatmap(perf_heatmap, annot=True, fmt=".3f", cmap=cmap_choice)

    plt.title(f"Performance Metric: {metric} per Reactor Group", fontsize=14)
    plt.xlabel("Reactor Group")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(
        HEATMAP_DIR / f"Performance_{metric}.png",
        dpi=300
    )
    plt.close()

    print(f"✅ Performance heatmap for {metric} saved.")

    
# ===============================================================
# BASE PATHS
# ===============================================================

BASE_FOLDER = FIGURES_DIR / "Heatmaps_Per_Output"
BASE_FOLDER.mkdir(exist_ok=True)

# ===============================================================
# LOAD DATA
# ===============================================================

performance_all = pd.read_csv(
    RESULTS_DIR / "All_Model_Results.csv"
)

with open(
    RESULTS_DIR
    / "Stacking_Interpretability"
    / "Meta_Learner_Coefficients_Optuna.json",
    "r"
) as f:
    meta_results = pd.DataFrame(json.load(f))

# ===============================================================
# FIXED ORDERING (GLOBAL CONSISTENCY)
# ===============================================================

reactor_order = [
    "PHWR_Standard",
    "PWR_Low",
    "PWR_Mid",
    "PWR_High",
    "BWR_Mid",
    "BWR_High"
]

model_order = [
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "KNN",
    "SVM",
    "CustomMLP",
    "Stacking"
]

outputs = performance_all["Target"].unique()

# ===============================================================
# LOOP PER OUTPUT
# ===============================================================

for target in outputs:

    print(f"Generating heatmaps for {target}...")

    out_dir = BASE_FOLDER / target
    out_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------
    # Filter data per output
    # -----------------------------------------------------------
    perf_sub = performance_all[performance_all["Target"] == target]
    meta_sub = meta_results[meta_results["Target"] == target]

    # ===========================================================
    # Meta-coefficient heat map
    # ===========================================================

    coef_heatmap = meta_sub.pivot_table(
        index="Base_Model",
        columns="Reactor",
        values="Meta_Coefficient",
        aggfunc="mean"
    ).fillna(0)

    coef_heatmap = coef_heatmap.reindex(
        index=model_order,
        columns=reactor_order
    )

    plt.figure(figsize=(12, 5))
    sns.heatmap(coef_heatmap, annot=True, fmt=".2f",
                cmap="coolwarm", center=0)

    plt.title(f"{target} - Meta-Learner Coefficients")
    plt.xlabel("Reactor Group")
    plt.ylabel("Base Model")

    plt.tight_layout()
    plt.savefig(
        out_dir / "MetaLearner_Coefficients.png",
        dpi=300
    )
    plt.close()

    # ===========================================================
    # Performance Heatmaps
    # ===========================================================

    metrics = ["R2", "RMSE", "MAE", "MAPE"]

    for metric in metrics:

        perf_heatmap = perf_sub.pivot_table(
            index="Model",
            columns="Reactor",
            values=metric,
            aggfunc="mean"
        )

        perf_heatmap = perf_heatmap.reindex(
            index=model_order,
            columns=reactor_order
        )

        plt.figure(figsize=(12, 5))

        cmap = "YlOrRd" if metric == "R2" else "YlGnBu"

        sns.heatmap(perf_heatmap, annot=True, fmt=".3f", cmap=cmap)

        plt.title(f"{target} - {metric}")
        plt.xlabel("Reactor Group")
        plt.ylabel("Model")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"Performance_{metric}.png",
            dpi=300
        )
        plt.close()

    print(f"✅ Completed: {target}")

    # ===============================================================
# 🔬 Nuclear Reactor PINN Evaluation with Extended Grouping
# ===============================================================

warnings.filterwarnings("ignore")

# ===============================================================
# Create folder for heatmaps
# ===============================================================


# ===============================================================
# Load Data
# ===============================================================
DATA_PATH = "Nuclear Data.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8", na_values=["NA","NaN",""," "])
df.columns = [c.strip() for c in df.columns]

expected_cols = [
    "Reactor_Type","Fuel_Enrichment","Core_Diameter","Core_Height",
    "Number_of_Fuel_Assemblies","Fuel_Linear_Heat_Generation_Rate",
    "Control_Rod_Assemblies","Coolant_Pressure","Outlet_Temperature",
    "Thermal_Power","Gross_Electrical_Power"
]

col_map = {}
for ec in expected_cols:
    for c in df.columns:
        if c.lower().replace(" ","_") == ec.lower().replace(" ","_"):
            col_map[c] = ec
df = df.rename(columns=col_map)

y_cols = ["Outlet_Temperature","Thermal_Power","Gross_Electrical_Power"]
X_cols = [c for c in expected_cols if c not in y_cols and c != "Reactor_Type"]

for c in X_cols + y_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Reactor_Type"] = df["Reactor_Type"].fillna("Unknown")
df = df.dropna(subset=y_cols, how="all").reset_index(drop=True)

# ===============================================================
# Reactor grouping by type + power level
# ===============================================================
def assign_reactor_group(row):
    rt = row["Reactor_Type"]
    p = row["Thermal_Power"]

    if pd.isna(rt) or pd.isna(p):
        return "Unknown"

    if rt == "PWR":
        if p < 2785:
            return "PWR_Low"
        elif p <= 3020:
            return "PWR_Mid"
        else:
            return "PWR_High"

    elif rt == "BWR":
        if p < 3200:
            return "BWR_Mid" 
        else: 
            return "BWR_High"

    elif rt == "PHWR":
        if p < 1000:
            return "PHWR_Small"
        elif p <= 2200:
            return "PHWR_Standard"
        else:
            return "PHWR_Large"

    else:
        return rt   # GCR, FBR, HTGR etc.

df["Reactor_Group"] = df.apply(assign_reactor_group, axis=1)

counts = df["Reactor_Group"].value_counts()
valid_groups = counts[counts >= 15].index
df = df[df["Reactor_Group"].isin(valid_groups)].reset_index(drop=True)

# ===============================================================
# Metrics
# ===============================================================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred)/denom))*100
    return {"R2":r2,"RMSE":rmse,"MAE":mae,"MAPE(%)":mape}

# ===============================================================
# PINN Model
# ===============================================================
class ReactorPINN(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,64), nn.Tanh(),
            nn.Linear(64,output_size)
        )
    def forward(self,x):
        return self.net(x)

# ===============================================================
# Physics Loss Functions
# ===============================================================
def saturation_temperature(P_mpa):
    P_mpa = np.clip(P_mpa, 0.01, 22.064)

    water = IAPWS97(P=P_mpa, x=0)
    return water.T - 273.15

def physics_loss(model, X, y_pred, lambdas):

    idx = {col:i for i,col in enumerate(X_cols)}

    T_out = y_pred[:,0]
    P_th  = y_pred[:,1]
    P_el  = y_pred[:,2]

    enrichment = X[:,idx["Fuel_Enrichment"]]
    diameter   = X[:,idx["Core_Diameter"]]
    height     = X[:,idx["Core_Height"]]
    assemblies = X[:,idx["Number_of_Fuel_Assemblies"]]
    lhgr       = X[:,idx["Fuel_Linear_Heat_Generation_Rate"]]
    rods       = X[:,idx["Control_Rod_Assemblies"]]
    pressure   = X[:,idx["Coolant_Pressure"]]
    pressure = torch.clamp(pressure, 0.01, 22.064)

    # 1️⃣ Efficiency constraint
    eta = 0.33
    L_eff = torch.mean((P_el - eta * P_th)**2)

    # 2️⃣ Saturation margin constraint
    T_sat = torch.tensor(
    [saturation_temperature(p.item()) for p in pressure],
    dtype=torch.float32,
    device=pressure.device)
    margin = 10.0
    violation = torch.relu(T_out - (T_sat + margin))
    L_sat = torch.mean(violation**2)


    # 3️⃣ Monotonicity via gradients
    grads_T = torch.autograd.grad(T_out.sum(), X, create_graph=True)[0]
    grads_Pth = torch.autograd.grad(P_th.sum(), X, create_graph=True)[0]
    grads_Pel = torch.autograd.grad(P_el.sum(), X, create_graph=True)[0]

    L_mono = 0

    # Thermal Power constraints
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Fuel_Linear_Heat_Generation_Rate"]]))
    L_mono += torch.mean(torch.relu(grads_Pth[:, idx["Control_Rod_Assemblies"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Number_of_Fuel_Assemblies"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Core_Diameter"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Core_Height"]]))

    # Outlet Temperature constraints
    L_mono += torch.mean(torch.relu(-grads_T[:, idx["Fuel_Linear_Heat_Generation_Rate"]]))

    total = (
        lambdas["eff"]*L_eff +
        lambdas["sat"]*L_sat +
        lambdas["mono"]*L_mono
    )

    return total

# ===============================================================
# Training per Reactor Group
# ===============================================================
pinn_results = {}

for group, df_sub in df.groupby("Reactor_Group"):

    print(f"\n===== ⚙️ Reactor Group: {group} =====")

    X = df_sub[X_cols]
    y = df_sub[y_cols]

    X_imp = SimpleImputer(strategy="mean").fit_transform(X)
    y_imp = SimpleImputer(strategy="mean").fit_transform(y)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_imp)
    y_scaled = scaler_y.fit_transform(y_imp)

    X_t = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)

    model = ReactorPINN(input_size=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambdas = {"eff":0.5,"sat":0.5,"mono":0.5}

    for epoch in range(300):
        optimizer.zero_grad()
        y_pred = model(X_t)

        data_loss = nn.MSELoss()(y_pred, y_t)
        phys_loss = physics_loss(model, X_t, y_pred, lambdas)

        loss = data_loss + phys_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    metrics = {}
    for i,target in enumerate(y_cols):
        metrics[target] = evaluate(y_imp[:,i], y_pred[:,i])

    pinn_results[group] = pd.DataFrame(metrics).T

for group, df_pinn in pinn_results.items():
    if group not in all_results:
        # If the group didn’t exist before (unlikely), just add it
        all_results[group] = {}
    # Add PINN as another "model"
    all_results[group]["PINN"] = df_pinn
    # =========================================================
    # 🔹 ADD PINN RESULTS TO UNIFIED RESULTS LOGGER
    # =========================================================

    # Load existing unified results
    ALL_RESULTS_FILE = RESULTS_DIR / "All_Model_Results.json"
    ALL_RESULTS_CSV  = RESULTS_DIR / "All_Model_Results.csv"
    if os.path.exists(ALL_RESULTS_FILE):
        with open(ALL_RESULTS_FILE, "r") as f:
            all_results_list = json.load(f)
    else:
        all_results_list = []

    # Helper to avoid duplicates
    def add_result(entry, results_list):
        key = (entry["Reactor"], entry["Target"], entry["Model"])
        existing_keys = {
            (r["Reactor"], r["Target"], r["Model"])
            for r in results_list
        }
        if key not in existing_keys:
            results_list.append(entry)

    # Add PINN results
    for reactor, df_pinn in pinn_results.items():
        for target in df_pinn.index:
            row = df_pinn.loc[target]

            add_result({
                "Reactor": reactor,
                "Target": target,
                "Model": "PINN",
                "R2": float(row["R2"]),
                "R2_std": np.nan,
                "RMSE": float(row["RMSE"]),
                "RMSE_std": np.nan,
                "MAE": float(row["MAE"]),
                "MAE_std": np.nan,
                "MAPE": float(row["MAPE(%)"]),
                "MAPE_std": np.nan
            }, all_results_list)

    # Save updated unified results
    with open(ALL_RESULTS_FILE, "w") as f:
        json.dump(all_results_list, f, indent=2)

    df_all_results = pd.DataFrame(all_results_list)
    df_all_results.to_csv(ALL_RESULTS_CSV, index=False)

    print("✅ PINN results appended to unified results files!")

# ===============================================================
# 🔹 SAVE PINN RESULTS TO CSV + JSON 
# ===============================================================

import json
import os

PINN_RESULTS_DIR = RESULTS_DIR / "PINN"
PINN_RESULTS_DIR.mkdir(exist_ok=True)

PINN_RESULTS_JSON = PINN_RESULTS_DIR / "PINN_Results.json"
PINN_RESULTS_CSV  = PINN_RESULTS_DIR / "PINN_Results.csv"

pinn_rows = []

for reactor, df_pinn in pinn_results.items():

    for target in df_pinn.index:

        row = df_pinn.loc[target]

        pinn_rows.append({
            "Reactor": reactor,
            "Target": target,
            "Model": "PINN",

            "R2": float(row["R2"]),
            "R2_std": np.nan,

            "RMSE": float(row["RMSE"]),
            "RMSE_std": np.nan,

            "MAE": float(row["MAE"]),
            "MAE_std": np.nan,

            # IMPORTANT:
            # use SAME naming convention everywhere
            "MAPE": float(row["MAPE(%)"]),
            "MAPE_std": np.nan
        })

# ---------------------------------------------------------------
# Save standalone PINN files
# ---------------------------------------------------------------

pinn_df = pd.DataFrame(pinn_rows)

pinn_df.to_csv(PINN_RESULTS_CSV, index=False)

with open(PINN_RESULTS_JSON, "w") as f:
    json.dump(pinn_rows, f, indent=2)

print("✅ PINN standalone files saved.")

# ===============================================================
# 🔹 APPEND PINN TO UNIFIED RESULTS
# ===============================================================

ALL_RESULTS_FILE = RESULTS_DIR / "All_Model_Results.json"
ALL_RESULTS_CSV  = RESULTS_DIR / "All_Model_Results.csv"

# ---------------------------------------------------------------
# Load existing results
# ---------------------------------------------------------------

if ALL_RESULTS_FILE.exists():

    with open(ALL_RESULTS_FILE, "r") as f:
        all_results_list = json.load(f)

else:
    all_results_list = []

# ---------------------------------------------------------------
# Avoid duplicates
# ---------------------------------------------------------------

existing_keys = {
    (r["Reactor"], r["Target"], r["Model"])
    for r in all_results_list
}

for row in pinn_rows:

    key = (row["Reactor"], row["Target"], row["Model"])

    if key not in existing_keys:
        all_results_list.append(row)

# ---------------------------------------------------------------
# Save updated unified results
# ---------------------------------------------------------------

with open(ALL_RESULTS_JSON, "w") as f:
    json.dump(all_results_list, f, indent=2)

df_all_results = pd.DataFrame(all_results_list)

df_all_results.to_csv(ALL_RESULTS_CSV, index=False)

print("✅ PINN results appended to unified results.")

# ===============================================================
# Heatmaps (All Metrics)
# ===============================================================

PINN_HEATMAP_DIR = FIGURES_DIR / "PINN_Heatmaps"
PINN_HEATMAP_DIR.mkdir(exist_ok=True)

# Fixed reactor order
reactor_order = ["PHWR_Standard", "PWR_Low", "PWR_Mid", "PWR_High", "BWR_Mid", "BWR_High"]

metric_names = ["R2","RMSE","MAE","MAPE(%)"]

for metric in metric_names:

    heatmap_data = pd.DataFrame({
        group: df_metric[metric]
        for group, df_metric in pinn_results.items()
    }).T

    # enforce consistent reactor order
    heatmap_data = heatmap_data.reindex(reactor_order)

    plt.figure(figsize=(14,7))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")

    plt.title(f"{metric} Performance of PINN per Reactor Group")
    plt.ylabel("Reactor Group")
    plt.xlabel("Target Variable")

    plt.tight_layout()
    plt.savefig(
        PINN_HEATMAP_DIR / f"PINN_{metric}_Heatmap.png",
        dpi=300
    )
    plt.close()

print("✅ PINN heatmaps saved inside 'PINN_Heatmap' folder!")

# ===============================================================
# HEATMAP GENERATION
# ===============================================================

COMPLETE_FOLDER = FIGURES_DIR / "Complete Heatmaps"
os.makedirs(COMPLETE_FOLDER, exist_ok=True)

# ===============================================================
# LOAD COMPLETE RESULTS
# ===============================================================

performance_all = pd.read_csv("All_Model_Results.csv")

with open(
    RESULTS_DIR
    / "Stacking_Interpretability"
    / "Meta_Learner_Coefficients_Optuna.json",
    "r"
) as f:

    meta_results = pd.DataFrame(json.load(f))

# ===============================================================
# FIXED ORDERING
# ===============================================================

reactor_order = [
    "PHWR_Standard",
    "PWR_Low",
    "PWR_Mid",
    "PWR_High",
    "BWR_Mid",
    "BWR_High"
]

model_order = [
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "KNN",
    "SVM",
    "CustomMLP",
    "Stacking",
    "PINN"
]

metrics = ["R2", "RMSE", "MAE", "MAPE"]

outputs = performance_all["Target"].unique()

# ===============================================================
# GLOBAL PERFORMANCE HEATMAPS
# ===============================================================

for metric in metrics:

    perf_heatmap = performance_all.pivot_table(
        index="Model",
        columns="Reactor",
        values=metric,
        aggfunc="mean"
    )

    perf_heatmap = perf_heatmap.reindex(
        index=model_order,
        columns=reactor_order
    )

    plt.figure(figsize=(12,5))

    # SAME STYLE AS PREVIOUS HEATMAPS
    cmap_choice = "viridis" if metric == "R2" else "YlGnBu"

    sns.heatmap(
        perf_heatmap,
        annot=True,
        fmt=".3f",
        cmap=cmap_choice
    )

    plt.title(f"Complete Model Comparison - {metric}")
    plt.xlabel("Reactor Group")
    plt.ylabel("Model")

    plt.tight_layout()

    plt.savefig(
            COMPLETE_FOLDER /
            f"Complete_{metric}.png",
        dpi=300
    )

    plt.close()

    print(f"✅ Saved complete {metric} heatmap")

# ===============================================================
# PER-OUTPUT HEATMAPS
# ===============================================================

for target in outputs:

    out_dir = COMPLETE_FOLDER / target
    out_dir.mkdir(exist_ok=True)

    perf_sub = performance_all[
        performance_all["Target"] == target
    ]

    for metric in metrics:

        perf_heatmap = perf_sub.pivot_table(
            index="Model",
            columns="Reactor",
            values=metric,
            aggfunc="mean"
        )

        perf_heatmap = perf_heatmap.reindex(
            index=model_order,
            columns=reactor_order
        )

        plt.figure(figsize=(12,5))

        cmap_choice = "viridis" if metric == "R2" else "YlGnBu"

        sns.heatmap(
            perf_heatmap,
            annot=True,
            fmt=".3f",
            cmap=cmap_choice
        )

        plt.title(f"{target} - {metric}")
        plt.xlabel("Reactor Group")
        plt.ylabel("Model")

        plt.tight_layout()

        plt.savefig(
                out_dir /
                f"{target}_{metric}.png",
            dpi=300
        )

        plt.close()

    print(f"✅ Completed {target}")

print("\n✅ COMPLETE HEATMAP EXPORT FINISHED")

# ===============================================================
# 🔍 Sobol Sensitivity Analysis Using Physics-Informed PINNs
# ===============================================================

# ---------------------------------------------------------------
# 0️⃣ Load Data
# ---------------------------------------------------------------
DATA_PATH = "Nuclear Data.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8", na_values=["NA","NaN",""," "])
df.columns = [c.strip() for c in df.columns]

y_cols = ["Outlet_Temperature","Thermal_Power","Gross_Electrical_Power"]
X_cols = [
    "Fuel_Enrichment",
    "Core_Diameter",
    "Core_Height",
    "Number_of_Fuel_Assemblies",
    "Fuel_Linear_Heat_Generation_Rate",
    "Control_Rod_Assemblies",
    "Coolant_Pressure"
]

for c in X_cols + y_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Reactor grouping
def assign_reactor_group(row):
    rt = row["Reactor_Type"]
    p = row["Thermal_Power"]
    if pd.isna(rt) or pd.isna(p): return "Unknown"
    if rt == "PWR":
        if p < 2785: return "PWR_Low"
        elif p <= 3020: return "PWR_Mid"
        else: return "PWR_High"
    elif rt == "BWR": return "BWR_Mid" if p < 3200 else "BWR_High"
    elif rt == "PHWR":
        if p < 1000: return "PHWR_Small"
        elif p <= 2200: return "PHWR_Standard"
        else: return "PHWR_Large"
    else: return rt

df["Reactor_Group"] = df.apply(assign_reactor_group, axis=1)
counts = df["Reactor_Group"].value_counts()
valid_groups = counts[counts >= 15].index
df = df[df["Reactor_Group"].isin(valid_groups)].reset_index(drop=True)

# ---------------------------------------------------------------
# Define PINN
# ---------------------------------------------------------------
class ReactorPINN(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,64), nn.Tanh(),
            nn.Linear(64,output_size)
        )
    def forward(self,x):
        return self.net(x)

def saturation_temperature(P_mpa):
    P_mpa = np.clip(P_mpa, 0.01, 22.064)

    water = IAPWS97(P=P_mpa, x=0)
    return water.T - 273.15

def physics_loss(model, X, y_pred, lambdas):

    idx = {col:i for i,col in enumerate(X_cols)}

    T_out = y_pred[:,0]
    P_th  = y_pred[:,1]
    P_el  = y_pred[:,2]

    enrichment = X[:,idx["Fuel_Enrichment"]]
    diameter   = X[:,idx["Core_Diameter"]]
    height     = X[:,idx["Core_Height"]]
    assemblies = X[:,idx["Number_of_Fuel_Assemblies"]]
    lhgr       = X[:,idx["Fuel_Linear_Heat_Generation_Rate"]]
    rods       = X[:,idx["Control_Rod_Assemblies"]]
    pressure   = X[:,idx["Coolant_Pressure"]]
    pressure = torch.clamp(pressure, 0.01, 22.064)

    # 1️⃣ Efficiency constraint
    eta = 0.33
    L_eff = torch.mean((P_el - eta * P_th)**2)

    # 2️⃣ Saturation margin constraint
    T_sat = torch.tensor(
    [saturation_temperature(p.item()) for p in pressure],
    dtype=torch.float32,
    device=pressure.device)
    margin = 10.0
    violation = torch.relu(T_out - (T_sat + margin))
    L_sat = torch.mean(violation**2)


    # 3️⃣ Monotonicity via gradients
    grads_T = torch.autograd.grad(T_out.sum(), X, create_graph=True)[0]
    grads_Pth = torch.autograd.grad(P_th.sum(), X, create_graph=True)[0]
    grads_Pel = torch.autograd.grad(P_el.sum(), X, create_graph=True)[0]

    L_mono = 0

    # Thermal Power constraints
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Fuel_Linear_Heat_Generation_Rate"]]))
    L_mono += torch.mean(torch.relu(grads_Pth[:, idx["Control_Rod_Assemblies"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Number_of_Fuel_Assemblies"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Core_Diameter"]]))
    L_mono += torch.mean(torch.relu(-grads_Pth[:, idx["Core_Height"]]))

    # Outlet Temperature constraints
    L_mono += torch.mean(torch.relu(-grads_T[:, idx["Fuel_Linear_Heat_Generation_Rate"]]))

    total = (
        lambdas["eff"]*L_eff +
        lambdas["sat"]*L_sat +
        lambdas["mono"]*L_mono
    )
    return total
    

# ---------------------------------------------------------------
# Evaluate function
# ---------------------------------------------------------------
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred)/denom))*100
    return {"R2":r2,"RMSE":rmse,"MAE":mae,"MAPE(%)":mape}

# ---------------------------------------------------------------
# Sobol problem
# ---------------------------------------------------------------
problem = {
    "num_vars": len(X_cols),
    "names": X_cols,
    "bounds": [[df[col].min(), df[col].max()] for col in X_cols]
}

N_SAMPLES = 2000
param_values = saltelli.sample(problem, N_SAMPLES, calc_second_order=False)

SAVE_DIR = FIGURES_DIR / "Sobol_PINN_Plots"
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Loop per Reactor Group + PINN + Sobol
# ---------------------------------------------------------------
sobol_tables = {}

for group, df_sub in df.groupby("Reactor_Group"):
    print(f"\n===== ⚙️ Reactor Group: {group} =====")
    X = df_sub[X_cols]
    y = df_sub[y_cols]

    # Preprocess
    X_imp = SimpleImputer(strategy="mean").fit_transform(X)
    y_imp = SimpleImputer(strategy="mean").fit_transform(y)
    scaler_X = StandardScaler().fit(X_imp)
    scaler_y = StandardScaler().fit(y_imp)

    X_scaled = scaler_X.transform(X_imp)
    y_scaled = scaler_y.transform(y_imp)

    X_t = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)

    # Train PINN
    model = ReactorPINN(input_size=len(X_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambdas = {"eff":0.5,"sat":0.5,"mono":0.5}
    epochs = 300

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        data_loss = nn.MSELoss()(y_pred, y_t)
        phys_loss = physics_loss(model, X_t, y_pred, lambdas)
        loss = data_loss + phys_loss
        loss.backward()
        optimizer.step()

    model.eval()
    # Predict Sobol samples
    X_sample_scaled = scaler_X.transform(param_values)
    X_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_sample_scaled = model(X_tensor).numpy()
    y_pred_sample = scaler_y.inverse_transform(y_pred_sample_scaled)

    # Compute Sobol indices per target
    sobol_tables[group] = {}
    for idx, target in enumerate(y_cols):
        Y_target = y_pred_sample[:, idx]
        Si = sobol.analyze(problem, Y_target, calc_second_order=False, print_to_console=False)
        df_sobol = pd.DataFrame({
            "Parameter": X_cols,
            "S1": Si["S1"],
            "ST": Si["ST"]
        }).sort_values("ST", ascending=False)
        sobol_tables[group][target] = df_sobol
        print(f"📊 {group} / {target} — Top 3 ST: {', '.join(df_sobol['Parameter'].iloc[:3])}")

# ---------------------------------------------------------------
# Generate heatmaps per target
# ---------------------------------------------------------------
base_output_folder = HEATMAP_DIR / "PINN_Sobol_Heatmaps"
base_output_folder.mkdir(exist_ok=True)

param_colors = {
    "Fuel_Enrichment": "#1f77b4",
    "Core_Diameter": "#ff7f0e",
    "Core_Height": "#2ca02c",
    "Number_of_Fuel_Assemblies": "#d62728",
    "Fuel_Linear_Heat_Generation_Rate": "#9467bd",
    "Control_Rod_Assemblies": "#8c564b",
    "Coolant_Pressure": "#e377c2"
}

targets = y_cols
sobol_combined = []
for group, reactors_dict in sobol_tables.items():
    for target, df_sobol in reactors_dict.items():
        for idx_type in ["S1","ST"]:
            df_tmp = df_sobol.copy()
            df_tmp["Reactor"] = group
            df_tmp["Target"] = target
            df_tmp["Index_Type"] = idx_type
            df_tmp.rename(columns={idx_type: "Value"}, inplace=True)
            sobol_combined.append(df_tmp[["Parameter","Reactor","Target","Index_Type","Value"]])

sobol_df = pd.concat(sobol_combined, ignore_index=True)
print("✅ Sobol DataFrame ready:", sobol_df.shape)

for target in targets:
    print(f"\n📊 Generating Sobol heatmaps for: {target}")
    df_target = sobol_df[sobol_df["Target"]==target]
    output_folder = base_output_folder / "target"
    output_folder.mkdir(exist_ok=True)

    for idx_type in ["S1","ST"]:
        df_idx = df_target[df_target["Index_Type"]==idx_type]
        heatmap_data = df_idx.pivot_table(
            index="Parameter",
            columns="Reactor",
            values="Value",
            aggfunc="mean"
        )
        param_order = heatmap_data.abs().mean(axis=1).sort_values(ascending=False).index
        heatmap_data = heatmap_data.loc[param_order]

        plt.figure(figsize=(12,6))
        cmap_choice = "YlGnBu" if idx_type=="ST" else "YlOrRd"
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap_choice)
        plt.title(f"{idx_type} Sobol Sensitivity — {target}", fontsize=14)
        plt.xlabel("Reactor Group")
        plt.ylabel("Input Parameter")
        plt.tight_layout()
        save_path = output_folder / f"{idx_type}_Sobol_{target}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ {idx_type} heatmap saved for {target} at {save_path}")

print("\n🎯 All per-target Sobol heatmaps successfully generated.")

# ---------------------------------------------------------------
# Input Variance Analysis (Raw + Per Reactor Group)
# ---------------------------------------------------------------
variance_tables = []

for group, df_sub in df.groupby("Reactor_Group"):
    X = df_sub[X_cols].copy()

    # ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # raw variance (no scaling, no imputation bias)
    var_values = X.var(skipna=True)

    df_var = pd.DataFrame({
        "Parameter": X_cols,
        "Variance": var_values.values,
        "Reactor_Group": group
    })

    variance_tables.append(df_var)

variance_df = pd.concat(variance_tables, ignore_index=True)

# save
VAR_PATH = RESULTS_DIR / "variance_by_group.csv"
VAR_PATH.mkdir(exist_ok=True)

variance_df.to_csv(VAR_PATH, index=False)

print("✅ Variance table saved at:", VAR_PATH)

# ---------------------------------------------------------------
# Variance Heatmap (Reactor Group vs Input Parameters)
# ---------------------------------------------------------------
pivot_var = variance_df.pivot_table(
    index="Parameter",
    columns="Reactor_Group",
    values="Variance"
)

# sort parameters by overall variance importance
param_order = pivot_var.mean(axis=1).sort_values(ascending=False).index
pivot_var = pivot_var.loc[param_order]

plt.figure(figsize=(12,6))
sns.heatmap(pivot_var, annot=True, fmt=".3e", cmap="viridis")

plt.title("Input Feature Variance Across Reactor Groups", fontsize=14)
plt.xlabel("Reactor Group")
plt.ylabel("Input Parameter")
plt.tight_layout()

HEATMAP_PATH = HEATMAP_DIR / "variance_heatmap.png"
plt.savefig(HEATMAP_PATH, dpi=300)
plt.close()

print("✅ Variance heatmap saved at:", HEATMAP_PATH)

# ---------------------------------------------------------------
# =Create Combined ST CSV
# Each cell format:
# Outlet_Temperature, Thermal_Power, Gross_Electrical_Power
# ---------------------------------------------------------------

combined_rows = []

# Desired reactor ordering
reactor_groups = [
    "PHWR_Standard",
    "PWR_Low",
    "PWR_Mid",
    "PWR_High",
    "BWR_Mid",
    "BWR_High"
]

# Keep only groups that actually exist in the dataframe
reactor_groups = [r for r in reactor_groups if r in sobol_df["Reactor"].unique()]
parameters = X_cols

for param in parameters:

    row_data = {"Parameter": param}

    for reactor in reactor_groups:

        values = []

        for target in y_cols:  # preserves requested order

            df_match = sobol_df[
                (sobol_df["Parameter"] == param) &
                (sobol_df["Reactor"] == reactor) &
                (sobol_df["Target"] == target) &
                (sobol_df["Index_Type"] == "ST")
            ]

            if not df_match.empty:
                val = df_match["Value"].values[0]
                values.append(f"{val:.3f}")
            else:
                values.append("NA")

        # Single CSV cell:
        # Outlet_Temperature, Thermal_Power, Gross_Electrical_Power
        row_data[reactor] = ", ".join(values)

    combined_rows.append(row_data)

combined_df = pd.DataFrame(combined_rows)

# ---------------------------------------------------------------
# Average Sobol Heatmaps Across All Outputs
# ---------------------------------------------------------------

print("\n📊 Generating AVERAGE Sobol Heatmaps Across Outputs")

avg_output_folder = HEATMAP_DIR / "Average_Sobol_All_Outputs.png"
avg_output_folder.mkdir(exist_ok=True)

for idx_type in ["S1", "ST"]:

    # ===========================================================
    # Average across ALL targets
    # ===========================================================

    avg_heatmap_data = df_idx.groupby(
        ["Parameter", "Reactor"]
    )["Value"].mean().reset_index()

    avg_heatmap_data = avg_heatmap_data.pivot_table(
        index="Parameter",
        columns="Reactor",
        values="Value",
        aggfunc="mean"
    )

    # ===========================================================
    # SAME PARAMETER ORDERING LOGIC
    # ===========================================================

    param_order = (
        avg_heatmap_data
        .abs()
        .mean(axis=1)
        .sort_values(ascending=False)
        .index
    )

    avg_heatmap_data = avg_heatmap_data.loc[param_order]

    # ===========================================================
    # HEATMAP
    # ===========================================================

    plt.figure(figsize=(12,6))

    cmap_choice = "YlGnBu" if idx_type == "ST" else "YlOrRd"

    sns.heatmap(
        avg_heatmap_data,
        annot=True,
        fmt=".3f",
        cmap=cmap_choice
    )

    plt.title(
        f"Average {idx_type} Sobol Sensitivity Across All Outputs",
        fontsize=14
    )

    plt.xlabel("Reactor Group")
    plt.ylabel("Input Parameter")

    plt.tight_layout()

    save_path = avg_output_folder / f"Average_{idx_type}_Sobol_All_Outputs.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Average {idx_type} heatmap saved at {save_path}")

print("\n🎯 Average Sobol heatmaps successfully generated.")

# ===========================================================
# SAVE AVERAGED DATA AS CSV
# ===========================================================

csv_save_path = avg_output_folder / f"Average_{idx_type}_Sobol_All_Outputs.csv"

avg_heatmap_data.to_csv(csv_save_path)

print(f"📄 CSV saved at {csv_save_path}")

# Filter desired index type
df_idx = sobol_df[sobol_df["Index_Type"] == idx_type]