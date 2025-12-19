import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import os
from pathlib import Path
from tqdm import tqdm
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, fbeta_score
from sklearn.utils.class_weight import compute_sample_weight

# --- PATH SETUP ---
# Ensures the script can run from anywhere and find 'src'
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.utils_ml import compute_features, prepare_inference_window, apply_nms
except ImportError:
    print("Error: Cannot import src.utils_ml.")
    print(f"Check that the src folder contains utils_ml.py in: {PROJECT_ROOT}")
    sys.exit(1)

# --- CONFIGURATION ---
# Matches the folder structure seen in main.py
INPUT_FOLDER = PROJECT_ROOT / "input_data" / "per_point_v2"
OUTPUT_MODEL_PATH = PROJECT_ROOT / "tennis_xgb_supervised.pkl"
FPS = 50
WINDOW_SIZE = 5

# --- DATA LOADING ---

def load_and_aggregate_data(input_dir):
    """
    Loads all JSON files from the input directory and combines them 
    into a single dictionary suitable for training.
    """
    aggregated_data = {}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")
        
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_path}")
    
    print(f"Aggregating {len(json_files)} files from {input_path.name}:")
    
    for file_path in tqdm(json_files, desc="Loading Data"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Use filename stem (e.g., 'ball_data_12') as key
                aggregated_data[file_path.stem] = data
        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")
            continue
            
    return aggregated_data

# --- ML PIPELINE FUNCTIONS ---

def create_dataset(full_data):
    X, y, groups = [], [], []
    label_map = {'air': 0, 'bounce': 1, 'hit': 2}
    
    print(f"Building features for {len(full_data)} sequences:")
    
    for seq_id, point_data in full_data.items():
        df = pd.DataFrame.from_dict(point_data, orient='index')
        df.index = df.index.astype(int)
        df = df.sort_index()
        
        # Ensure action column exists
        if 'action' not in df.columns:
            df['action'] = 'air'
        
        # Map text labels to integers
        df['label'] = df['action'].fillna('air').map(label_map)
        
        # Feature Engineering (from src.utils_ml)
        df = compute_features(df, fps=FPS)
        
        # Windowing (from src.utils_ml)
        X_seq = prepare_inference_window(df, window_size=WINDOW_SIZE)
        y_seq = df['label'].values
        
        # Quality check
        if len(X_seq) == len(y_seq) and len(X_seq) > 0:
            X.append(X_seq)
            y.append(y_seq)
            groups.append([seq_id] * len(y_seq))
            
    if not X:
        raise ValueError("No valid data extracted. Check input JSON format.")
        
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)

def split_data_3way(X, y, groups):
    # 1. Test Set (20%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    rest_idx, test_idx = next(gss_test.split(X, y, groups))
    
    X_rest, y_rest, g_rest = X[rest_idx], y[rest_idx], groups[rest_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 2. Train / Val (80% / 20% of remainder)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_val.split(X_rest, y_rest, g_rest))
    
    X_train, y_train = X_rest[train_idx], y_rest[train_idx]
    X_val, y_val = X_rest[val_idx], y_rest[val_idx]
    
    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_with_tolerance(y_true, y_pred, tolerance=2):
    labels = [1, 2]
    names = {1: "Bounce", 2: "Hit"}
    
    print(f"\nResults with tolerance (+/- {tolerance} frames)")
    print("-" * 60)
    print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    
    for label in labels:
        true_indices = np.where(y_true == label)[0]
        pred_indices = np.where(y_pred == label)[0]
        
        # Recall
        tp_recall = 0
        for t_idx in true_indices:
            if np.any(np.abs(pred_indices - t_idx) <= tolerance):
                tp_recall += 1
        recall = tp_recall / len(true_indices) if len(true_indices) > 0 else 0
        
        # Precision
        tp_precision = 0
        for p_idx in pred_indices:
            if np.any(np.abs(true_indices - p_idx) <= tolerance):
                tp_precision += 1
        precision = tp_precision / len(pred_indices) if len(pred_indices) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{names[label]:<10} | {precision:.3f}      | {recall:.3f}      | {f1:.3f}")

def run_optuna(X_train, y_train, X_val, y_val, n_trials=1):
    # Class weights with aggressive boost for rare events
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    sample_weights[y_train > 0] *= 4.0 
    
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train, 
            sample_weight=sample_weights, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        # Optimize for F0.5 score (Precision over Recall preference)
        return fbeta_score(y_val, y_pred, beta=0.5, average='macro')

    print(f"Starting Optuna optimization ({n_trials} trials):")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def main():
    # 1. Load and Aggregate Data
    try:
        full_data = load_and_aggregate_data(INPUT_FOLDER)
    except FileNotFoundError as e:
        print(f"Critical Error: {e}")
        return

    if not full_data:
        print("Error: Dataset is empty.")
        return
        
    # 2. Create Dataset Arrays
    X, y, groups = create_dataset(full_data)
    
    # 3. Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(X, y, groups)
    
    # 4. Optuna Optimization
    best_params = run_optuna(X_train, y_train, X_val, y_val, n_trials=30)
    print(f"Best Params: {best_params}")
    
    # 5. Final Training
    print("\nTraining Final Model (Train + Val):")
    X_final = np.vstack((X_train, X_val))
    y_final = np.concatenate((y_train, y_val))
    
    final_weights = compute_sample_weight(class_weight='balanced', y=y_final)
    final_weights[y_final > 0] *= 4.0
    
    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        **best_params
    )
    final_model.fit(X_final, y_final, sample_weight=final_weights)
    
    # 6. Evaluation
    print("Evaluating on Test Set:")
    probs = final_model.predict_proba(X_test)
    preds = apply_nms(probs, threshold_hit=0.60, threshold_bounce=0.60, refractory=10)
    
    print(classification_report(y_test, preds, target_names=['air', 'bounce', 'hit']))
    evaluate_with_tolerance(y_test, preds, tolerance=2)
    
    # 7. Save
    joblib.dump(final_model, OUTPUT_MODEL_PATH)
    print(f"Model saved to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()