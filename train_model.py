import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from src.utils_ml import compute_features, prepare_inference_window

def train_new_model(dataset_path, output_model_path):
    print("Loading dataset...")
    with open(dataset_path, 'r') as f:
        full_data = json.load(f)
        
    print("Preparing features...")
    X, y, groups = [], [], []
    label_map = {'air': 0, 'bounce': 1, 'hit': 2}
    
    for seq_id, point_data in full_data.items():
        df = pd.DataFrame.from_dict(point_data, orient='index')
        df.index = df.index.astype(int)
        df = df.sort_index()
        
        if 'action' not in df.columns: df['action'] = 'air'
        df['label'] = df['action'].fillna('air').map(label_map)
        
        df = compute_features(df)
        X_seq = prepare_inference_window(df)
        y_seq = df['label'].values
        
        X.append(X_seq)
        y.append(y_seq)
        groups.append([seq_id] * len(y_seq))
        
    X = np.vstack(X)
    y = np.concatenate(y)
    
    print("Training XGBoost model...")
    weights = compute_sample_weight(class_weight='balanced', y=y)
    weights[y > 0] *= 4.0 # Boost weight for events
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500
    )
    model.fit(X, y, sample_weight=weights)
    
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    train_new_model('full_dataset.json', 'tennis_xgb_supervised.pkl')