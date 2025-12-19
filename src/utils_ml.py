import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def compute_features(df, fps=50):
    """
    Computes kinematic features for the ML model.
    """
    dt = 1/fps
    # Robust interpolation
    df['x'] = df['x'].astype(float).interpolate(method='linear').ffill().bfill()
    df['y'] = df['y'].astype(float).interpolate(method='linear').ffill().bfill()
    
    # Smoothing
    try:
        df['x_smooth'] = savgol_filter(df['x'], 7, 2)
        df['y_smooth'] = savgol_filter(df['y'], 7, 2)
    except ValueError:
        df['x_smooth'] = df['x']
        df['y_smooth'] = df['y']
    
    # Gradients
    df['vx'] = np.gradient(df['x_smooth'], dt)
    df['vy'] = np.gradient(df['y_smooth'], dt)
    df['ax'] = np.gradient(df['vx'], dt)
    df['ay'] = np.gradient(df['vy'], dt)
    
    # Magnitude features
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['acc_norm'] = np.sqrt(df['ax']**2 + df['ay']**2)
    
    # Jerk (derivative of acceleration)
    df['jerk_x'] = np.gradient(df['ax'], dt)
    df['jerk_y'] = np.gradient(df['ay'], dt)
    df['jerk_norm'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2)
    
    # Geometric features
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    df['angle_change'] = np.abs(np.gradient(df['angle']))
    
    # Log transformations
    df['log_jerk'] = np.log1p(df['jerk_norm'])
    df['log_acc'] = np.log1p(df['acc_norm'])
    
    return df

def prepare_inference_window(df, window_size=5):
    """
    Formats the data into a windowed matrix (t-N to t+N) for inference.
    """
    features = ['vx', 'vy', 'ax', 'ay', 'speed', 'acc_norm', 'jerk_norm', 
                'angle', 'angle_change', 'log_jerk', 'log_acc']
    
    cols = []
    for f in features:
        if f in df.columns:
            cols.append(df[f].rename(f"{f}_t"))
            for i in range(1, window_size + 1):
                cols.append(df[f].shift(i).rename(f"{f}_t-{i}"))
                cols.append(df[f].shift(-i).rename(f"{f}_t+{i}"))
    
    X_df = pd.concat(cols, axis=1)
    # Fill boundaries with 0
    X = X_df.fillna(0).values
    return X

def apply_nms(probs, threshold_hit=0.60, threshold_bounce=0.60, refractory=10):
    """
    Applies Non-Maximum Suppression to filter probability outputs.
    Ensures a single event detection per time window.
    """
    n_samples = len(probs)
    cleaned_preds = np.zeros(n_samples, dtype=int)
    candidates = []
    
    # Select candidates based on thresholds
    for idx in np.where(probs[:, 1] > threshold_bounce)[0]:
        candidates.append((idx, 1, probs[idx, 1])) # 1 = Bounce
    for idx in np.where(probs[:, 2] > threshold_hit)[0]:
        candidates.append((idx, 2, probs[idx, 2])) # 2 = Hit
        
    # Sort by confidence
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    final_indices = set()
    for idx, label, conf in candidates:
        is_duplicate = False
        for valid_idx in final_indices:
            if abs(valid_idx - idx) < refractory:
                is_duplicate = True
                break
        
        if not is_duplicate:
            cleaned_preds[idx] = label
            final_indices.add(idx)
            
    return cleaned_preds