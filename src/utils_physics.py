import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

def preprocess_trajectory(point_data):
    """
    Standardizes trajectory data by sorting frames and interpolating missing values.
    """
    df = pd.DataFrame.from_dict(point_data, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()
    
    # Fill missing intermediate frames
    if not df.empty:
        full_idx = np.arange(df.index.min(), df.index.max() + 1)
        df = df.reindex(full_idx)
        
    df['x'] = df['x'].astype(float).interpolate(method='linear')
    df['y'] = df['y'].astype(float).interpolate(method='linear')
    return df

def compute_physics_features(df, fps=50):
    """
    Computes smoothed velocity and acceleration vectors using Savitzky-Golay filtering.
    """
    dt = 1/fps
    # Smoothing parameters: window_length=7, polyorder=2
    try:
        df['x_smooth'] = savgol_filter(df['x'], 7, 2)
        df['y_smooth'] = savgol_filter(df['y'], 7, 2)
    except ValueError:
        # Fallback for sequences shorter than the window length
        df['x_smooth'] = df['x']
        df['y_smooth'] = df['y']
        
    # Numerical differentiation
    df['vx'] = np.gradient(df['x_smooth'], dt)
    df['vy'] = np.gradient(df['y_smooth'], dt)
    df['ax'] = np.gradient(df['vx'], dt)
    df['ay'] = np.gradient(df['vy'], dt)
    return df

def detect_events_heuristics(df):
    """
    Detects events based on physical rules:
    - Bounce: Concave vertical acceleration peaks.
    - Hit: Significant horizontal velocity changes or high energy spikes.
    """
    # Detection Thresholds
    PEAK_HEIGHT = 2500      
    CONVEX_THRESH = -1200   
    ENERGY_THRESH = 200     
    VIOLENCE_THRESH = 25000 
    REFRACTORY_PERIOD = 10  
    
    # Peak detection on vertical acceleration
    df['ay_abs'] = df['ay'].abs()
    peaks, _ = find_peaks(df['ay_abs'], height=PEAK_HEIGHT, prominence=1000, distance=5)
    
    candidates = []

    for i in peaks:
        frame_id = df.index[i]
        
        # Check temporal context (+/- 5 frames)
        offset = 5
        if i - offset < 0 or i + offset >= len(df): continue
            
        vx_pre = df['vx'].iloc[i - offset]
        vx_post = df['vx'].iloc[i + offset]
        
        # Metrics for classification
        vx_prod = vx_pre * vx_post # Negative implies direction change
        delta_vx_mag = abs(vx_post) - abs(vx_pre)
        max_vx = max(abs(vx_pre), abs(vx_post))
        ay_val_raw = df['ay'].iloc[i]
        
        is_convex_bounce = (ay_val_raw < CONVEX_THRESH)
        is_moving = (max_vx > 150)
        
        pred = None
        priority = 0
        
        # Classification Logic
        if ((vx_prod < 0) and is_moving) or \
           (delta_vx_mag > ENERGY_THRESH) or \
           (abs(ay_val_raw) > VIOLENCE_THRESH):
            pred = 'hit'
            priority = 2
            
        elif (vx_prod > 0) and is_convex_bounce:
            pred = 'bounce'
            priority = 1
            
        if pred:
            # Store absolute acceleration as confidence proxy
            candidates.append((frame_id, pred, abs(ay_val_raw), priority))
            
    # Temporal filtering (Non-Maximum Suppression)
    final_events = {}
    candidates.sort(key=lambda x: x[0])
    
    if candidates:
        curr = candidates[0]
        for next_evt in candidates[1:]:
            # If events are too close, keep the one with higher priority or magnitude
            if next_evt[0] - curr[0] < REFRACTORY_PERIOD:
                if next_evt[3] > curr[3]: 
                    curr = next_evt
                elif next_evt[3] == curr[3] and next_evt[2] > curr[2]:
                    curr = next_evt
            else:
                final_events[curr[0]] = curr[1]
                curr = next_evt
        final_events[curr[0]] = curr[1]
        
    return final_events