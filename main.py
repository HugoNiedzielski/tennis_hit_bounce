import json
import joblib
import os
import pandas as pd
from tqdm import tqdm

# Import local utility modules
from src.utils_physics import preprocess_trajectory, compute_physics_features, detect_events_heuristics
from src.utils_ml import compute_features, prepare_inference_window, apply_nms

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "input_data/per_point_v2"      # Folder containing input JSONs
OUTPUT_FOLDER = "output_results"        # Folder for results
MODEL_PATH = "tennis_xgb_supervised.pkl" # Pre-trained model

# ==========================================
# METHOD 1: UNSUPERVISED
# ==========================================
def unsupervised_hit_bounce_detection(ball_data):
    """
    Method 1: Unsupervised Hit & Bounce Detection.
    Relies on physics-based heuristics (acceleration peaks, velocity changes)
    to identify events from trajectory data without training labels.
    
    Args:
        ball_data (dict): Dictionary of frame-indexed ball positions.
    Returns:
        dict: The input dictionary enriched with a 'pred_action' key.
    """
    # 1. Preprocessing & Physics Computation
    df = preprocess_trajectory(ball_data)
    df = compute_physics_features(df, fps=50)
    
    # 2. Event Detection (Heuristics)
    detected_events = detect_events_heuristics(df)
    
    # 3. Output Formatting
    output_data = ball_data.copy()
    
    for frame_str in output_data:
        try:
            frame_id = int(frame_str)
            if frame_id in detected_events:
                output_data[frame_str]['pred_action'] = detected_events[frame_id]
            else:
                output_data[frame_str]['pred_action'] = 'air'
        except ValueError:
            continue
            
    return output_data

# ==========================================
# METHOD 2: SUPERVISED
# ==========================================
def supervised_hit_bounce_detection(ball_data, model_path=MODEL_PATH):
    """
    Method 2: Supervised Hit & Bounce Detection.
    Uses a pre-trained XGBoost classifier to identify events based on
    learned kinematic patterns.
    
    Args:
        ball_data (dict): Dictionary of frame-indexed ball positions.
        model_path (str): Path to the .pkl model file.
    Returns:
        dict: The input dictionary enriched with a 'pred_action' key.
    """
    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please provide a valid model.")
    
    model = joblib.load(model_path)
    
    # 2. Data Preparation
    df = pd.DataFrame.from_dict(ball_data, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()
    
    # 3. Feature Engineering (Must match training pipeline)
    df = compute_features(df, fps=50)
    X = prepare_inference_window(df, window_size=5)
    
    # 4. Inference
    probs = model.predict_proba(X)
    
    # 5. Post-Processing (NMS)
    # Thresholds are calibrated on validation set for F0.5 Score optimization
    preds = apply_nms(probs, threshold_hit=0.60, threshold_bounce=0.60, refractory=10)
    
    # 6. Output Formatting
    output_data = ball_data.copy()
    frame_ids = df.index.values
    
    # Initialize defaults
    for k in output_data:
        output_data[k]['pred_action'] = 'air'
        
    # Map predictions (0=Air, 1=Bounce, 2=Hit)
    for i, p in enumerate(preds):
        if p == 0: continue
        
        f_id = str(frame_ids[i])
        if f_id in output_data:
            if p == 1:
                output_data[f_id]['pred_action'] = 'bounce'
            elif p == 2:
                output_data[f_id]['pred_action'] = 'hit'
                
    return output_data

# ==========================================
# MAIN EXECUTION (BATCH PROCESSING)
# ==========================================
if __name__ == "__main__":
    
    # Check if input data exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder '{INPUT_FOLDER}' not found.")
        print("Please configure INPUT_FOLDER variable in main.py")
        exit()

    # Create output directories
    unsup_out_dir = os.path.join(OUTPUT_FOLDER, "unsupervised")
    sup_out_dir = os.path.join(OUTPUT_FOLDER, "supervised")
    os.makedirs(unsup_out_dir, exist_ok=True)
    os.makedirs(sup_out_dir, exist_ok=True)
    
    # Get list of JSON files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    print(f"Starting batch processing for {len(files)} files...")
    print(f"Inputs: {INPUT_FOLDER}")
    print(f"Outputs: {OUTPUT_FOLDER}")

    # Process files with progress bar
    for filename in tqdm(files, desc="Processing"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # --- 1. Run Unsupervised Method ---
            res_unsup = unsupervised_hit_bounce_detection(data.copy())
            with open(os.path.join(unsup_out_dir, filename), 'w') as f:
                json.dump(res_unsup, f, indent=4)
                
            # --- 2. Run Supervised Method ---
            if os.path.exists(MODEL_PATH):
                res_sup = supervised_hit_bounce_detection(data.copy(), model_path=MODEL_PATH)
                with open(os.path.join(sup_out_dir, filename), 'w') as f:
                    json.dump(res_sup, f, indent=4)
            else:
                # Warning printed only once at start usually, but safe here
                pass 
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nProcessing complete. All results saved.")