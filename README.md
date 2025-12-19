# Roland-Garros 2025: Tennis Hit & Bounce Detection 🎾

## Project Overview

This repository contains a solution for the **Sport Scientist Computer Vision & ML Exercise**.The goal is to detect **Hit** (racket impact) and **Bounce** (court impact) events using 2D ball-tracking data from the Roland-Garros 2025 Final.

In a sports analytics context, reliable event detection is critical for reconstructing 3D trajectories and automating game statistics.This project tackles the inherent noise in 2D tracking data using two complementary approaches: 

1.**Unsupervised Method (Physics-Based)**: A heuristics-driven approach focusing on kinematic discontinuities.
2.**Supervised Method (Machine Learning)**: A high-precision classifier (XGBoost) designed to filter out tracking artifacts and "ghost" events.

---

## The Challenge & Strategy

Raw ball-tracking data is often noisy/jittery.A simple physics model can detect most impacts (High Recall) but often mistakes tracking noise for events (Low Precision).

**My Strategy:**
- **Physics First:** Use kinematics to understand the signal and establish a strong baseline.
- **ML for Precision:** Use XGBoost with a **F0.5-Score optimization**.This metric weighs Precision higher than Recall.
    - *Why?* In 3D reconstruction, missing one bounce (False Negative) can be interpolated, but inventing a bounce (False Positive) breaks the entire physics engine.

---

## Repository Structure

```text
tennis_project/
│
├── input_data/                 # Input folder (JSONs and MP4 here)
│   └── (ball_data_*.json)
│
├── output_results/             # Outputs (Enriched JSONs generated here)
│   ├── unsupervised/
│   └── supervised/
│
├── src/                        # Core Logic
│   ├── utils_physics.py        # Signal processing & Heuristics
│   └── utils_ml.py             # Feature Engineering & NMS
│
├── notebooks/                  # Research & Visualization
│   └── visualizations.ipynb  # EDA, Threshold tuning, and Video Overlay
│
├── main.py                     # Entry point (Implements requested functions)
├── requirements.txt            # Dependencies
├── tennis_xgb_supervised.pkl   # Trained XGBoost Model
└── README.md                   # Documentation
```

---

## Methods Description

### 1.Unsupervised Method (Physics-Based)

**Implemented in `main.py` as `unsupervised_hit_bounce_detection`**

Relies on signal processing (Savitzky-Golay smoothing) to calculate derivatives.

- **Bounces**: Detected via Vertical Acceleration (A_y) peaks (> 2500 px/s²) and trajectory convexity.
- **Hits**: Detected via sudden Horizontal Velocity (V_x) reversals or energy spikes.

### 2.Supervised Method (XGBoost)

**Implemented in `main.py` as `supervised_hit_bounce_detection`**

A Gradient Boosting model trained to distinguish true events from noise.

- **Feature Engineering**: 33 features including Jerk (derivative of acceleration), Angles, and Windowed Lag/Lead features (t-5 to t+5).
- **Post-Processing**: Non-Maximum Suppression (NMS) ensures only the highest confidence event is selected within a 200ms refractory period (to ensure no duplicates).

---

## Example Performance Benchmark for the bounces

Evaluated on a hidden test set for the XGBoost and on the whole dataset for the physics-based.(Time tolerance: ± 2 frames).

| Method         | Bounce Precision | Bounce Recall | F1-Score | Analysis                                                                                                       |
|----------------|------------------|---------------|----------|----------------------------------------------------------------------------------------------------------------|
| Unsupervised   | 61.6%            | 87.7%         | 72.4%    | **High Sensitivity**: Excellent at finding physical impacts, but generates too many false positives due to noise.|
| Supervised     | 85.1%            | 90.6%         | 87.7%    | **High Fidelity**:  The ML model successfully learned to ignore tracking noise, increasing precision by +24% while maintaining Recall.|

**Conclusion**: The Supervised method is the recommended pipeline for production, offering a stable and clean event detection.

---

## Quick Start

### 1.Installation

```bash
git clone <repository_url>
cd tennis_project
pip install -r requirements.txt
```

### 2.Data Setup

- Create a folder named `input_data` at the root.
- Copy all JSON files (`ball_data_*.json`) into `input_data/`.
- (Optional) Copy the video `Alcaraz_Sinner_2025.mp4` into `input_data/` for visualization.

### 3.Execution

Run the batch processor. This script will automatically create the output folders and process all files.

```bash
python main.py
```

If one wanted to re-train the model, he would be able to do so using the following command : 

```bash
python train_model.py
```

It would generate a new `tennis_xgb_supervised.pkl`, or with a new name if you change it manually at the beginning of the file.
For the purpose of this demonstration, it is not recommended though.

**Output:**  
Enriched JSON files with the new key `"pred_action":  "hit" | "bounce" | "air"` will be available in `output_results/`. One file is created per method, so you will find two files `supervised` and `unsupervised`. 

---

## Visualization

To visualize the results (Ground Truth vs Prediction) overlayed on the match videoand some EDA that helped building the physics-based model, please run the Jupyter Notebook :

**`notebooks/visualization.ipynb`**

---

## Requirements

- numpy==2.1.3
- pandas==2.3.3
- scipy==1.16.3
- scikit-learn==1.7.2
- xgboost==3.0.5
- joblib==1.5.2
- optuna==4.5.0
- tqdm==4.67.1
- matplotlib==3.10.0
- seaborn==0.13.2
- opencv-python==4.12.0.88
- ipykernel==6.30.0

---

## Video examples 

You will find in output_results two videos from random .json to assess the supervised model performance. 

In the result_point_102_comparison.mp4, you can see that the GT says "HIT" while it is a "BOUNCE" and that the model predicts it correctly. However, we can still see flaws, the model did not capture a HIT at all for instance.

**Author**: Hugo Niedzielski
