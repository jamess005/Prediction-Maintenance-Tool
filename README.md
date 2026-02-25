# Predictive Maintenance — AI4I 2020

Binary machine failure prediction on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). The project covers the full ML workflow — EDA, feature engineering, Optuna hyperparameter tuning, class-imbalance handling, custom threshold selection, and deployment via FastAPI and Docker.

Two classifiers were developed and compared. The final **Random Forest** model achieves **0.930 MCC** on the held-out test set.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-James%20Scott-0077B5?logo=linkedin)](https://www.linkedin.com/in/jamesscott005)

---

## Results

| Model | Mean MCC | Recall | Precision | Threshold |
|-------|----------|--------|-----------|-----------|
| **Random Forest** | **0.930** | **0.912** | **0.954** | 0.48 |
| XGBoost | 0.880 | 0.897 | 0.872 | 0.57 |

Metrics are averaged across the independent validation and test splits (equal-size 10% each). MCC is used as the primary metric due to the severe class imbalance (~3.4% failure rate), where accuracy is uninformative.

**Selected model: Random Forest.** It outperforms XGBoost on every metric across both splits.

---

## Dataset

The AI4I 2020 dataset simulates a milling machine and contains 10,000 readings with five failure modes:

| Failure mode | Count | Handling |
|---|---|---|
| TWF — tool wear failure | 46 | Kept |
| HDF — heat dissipation failure | 115 | Kept |
| PWF — power failure | 95 | Kept |
| OSF — overstrain failure | 98 | Kept |
| RNF — random failure | 19 | **Dropped** — no learnable signal |

After dropping the 19 RNF rows, **9,981 rows** remain for training.

---

## Notebook Workflow

The notebooks document each stage of the project in order:

### 1 · `data_checks.ipynb` — EDA and data quality

- Confirmed the 1:28 failure-to-no-failure imbalance, which rules out accuracy as a metric and informed the use of `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Random Forest).
- Plotted raw sensor distributions (air temp, process temp, RPM, torque, tool wear) split by failure outcome — torque and tool wear showed the clearest separation.
- Ran a correlation analysis confirming torque and tool wear as the strongest individual predictors.
- **PCA on the raw sensors** (5 components) showed that 3 components capture ≥85% of the variance, with PC1 dominated by temperature, PC2 by torque/wear, and PC3 by rotation speed. This informed the choice of three engineered interaction features.
- Established baselines: a `most_frequent` Dummy classifier (high accuracy, zero skill) and Naive Bayes (positive MCC) — the floor any real model must clear.

### 2 · `feature_evaluation.ipynb` — Feature selection and engineering

- Applied `engineer_features()` to produce the full 9-feature set, then quantified the contribution of each feature.
- Point-biserial correlation of all 9 features vs failure confirmed the engineered features (`torque_wear`, `power_kW`) rank among the highest predictors alongside the raw sensors.
- Redundancy heatmap showed no near-±1.0 correlations — no features were dropped.
- 5-fold cross-validated XGBoost comparison (**raw-only 5 features vs all 9 features**) measured the direct MCC lift from the engineered features.
- Distribution plots per feature (failure vs no failure) confirmed clear bimodal or shifted distributions for the most predictive features.

### 3 · `evaluate_xgb.ipynb` — XGBoost model evaluation

- Confusion matrices and metrics on val and test splits.
- Full threshold sweep (0.10–0.95) with MCC, F1, recall, and precision curves — shows the near-plateau region where the selected threshold sits.
- Probability calibration using `calibration_curve` and Brier score.
- SHAP values: bar chart (mean |SHAP|) and beeswarm — `torque_wear`, `torque_Nm`, and `tool_wear_min` dominate.
- False-negative deep dive: scatter plots (torque vs wear, speed vs torque) and confidence score distributions for caught vs missed failures.

### 4 · `evaluate_rf.ipynb` — Random Forest model evaluation

- Same evaluation suite as XGBoost.
- Permutation importance (val set) used instead of SHAP — confirms similar feature ranking to XGBoost, with `torque_Nm` and `torque_wear` at the top.
- RF's MCC–threshold curve is noticeably flatter than XGBoost's — a wider plateau — which is why the recall-biased near-plateau selection produces a lower, more recall-friendly threshold (0.48 vs 0.57).

### 5 · `model_comparison.ipynb` — Side-by-side comparison

- Metric comparison table and bar chart (val and test, both models).
- Overlaid threshold sweep curves — shows RF's broader MCC hump.
- Calibration comparison.
- Combined miss-analysis scatter: how many failures each model misses, and where they sit in feature space.

---

## Feature Engineering

Five raw sensor readings are retained directly. Three interaction features are computed and the product quality grade is ordinal-encoded:

| Feature | Type | Formula / Source | Rationale |
|---------|------|------------------|-----------|
| `air_temp_K` | Raw | Air temperature (K) | Direct sensor |
| `proc_temp_K` | Raw | Process temperature (K) | Direct sensor |
| `rot_speed_rpm` | Raw | Rotational speed (RPM) | Direct sensor |
| `torque_Nm` | Raw | Torque (Nm) | Direct sensor |
| `tool_wear_min` | Raw | Tool wear (minutes) | Direct sensor |
| `power_kW` | Engineered | `torque_Nm × rot_speed_rpm / 9550` | Mechanical power — captures combined load |
| `temp_delta_K` | Engineered | `proc_temp_K − air_temp_K` | Cooling effectiveness — high delta → thermal stress |
| `torque_wear` | Engineered | `torque_Nm × tool_wear_min` | Cumulative stress proxy — high when both load and wear are elevated |
| `product_type` | Encoded | `L=0, M=1, H=2` | Ordinal product quality grade |

Failure-mode sub-type columns (TWF, HDF, PWF, OSF, RNF) are dropped before training — they are sub-labels of the target and represent direct data leakage.

---

## Methodology

### Class imbalance

The 1:28 imbalance was addressed differently per model:

- **XGBoost** — `scale_pos_weight = n_negative / n_positive ≈ 28.57`. This re-weights the loss function so each positive sample is worth 28× a negative sample.
- **Random Forest** — `class_weight='balanced'`. scikit-learn computes per-class weights as `n_samples / (n_classes × class_count)` and applies them during tree growth.

**SMOTE was evaluated and rejected.** Applied correctly (train-only), it reduced MCC below the baseline. The four distinct failure modes (TWF, HDF, PWF, OSF) occupy separated regions of feature space; oversampling generates synthetic points that blur the boundaries between them. The built-in weighting approaches proved more robust.

### Hyperparameter tuning

Both models are tuned with **Optuna TPE (Tree-structured Parzen Estimator)**, 100 trials, 5-fold stratified cross-validation maximising MCC:

- **XGBoost** search space: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- **Random Forest** search space: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

### Threshold selection

A fixed 0.5 threshold is suboptimal for imbalanced classification. The deployment threshold is selected on the validation set using a **recall-biased near-plateau method**:

1. Sweep thresholds from 0.10 to 0.95 in steps of 0.01.
2. Discard any threshold where recall drops below 80% — a missed failure is a maintenance miss.
3. Find the peak MCC among the remaining candidates.
4. Collect all thresholds within 0.01 MCC of that peak (the **near-plateau**).
5. Return the **lowest** threshold in that plateau — this maximises recall without meaningfully sacrificing MCC.

This is why RF gets a lower threshold (0.48) than XGBoost (0.57). RF's flatter MCC curve means many thresholds tie near the peak; the algorithm naturally drifts left toward higher recall.

---

## Project Structure

```
predmain/
├── data/
│   └── ai4i2020.csv                # AI4I 2020 dataset (10,000 rows)
├── notebooks/
│   ├── data_checks.ipynb           # EDA: imbalance, distributions, PCA, baselines
│   ├── feature_evaluation.ipynb    # Feature utility: correlation, redundancy, lift
│   ├── evaluate_xgb.ipynb          # XGBoost: confusion matrix, SHAP, calibration
│   ├── evaluate_rf.ipynb           # Random Forest: confusion matrix, permutation importance
│   └── model_comparison.ipynb      # Side-by-side XGB vs RF comparison
├── src/
│   ├── features.py                 # engineer_features(), get_feature_columns()
│   ├── models.py                   # Model builders, Optuna tuning, find_best_threshold()
│   ├── evaluate.py                 # plot_confusion_matrix(), metric helpers
│   ├── pipeline.py                 # End-to-end train → tune → evaluate → save
│   └── api.py                      # FastAPI inference endpoint
├── outputs/
│   ├── models/                     # rf_model.pkl, xgb_model.pkl
│   ├── figures/                    # Generated plots
│   └── reports/
├── Dockerfile
├── requirements.txt                # Full environment (pip freeze)
├── requirements-docker.txt         # Lean production dependencies
└── README.md
```

---

## Pipeline

```
Raw CSV  (10,000 rows, 14 columns)
  │
  ├─ Drop UDI, Product ID (identifiers)
  ├─ Drop 19 RNF rows (random failures — no learnable pattern)
  ├─ Rename bracket-heavy column names (required by XGBoost)
  ├─ Engineer power_kW, temp_delta_K, torque_wear
  ├─ Ordinal-encode product_type  (L=0, M=1, H=2)
  ├─ Drop failure sub-type columns  (TWF, HDF, PWF, OSF — data leakage)
  │
  └─ Stratified 80 / 10 / 10 split  (random_state=42)
         │
         ├─ Optuna TPE — 100 trials, 5-fold stratified CV, metric: MCC
         │     XGBoost:        scale_pos_weight = n_neg / n_pos
         │     Random Forest:  class_weight = 'balanced'
         │
         ├─ Fit final model on full training split
         │
         ├─ Recall-biased near-plateau threshold selection  (val set)
         │     min_recall=0.80, mcc_tolerance=0.01
         │
         ├─ Evaluate on val + test
         │
         └─ Save  {model, threshold, model_type}  →  outputs/models/*.pkl
```

---

## Quick Start

### Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train Random Forest (recommended)
python src/pipeline.py rf

# Train XGBoost
python src/pipeline.py xgb

# Interactive selection
python src/pipeline.py
```

### API

```bash
PYTHONPATH=src uvicorn api:app --reload

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temp_K": 298.1,
    "proc_temp_K": 308.6,
    "rot_speed_rpm": 1551,
    "torque_Nm": 42.8,
    "tool_wear_min": 0,
    "product_type": "M"
  }'

# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"air_temp_K": 298.1, "proc_temp_K": 308.6, "rot_speed_rpm": 1551, "torque_Nm": 42.8, "tool_wear_min": 0,   "product_type": "M"},
    {"air_temp_K": 305.0, "proc_temp_K": 315.0, "rot_speed_rpm": 1200, "torque_Nm": 70.0, "tool_wear_min": 220, "product_type": "L"}
  ]'

# Model metadata
curl http://localhost:8000/model/info

# Retrain from dataset on disk
curl -X POST "http://localhost:8000/retrain?model_type=rf"
```

### Docker

```bash
# Build the image (includes pre-trained model — no training required)
docker build -t predmain .

# Run
docker run -p 8000:8000 predmain

# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

The container copies `outputs/models/` at build time so the pre-trained RF model is available immediately. Training can be triggered at runtime via `POST /retrain` if needed.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — confirms model is loaded |
| `GET` | `/model/info` | Model type, threshold, and feature list |
| `POST` | `/predict` | Single `SensorReading` → failure probability + binary prediction |
| `POST` | `/predict/batch` | Array of `SensorReading` → predictions |
| `POST` | `/retrain?model_type=rf` | Retrain from `data/ai4i2020.csv` on disk, reload model |

**Request body for `/predict`:**
```json
{
  "air_temp_K":    300.0,
  "proc_temp_K":   310.0,
  "rot_speed_rpm": 1500,
  "torque_Nm":     40.0,
  "tool_wear_min": 100,
  "product_type":  "M"
}
```

**Response:**
```json
{
  "failure_probability": 0.12,
  "failure_predicted":   false,
  "threshold":           0.48,
  "model_type":          "rf"
}
```

The API accepts the five raw sensor readings and product type. Feature engineering (`power_kW`, `temp_delta_K`, `torque_wear`, `product_type` encoding) is applied server-side before inference.

---

## Tech Stack

| Category | Libraries |
|---|---|
| ML | scikit-learn, XGBoost, Optuna |
| Explainability | SHAP (XGBoost), permutation importance (RF) |
| API | FastAPI, Uvicorn, Pydantic |
| Visualisation | Matplotlib, Seaborn |
| Packaging | Docker |
| Language | Python 3.12 |

---

## License

[MIT](LICENSE) © James Scott

