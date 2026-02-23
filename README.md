# Predictive Maintenance Tool

A structured ML project for predictive maintenance classification using the AI4I 2020 synthetic milling dataset.

## Objective
Build a rigorous ML pipeline focusing on proper data auditing, baseline establishment, feature engineering, model comparison, and robustness testing — moving beyond ad-hoc model training toward a repeatable, documented process.

## Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Project Structure
```
predmain/
├── data/               # Raw data (not tracked in git)
├── notebooks/          # Exploratory analysis and phase notebooks
├── src/                # Reusable Python modules
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── utils.py
├── outputs/            # Figures, saved models, reports (not tracked)
├── requirements.txt
└── README.md
```

## Phases
1. Data Audit
2. Baseline Establishment
3. Feature Engineering
4. Model Suite Comparison
5. Hyperparameter Tuning
6. Robustness & Drift Testing

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Optuna, SHAP, Seaborn, PyTorch (ROCm)
