# ⚽ Football Player Market Value AI Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning decision support platform designed to appraise European football player market values based on physical characteristics, positional specializations, disciplinary records, injury histories, and per-90 match performance metrics.

---

## 🌟 Key Features

- **🎯 Live Valuation Calculator**: Calculate instant player market valuations (€ Millions/Thousands) with 90% confidence bounds based on real-time parameter inputs.
- **🔍 Squad & Player Explorer**: Browse 10,756 European player records with multi-column filtering and perform 2-player head-to-head comparisons.
- **🤖 Machine Learning Benchmarks**: Compare 4 baseline algorithms (Ridge Regression, Random Forest, Gradient Boosting, LightGBM) with hyperparameter tuning results.
- **📈 Executive KPI Scorecard**: Tracks 5 project KPIs across Business, Model Performance, Data Quality, and Product Engineering categories.
- **🎨 Modern Dark Mode UI**: Cyber-sports dark aesthetic featuring glassmorphism containers, smooth scroll animations, and dark-mode data visualizations.

---

## 🏗️ Architecture & Project Structure

```text
ml-project-sem-3/
├── final_data.csv             # Raw dataset (10,756 European player records)
├── run_pipeline.py            # Master ML pipeline runner (Preprocessing, Training, Tuning, Evaluation)
├── app.py                     # Streamlit Interactive Web Application UI
├── requirements.txt           # Python dependencies
├── report.md                  # Comprehensive academic project report
├── src/                       # Core Python Modules
│   ├── __init__.py
│   ├── data_processing.py     # Data cleaning, missing value handling, feature engineering & split
│   ├── train_models.py        # Model baseline training & hyperparameter tuning
│   ├── evaluate.py            # Evaluation metrics & KPI calculation logic
│   └── predict.py             # Real-time inference engine
├── models/                    # Saved Artifacts & Model Binary
│   ├── best_model.pkl         # Trained production model (LightGBM)
│   ├── feature_scaler.pkl     # Fitted StandardScaler object
│   ├── feature_columns.json   # Feature list order mapping
│   ├── kpi_metrics.json       # Exported KPI metrics scorecard
│   └── model_comparison.csv   # Benchmark results comparison matrix
└── data/                      # Processed Data Exports
    └── processed_players.csv  # Cleaned feature dataframe
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/aymanshaikh2007/ml-project-sem-3.git
cd ml-project-sem-3

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Master ML Pipeline (Optional - Pre-trained models included)
```bash
python run_pipeline.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### Deploying to Streamlit Community Cloud (Recommended)
1. Fork or push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"**, select `aymanshaikh2007/ml-project-sem-3`, set main file path to `app.py`.
4. Click **Deploy!**

---

## 📊 Model Performance & KPI Summary

| Category | Metric | Target | Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Business** | Transfer Overpay Risk Avoidance (MedAPE) | $< 25.0\%$ | $61.41\%$ | WARNING |
| **ML / Model 1** | Model Variance Explained (Log R²) | $\ge 0.60$ | **0.6320** | **PASSED** |
| **ML / Model 2** | Mean Absolute Error (MAE) | $\le \text{EUR 2.50M}$ | **EUR 2.47M** | **PASSED** |
| **Data Quality** | Data Pipeline Completeness & Zero-Leakage | $100.0\%$ | **100.0%** | **PASSED** |
| **Product / Eng**| Single Prediction Inference Latency | $< 50\text{ms}$ | **12.50 ms** | **PASSED** |

---

## 📜 License
This project is licensed under the MIT License.
