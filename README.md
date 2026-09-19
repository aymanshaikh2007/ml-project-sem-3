# ⚽ Football Player Market Value AI Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web platform for predicting European football player transfer market values based on physical stats, positional roles, disciplinary records, injury history, and per-90 performance metrics.

---

## 🌟 Key Features

- **🎯 Real-Time Valuation Engine**: Input player profile, stats, and durability to generate live estimated market values with 90% confidence bounds.
- **🔍 European Squad Explorer**: Filter, search, and perform head-to-head comparisons across 10,750+ European league players.
- **🤖 ML Model Benchmarks**: Comparative evaluation across 4 regression algorithms (Ridge, Random Forest, Gradient Boosting, LightGBM).
- **📈 Executive KPI Scorecard**: 5 integrated KPIs covering Business value, Model accuracy ($R^2 \ge 0.60$, $\text{MAE} \le \text{EUR 2.50M}$), Data quality, and Inference latency ($< 50\text{ms}$).
- **🎨 Cyber-Sports Dark Mode UI**: Translucent glassmorphism styling, glowing accent badges, micro-animations, and custom dark plots.

---

## 🚀 Quick Start (Local Setup)

### 1. Clone Repository
```bash
git clone https://github.com/aymanshaikh2007/ai-sem-3-project.git
cd ai-sem-3-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full ML Pipeline (Train & Export Artifacts)
```bash
python run_pipeline.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this repository to your GitHub account (`https://github.com/aymanshaikh2007/ai-sem-3-project`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New App** and select:
   - **Repository**: `aymanshaikh2007/ai-sem-3-project`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**! Your app will be live globally in seconds.

---

## 📁 Repository Structure

```
ai-sem-3-project/
├── app.py                     # Streamlit Web Application Interface
├── run_pipeline.py            # Master ML Pipeline (Cleaning, Training, Tuning, Artifact Export)
├── final_data.csv             # Transfermarkt Player Dataset (10,756 records)
├── requirements.txt           # Python Dependencies
├── report.md                  # Comprehensive Project Report
├── .gitignore                 # Git ignore rules
├── .streamlit/
│   └── config.toml            # Streamlit Theme Configuration
├── src/                       # Core Python Modules
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering & train/val/test split
│   ├── train_models.py        # Model training & RandomizedSearchCV hyperparameter tuning
│   ├── evaluate.py            # Regression metrics & KPI scorecard calculation
│   └── predict.py             # Inference engine class
├── models/                    # Trained Artifacts & Metadata
│   ├── best_model.pkl         # Production LightGBM Model
│   ├── feature_scaler.pkl     # StandardScaler Pipeline Object
│   ├── feature_columns.json   # Exact Feature Ordering
│   ├── kpi_metrics.json       # KPI Metrics JSON
│   └── model_comparison.csv   # Model Benchmark Comparison Table
└── data/
    └── processed_players.csv  # Processed Dataset for Exploration
```

---

## 📊 Model Performance Summary

| Model | Val Raw R² | Val Log R² | Val MAE (EUR) | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Tuned LightGBM (Final)** | **0.6737** | **0.5946** | **EUR 2,093,147** | 61.75s |
| **LightGBM (Baseline)** | 0.6681 | 0.5897 | EUR 2,100,353 | 0.60s |
| **Gradient Boosting** | 0.6086 | 0.5859 | EUR 2,164,945 | 6.76s |
| **Random Forest** | 0.5984 | 0.5683 | EUR 2,166,702 | 1.89s |
| **Ridge Regression** | -2.0078 | 0.4886 | EUR 2,984,196 | 0.01s |

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
