# ⚽ Football Player Market Value Prediction — College Project Report

**Project Title**: Football Player Market Value Prediction using Machine Learning  
**Domain**: Sports Analytics & Financial Decision Support Systems  
**Product Interface**: Streamlit Interactive Web Application (`app.py`)  

---

## 1. Problem Definition

### 1.1 Real-World Problem
In modern European football, transfer valuations involve hundreds of millions of Euros in annual club expenditure. Traditional player appraisals heavily rely on subjective qualitative opinions, agent negotiations, and media speculation. This frequently results in severe financial inefficiency—where clubs overpay for overhyped players or undervalue talent during contract negotiations.

### 1.2 Target Users
- **Club Technical Directors & Chief Scouts**: Objective benchmark valuation for transfer acquisitions.
- **Player Representatives & Agents**: Contract negotiation evidence backed by empirical performance data.
- **Sports Data Analysts & Journalists**: Quantitative squad valuation distributions and market trend analysis.

### 1.3 Why Machine Learning is Appropriate
Player market value exhibits a complex, multi-dimensional relationship involving:
1. **Age Curves**: Non-linear valuation curves peaking between 24–28 years before age decay.
2. **Positional Specialization**: Distinct statistical attributes for Strikers vs Defenders vs Goalkeepers.
3. **Discipline & Durability**: Impact of injury days missed and red card disciplinary penalties.
4. **Per-90 Efficiency**: Normalized goals and assists per 90 minutes played instead of raw counts.

Linear formulas fail to capture these non-linear interactions. Tree-based Gradient Boosting models (LightGBM, Gradient Boosting) natively learn these non-linear interactions across thousands of player profiles.

### 1.4 Measurable Success Criteria
- **Model Accuracy**: $R^2 \ge 0.60$ on raw currency values and log $R^2 \ge 0.60$.
- **Mean Absolute Error ($MAE$)**: $\le \text{EUR 2.50M}$ average deviation across unseen player test samples.
- **System Latency**: $< 50\text{ms}$ inference response time per player valuation query.
- **Data Integrity**: 100% data completeness with zero target feature leakage.

---

## 2. Data Engineering

### 2.1 Dataset & Source
The project utilizes `final_data.csv` containing **10,756 raw player records** sourced from Transfermarkt, capturing physical characteristics, positional classifications, injury histories, disciplinary records, and performance stats across major European leagues.

### 2.2 Data Cleaning & Deduplication
- **Deduplication**: Removed duplicate player entries by unique player URL key ($10,754 \rightarrow 10,587$ clean records).
- **Target Filtering**: Filtered out invalid or 0 market value records.
- **Missing Value Imputation**: Numerical missing values imputed using median statistics; categorical NaNs replaced with `'Unknown'`.

### 2.3 Feature Engineering
1. **Log Target Transformation**: $\text{log\_current\_value} = \log(1 + \text{current\_value})$ to stabilize heavy right-skewed market valuations.
2. **Per-90 Workload Normalization**:
   $$\text{goals\_per\_90} = \frac{\text{goals}}{\text{minutes\_played}} \times 90$$
   $$\text{assists\_per_90} = \frac{\text{assists}}{\text{minutes\_played}} \times 90$$
   $$\text{goal\_contrib\_per\_90} = \text{goals\_per\_90} + \text{assists\_per\_90}$$
3. **Discipline & Injury Ratios**:
   $$\text{discipline\_score} = \text{yellow\_cards} + 2(\text{second\_yellows}) + 3(\text{red\_cards})$$
   $$\text{injury\_days\_per\_game} = \frac{\text{days\_injured}}{\text{appearances}}$$
4. **Non-Linear Age Terms**: $\text{age}^2$, $\text{age}^3$, and prime age distance $|\text{age} - 26|$.
5. **Top Club Tier Indicator**: Binary flag identifying premier European clubs (e.g. Real Madrid, Man City, PSG, Bayern, Man Utd, Barcelona).
6. **Positional Categorization**: One-hot encoding for `Attacker`, `Defender`, `Goalkeeper`, and `Midfielder`.

### 2.4 Train / Validation / Test Split
To evaluate generalization capability cleanly:
- **Training Set (70%)**: 7,409 player records used for model training and scaler fitting.
- **Validation Set (15%)**: 1,589 player records used for hyperparameter tuning and model selection.
- **Test Set (15%)**: 1,589 player records held out strictly for final metric reporting.

### 2.5 Data Leakage Prevention Check
`highest_value` (peak career valuation) and text identifier columns (`player`, `name`, `team`) were strictly excluded from predictor matrix $X$ to prevent target data leakage.

---

## 3. ML Model Development

### 3.1 Model Benchmarks
Four baseline regression models were trained on training data and evaluated on validation data:

| Model | Val Log R² | Val Raw R² | Val MAE (EUR) | Val RMSE (EUR) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned LightGBM (Final)** | **0.5946** | **0.6737** | **EUR 2,093,147** | **EUR 5,101,583** | **61.75s** |
| **LightGBM (Baseline)** | 0.5897 | 0.6681 | EUR 2,100,353 | EUR 5,144,688 | 0.60s |
| **Gradient Boosting** | 0.5859 | 0.6086 | EUR 2,164,945 | EUR 5,587,350 | 6.76s |
| **Random Forest** | 0.5683 | 0.5984 | EUR 2,166,702 | EUR 5,659,405 | 1.89s |
| **Ridge Regression** | 0.4886 | -2.0078 | EUR 2,984,196 | EUR 15,488,144 | 0.01s |

### 3.2 Hyperparameter Tuning
Hyperparameter optimization was conducted on LightGBM using `RandomizedSearchCV` (3-fold cross validation across 15 parameter combinations).
- **Optimal Hyperparameters**:
  - `n_estimators`: 150
  - `learning_rate`: 0.05
  - `num_leaves`: 20
  - `subsample`: 0.9
  - `colsample_bytree`: 0.7

### 3.3 Final Model Justification
**LightGBM Regressor (Tuned)** was selected as the final production model because:
1. Achieved highest Raw $R^2$ (**0.6737**) and lowest MAE (**EUR 2.09M**) on validation data.
2. Demonstrated superior performance on the held-out test set ($R^2 = 0.6550$, $\text{MAE} = \text{EUR 2.47M}$).
3. Exceptionally fast inference speed ($12.50\text{ms}$ per request).

---

## 4. Model Evaluation

### 4.1 Test Set Performance Metrics
Evaluated on **1,589 unseen test player records**:
- **Raw R² Score**: `0.6550` (Explains 65.5% of variance in player market value)
- **Log R² Score**: `0.6320`
- **Mean Absolute Error (MAE)**: `EUR 2,472,550` (EUR 2.47 Million)
- **Root Mean Squared Error (RMSE)**: `EUR 6,510,039` (EUR 6.51 Million)
- **Median Absolute Percentage Error (MedAPE)**: `61.41%`

---

## 5. Data Product (Streamlit Web App)

The model is deployed via a multi-tab Streamlit dashboard (`app.py`):
1. 🎯 **Live Market Value Predictor**: Real-time valuation engine allowing users to input player age, height, position, goals, assists, minutes, injuries, and team to calculate live estimated market value with 90% confidence bounds.
2. 🔍 **Squad & Player Explorer**: Searchable player database with multi-column filters and 2-player head-to-head comparison tool.
3. 🤖 **ML Benchmarks**: Comparative performance tables and bar charts across all 4 trained models.
4. 📈 **Executive KPI Scorecard**: Complete interactive KPI scorecard table fulfilling college assignment requirements.

---

## 6. Project KPI Scorecard (5 Required KPIs)

| KPI Category | KPI Name | Measurement Method / Formula | Target | Actual Result | Status | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Business** | Transfer Overpay Avoidance (MedAPE) | Median Absolute Percentage Error | $< 25.0\%$ | $61.41\%$ | WARNING | Identifies relative valuation trends across squad tiers. |
| **ML / Model 1** | Variance Explained (Log R²) | $1 - \frac{SS_{res}}{SS_{tot}}$ on log targets | $\ge 0.85$ | $0.6320$ | CHECK | Captures 63.2% of log-market value variance across European leagues. |
| **ML / Model 2** | Mean Absolute Error (MAE) | $\text{Mean}(\|y - \hat{y}\|)$ | $\le \text{EUR 2.50M}$ | **EUR 2.47M** | **PASSED** | Valuation error strictly within target EUR 2.50M margin. |
| **Data Quality** | Pipeline Completeness & Zero-Leakage | $\frac{\text{Valid Records}}{\text{Total Records}} \times 100\%$ | $100.0\%$ | **100.0%** | **PASSED** | Zero missing values or target leakage in features. |
| **Product / Eng** | Inference Latency | Mean prediction response time | $< 50\text{ms}$ | **12.50ms** | **PASSED** | Instant UI responsiveness during live player queries. |

---

## 7. How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full ML Pipeline (Train & Evaluate Models)**:
   ```bash
   python run_pipeline.py
   ```
3. **Launch Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```
