# Telco Customer Churn Prediction

> **End-to-end Machine Learning Pipeline** — From raw telecommunications customer data to production-ready churn prediction models with full reproducibility.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0+-green.svg)](https://xgboost.readthedocs.io)
[![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)

---

## 📌 Problem Statement

**Business Context:** A telecommunications company loses ~26.5% of its customers annually (churn). Each lost customer represents lost revenue, increased acquisition costs, and reduced market share.

**Objective:** Build a predictive model that identifies customers likely to churn *before* they leave, enabling proactive retention campaigns.

**Target Variable:** `Churn` (Binary: `Yes` = customer left, `No` = customer stayed)

**Success Metric:** **ROC-AUC** (primary) — balances true positive rate vs false positive rate across all thresholds, critical for imbalanced classification where Accuracy is misleading.

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | IBM Telco Customer Churn (Kaggle / IBM Sample Data) |
| **Samples** | 7,043 customers |
| **Features** | 20 (3 numerical, 17 categorical) |
| **Target** | Churn: 26.5% Yes / 73.5% No |
| **Missing Data** | 11 blank `TotalCharges` (all tenure=0, all non-churners) |
| **Duplicates** | None |
| **Unique IDs** | `customerID` (dropped — non-predictive) |

**Features:**
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

---

## 🔍 Exploratory Data Analysis — Key Findings

| Insight | Churn Rate | Business Implication |
|---------|-----------|---------------------|
| **Month-to-month contract** | 42.7% | Highest risk segment — target for retention offers |
| **Two-year contract** | 2.8% | Strong loyalty signal — low intervention needed |
| **Tenure 0–6 months** | 52.9% | Critical onboarding window — early engagement vital |
| **Tenure 61+ months** | 6.6% | Long-term customers rarely churn |
| **Fiber optic internet** | 41.9% | Premium service but high churn — investigate quality/price |
| **Electronic check payment** | 45.3% | Payment friction correlates with churn |
| **No Tech Support** | 41.6% vs 15.2% (with) | Support access reduces churn by 26pp |
| **No Online Security** | 41.8% vs 14.6% (with) | Security features strongly retain customers |
| **Senior citizens** | 41.7% vs 23.6% (non-seniors) | Age-specific retention strategies needed |

> **Key Takeaway:** Churn is driven by **contract flexibility**, **early tenure**, **service quality perception**, and **support accessibility** — not demographics alone.

---

## ⚙️ Preprocessing Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Drop 11 rows with blank `TotalCharges`** | All have `tenure=0` and `Churn=No` — logically $0 total charges. Keeping them as NaN would require imputation strategy with no clear benefit. |
| **Convert `SeniorCitizen` to categorical** | It's a binary flag (0/1), not a continuous measure. One-hot encoding treats it correctly. |
| **`ColumnTransformer`: StandardScaler + OneHotEncoder(drop='first')** | Scales numerical features for LR/KNN/SVM; one-hot with drop-first avoids dummy variable trap. `handle_unknown='ignore'` ensures production robustness. |
| **Stratified 80/20 train/test split** | Preserves 26.5% churn rate in both sets — essential for reliable evaluation on imbalanced data. |
| **No SMOTE/oversampling** | Tree-based models (RF, XGBoost) handle imbalance via `class_weight`/`scale_pos_weight`. LR uses `class_weight='balanced'`. ANNs use class weights in loss. |
| **`customerID` dropped** | Identifier, not predictive. Leaks no information but adds noise. |

---

## 🤖 Modeling Approach

### Traditional ML (3 models)
| Model | Tuning | Key Hyperparameters |
|-------|--------|---------------------|
| **Logistic Regression** | RandomizedSearchCV (50 iter, 5-fold CV) | `C` (log-uniform), `class_weight`, `solver` |
| **Random Forest** | RandomizedSearchCV (60 iter, 5-fold CV) | `n_estimators`, `max_depth`, `min_samples_split`, `class_weight` |
| **XGBoost** | RandomizedSearchCV (60 iter, 5-fold CV) | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |

### Deep Learning (3 ANN architectures)
| Architecture | Hidden Layers | Regularization |
|--------------|---------------|----------------|
| **ANN-1** | 32 → 16 | Dropout(0.3), L2(0.01) |
| **ANN-2** | 16 → 8 | Dropout(0.3), L2(0.01) |
| **ANN-3** | 64 → 32 → 16 | Dropout(0.3), L2(0.01) |

**Shared ANN Settings:** Adam (lr=0.001), Binary Crossentropy, Batch=32, Epochs=100, EarlyStopping(patience=10, restore_best_weights), 20% validation split from training data.

---

## 📈 Results — All Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.80** | 0.64 | 0.57 | 0.60 | **0.84** |
| **Random Forest** | 0.75 | 0.52 | **0.80** | 0.63 | **0.84** |
| **XGBoost** | 0.74 | 0.51 | **0.81** | 0.62 | **0.84** |
| **ANN-1 (32, 16)** | 0.80 | **0.66** | 0.54 | 0.59 | **0.84** |
| **ANN-2 (16, 8)** | 0.80 | 0.64 | 0.58 | **0.61** | **0.84** |
| **ANN-3 (64, 32, 16)** | 0.80 | **0.66** | 0.53 | 0.59 | **0.84** |

> All models achieve **ROC-AUC = 0.84** — strong discriminative ability.

### 🎯 Decision Rationale from Results

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| **RF & XGBoost: High Recall (0.80-0.81), Lower Precision** | Tree ensembles capture complex interactions, flag more true churners but with more false alarms. | **Choose if cost of missing a churner > cost of false retention offer.** |
| **Logistic Regression & ANNs: Higher Precision (0.64-0.66), Lower Recall** | Linear/ANN models are more conservative, predict churn only when very confident. | **Choose if retention budget is limited and offers are expensive.** |
| **ANN-2 (16,8) achieves best F1 (0.61)** | Smaller architecture generalizes slightly better on this dataset size. | **ANN-2 is best neural net candidate.** |
| **All ROC-AUC identical (0.84)** | Models rank customers similarly; threshold choice drives precision/recall tradeoff. | **Calibrate threshold per business cost matrix.** |
| **Feature importance aligns with EDA** | Contract, tenure, InternetService, PaymentMethod, TechSupport top features. | **Confirms causal hypotheses — model is learning real patterns.** |

---

## 🛠️ Skills Demonstrated

| Category | Competencies |
|----------|--------------|
| **Problem Framing** | Business-to-ML translation, metric selection for imbalance, stakeholder-aligned evaluation |
| **Data Engineering** | Data quality audit, missing value diagnosis, type conversion, leakage prevention |
| **EDA** | Univariate/bivariate/multivariate analysis, churn-rate segmentation, correlation analysis |
| **Feature Engineering** | Categorical encoding strategy, scaling rationale, train/test split discipline |
| **Model Development** | 6 diverse algorithms, hyperparameter tuning (RandomizedSearchCV), early stopping |
| **Model Evaluation** | ROC-AUC, Precision/Recall/F1, confusion matrices, cross-validation, learning curves |
| **Reproducibility** | Pinned dependencies, seeded randomness, single-command environment setup |
| **Communication** | Executive summary, technical rationale, actionable recommendations |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Git

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook 01_Telco_Customer_Churn_EDA.ipynb
```

### Run Notebook Headlessly (CI/CD)
```bash
# Execute all cells and save outputs
jupyter nbconvert --to notebook --execute --inplace 01_Telco_Customer_Churn_EDA.ipynb
```

### Expected Outputs
- All 36 code cells execute without errors
- 6 confusion matrices displayed
- Feature importance plots for RF & XGBoost
- Logistic Regression coefficient plot
- ANN training curves (accuracy/loss/AUC for 3 architectures)
- Model comparison bar chart & heatmap
- Final results table with all metrics

---

## 📁 Repository Structure

```
telco-customer-churn-prediction/
├── 01_Telco_Customer_Churn_EDA.ipynb   # Complete EDA + Modeling pipeline (executed)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Raw dataset (included for reproducibility)
├── requirements.txt                     # Pinned dependencies
├── .gitignore                          # Excludes cache, checkpoints, env files
└── README.md                           # This file
```

---

## 📦 Requirements (Pinned)

```
pandas==3.0.5
numpy==2.5.2
scikit-learn==1.5.2
matplotlib==3.11.1
seaborn==0.13.2
xgboost==3.4.1
tensorflow==2.21.0
nbformat==5.11.1
nbclient==0.11.0
jupyter-core==5.9.1
ipykernel==7.3.0
```

> Install with: `pip install -r requirements.txt`

---

## 🔒 Reproducibility Guarantees

- **Fixed random seeds:** `random_state=42` throughout (train/test split, CV, model initialization)
- **Pinned dependencies:** Exact versions in `requirements.txt`
- **Executed notebook:** All outputs saved — no re-run required to view results
- **No external API calls:** Fully offline execution

---

## 🎯 Next Steps (Production Readiness)

1. **Threshold Optimization** — Calibrate decision threshold using business cost matrix (cost of false positive vs false negative)
2. **Probability Calibration** — Apply Platt scaling / isotonic regression for reliable churn probabilities
3. **SHAP Explainability** — Global & local feature attribution for stakeholder trust
4. **Model Monitoring** — Data drift detection (PSI), prediction drift, performance tracking
5. **A/B Test Framework** — Champion/challenger deployment with statistical guardrails
6. **Feature Store** — Centralized feature definitions for training/serving consistency

---

## 📄 License

This project is for portfolio demonstration purposes. Dataset sourced from IBM Sample Data (public domain).

---

## 👤 Author

**Vishal** — Data Science / ML Engineering Portfolio  
*Demonstrating end-to-end ML pipeline competency: from raw data to business-ready insights.*

---

> **This repository showcases the ability to take a messy real-world dataset, diagnose data quality issues, perform rigorous exploratory analysis, engineer robust features, train and tune multiple model families, evaluate with business-appropriate metrics, and communicate findings with clear rationale — all in a reproducible, production-minded workflow.**