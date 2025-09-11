# Insurance Loss Analytics Project

## 📌 Project Overview

This project focuses on predicting **insurance claim losses**, which is critical for setting fair and competitive premiums in the insurance industry. Accurate prediction helps avoid **adverse selection** (overpricing low-risk customers and underpricing high-risk ones) and ensures a balanced, profitable portfolio.

Insurance claim data is highly skewed, with many policyholders reporting **zero claims**, making modeling challenging. Such data is often assumed to follow a **Tweedie distribution**, which can handle both claim frequency and severity.

We explore predictive modeling approaches (statistical and machine learning) to estimate:

1. **Loss Cost per Exposure Unit (LC)**
2. **Historically Adjusted Loss Cost (HALC)**
3. **Claim Status (CS: whether a claim occurs)**

---

## 📂 Datasets

Two datasets are used:

* **insurance\_train.csv** → Training dataset with 37,451 policy transactions
* **insurance\_test.csv** → Test dataset with 15,787 policy transactions

Both datasets include variables **X.1 – X.28**, though some are excluded from the test set (X.1, X.15, X.16, X.17, X.18).

---

## 🧮 Key Formulas

1. **Loss Cost per Exposure Unit (LC):**

$$
LC = \frac{X.15}{X.16}
$$

2. **Historically Adjusted Loss Cost (HALC):**

$$
HALC = \frac{X.15}{X.16} \times X.18
$$

3. **Claim Status (CS):**

$$
CS =
\begin{cases} 
1 & \text{if a claim occurs} \\ 
0 & \text{otherwise} 
\end{cases}
$$

---

## 🎯 Project Tasks

1. **Task 1: Predict LC and HALC**

   * Build models to predict continuous insurance metrics: LC and HALC.
   * Evaluate models using **Mean Squared Error (MSE)**.

2. **Task 2: Predict Claim Likelihood (CS)**

   * Binary classification problem.
   * Evaluate using **ROC-AUC**.

---

## 📊 Variable Descriptions (Selected)

| Variable | Description                                                                 |
| -------- | --------------------------------------------------------------------------- |
| X.1      | Internal policy ID                                                          |
| X.7      | Contract channel (0 = Agent, 1 = Broker)                                    |
| X.8      | Years with insurance entity                                                 |
| X.12     | Number of policies canceled for nonpayment (current year)                   |
| X.14     | Net premium amount (current year)                                           |
| X.15     | Total cost of claims (current year)                                         |
| X.16     | Total number of claims (current year)                                       |
| X.18     | Claims-to-duration ratio                                                    |
| X.19     | Vehicle risk type (1: Motorbike, 2: Van, 3: Passenger Car, 4: Agricultural) |
| X.20     | Region type (0 = Rural, 1 = Urban)                                          |
| X.22     | Vehicle registration year                                                   |
| X.23     | Vehicle power (horsepower)                                                  |
| X.25     | Market value of vehicle                                                     |
| X.27     | Energy source (P = Petrol, D = Diesel)                                      |
| X.28     | Vehicle weight (kg)                                                         |

---

## ⚙️ Methodology

### 1. Data Stratification

To ensure representative training and validation splits, we created a composite **“Strata” column** by combining:

* Claim Status (CS)
* Percentile-binned LC
* Percentile-binned HALC

This stratification preserved the distribution of all targets across folds.

---

### 2. Feature Engineering

We engineered several new predictors:

* `age_at_last_renewal` → Captures demographic risk
* `driving_experience` → Years since license issuance
* `vehicle_age` → Depreciation effects
* `policy_tenure` → Customer loyalty
* `claim_frequency_total` → Normalized historical claim frequency

---

### 3. Feature Selection

We applied **Recursive Feature Elimination with Cross-Validation (RFECV)** to reduce dimensionality and select impactful variables. This reduced model complexity, improved interpretability, and maintained predictive power.

---

### 4. Modeling Approaches

We experimented with multiple strategies:

* **Individual Models (Baseline):** Separate models for CS (classification) and LC/HALC (regression).
* **Two-Step Approach:** Predict CS first, then predict LC/HALC only for likely claimants.
* **Three-Headed Neural Network:** Multi-task learning with a shared network and three outputs (CS, LC, HALC).
* **Final Approach (Chosen):**

  * Stage 1: Train **CatBoostClassifier** for CS → generate probability scores (`p_cs_1`).
  * Stage 2: Feed `p_cs_1` into **LightGBM regressors** (with Tweedie loss) for LC and HALC.
  * To avoid leakage, `p_cs_1` was generated with out-of-fold cross-validation.

---

### 5. Model Training & Optimization

* **Cross-validation:** 3-fold CV to ensure generalizability.
* **Hyperparameter tuning:** Performed using **Optuna** (Bayesian optimization).
* **Interpretability:** Applied **SHAP values** to analyze feature importance and business relevance.

---

## 📈 Results

* **Claim Status (CS):** CatBoost Classifier → **AUC ≈ 0.822**
* **Loss Cost (LC):** LightGBM Regressor (Tweedie) → **RMSE ≈ 835.63**
* **HALC:** LightGBM Regressor (Tweedie) → **RMSE ≈ 1659.27**

**Key Predictors:**

* Claim status probability (`p_cs_1`)
* Policy tenure & years with company
* Vehicle characteristics (age, cylinder capacity, horsepower, weight)

---

## 🚀 Deliverables

* **Prediction file (`group_x_prediction.csv`)** containing three columns:

  * `LC`
  * `HALC`
  * `CS`

* **Project report** summarizing methodology, results, and business implications.

---

✅ This README now combines both the **project brief** and your **implemented methodology/results**.

Do you want me to also include a **“Future Work”** section from your report (e.g., autoencoder augmentation, limitations, business insights), or should I keep it focused on the main final approach?
