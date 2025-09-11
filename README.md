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

## 📊 Variable Descriptions 

| Variable | Description                                                                               |
| -------- | ----------------------------------------------------------------------------------------- |
| **X.1**  | Internal identification number assigned to each annual contract formalized by an insured. |
| **X.2**  | Start date of the policyholder’s contract *(DD/MM/YYYY)*.                                 |
| **X.3**  | Date of last contract renewal *(DD/MM/YYYY)*.                                             |
| **X.4**  | Date of the next contract renewal *(DD/MM/YYYY)*.                                         |
| **X.5**  | Date of birth of the insured declared in the policy *(DD/MM/YYYY)*.                       |
| **X.6**  | Date of issuance of the insured person’s driver’s license *(DD/MM/YYYY)*.                 |
| **X.7**  | Channel through which the policy was contracted *(0 = Agent, 1 = Insurance broker)*.      |
| **X.8**  | Total number of years the insured has been associated with the insurance entity.          |
| **X.9**  | Total number of policies held by the insured in the insurance entity.                     |
| **X.10** | Maximum number of policies that the insured has ever had in force.                        |
| **X.11** | Maximum number of products that the insured has simultaneously held at any given point.   |
| **X.12** | Number of policies canceled or terminated for nonpayment in the current year.             |
| **X.13** | Last payment method of the reference policy *(1 = half-yearly, 0 = annual)*.              |
| **X.14** | Net premium amount associated with the policy during the current year.                    |
| **X.15** | Total cost of claims for the insurance policy during the current year.                    |
| **X.16** | Total number of claims incurred for the insurance policy during the current year.         |
| **X.17** | Total number of claims filed throughout the entire duration of the policy.                |
| **X.18** | Ratio of number of claims filed to total duration (in years) of the policy in force.      |
| **X.19** | Type of risk *(1 = Motorbikes, 2 = Vans, 3 = Passenger cars, 4 = Agricultural vehicles)*. |
| **X.20** | Area type *(0 = Rural, 1 = Urban — more than 30,000 inhabitants)*.                        |
| **X.21** | Driver information *(1 = Multiple regular drivers declared, 0 = Single driver declared)*. |
| **X.22** | Year of vehicle registration *(YYYY)*.                                                    |
| **X.23** | Vehicle power measured in horsepower.                                                     |
| **X.24** | Cylinder capacity of the vehicle.                                                         |
| **X.25** | Market value of the vehicle as of 31/12/2019.                                             |
| **X.26** | Number of vehicle doors.                                                                  |
| **X.27** | Energy source used to power the vehicle *(P = Petrol, D = Diesel)*.                       |
| **X.28** | Vehicle weight in kilograms.                                                              |

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
