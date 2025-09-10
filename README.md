# Insurance-Loss-Analytics-Project
Here’s a clean, well-structured **README.md** draft for your GitHub repo based on the project details. It includes problem context, formulas, tasks, dataset descriptions, and variable definitions, without unnecessary class admin details.

---

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

Where:

* `X.15` = Total cost of claims during the current year
* `X.16` = Total number of claims during the current year

---

2. **Historically Adjusted Loss Cost (HALC):**

$$
HALC = \frac{X.15}{X.16} \times X.18
$$

Where:

* `X.18` = Ratio of claims filed to total duration (years) of policy in force

---

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
| X.2      | Policy start date                                                           |
| X.5      | Insured’s date of birth                                                     |
| X.7      | Contract channel (0 = Agent, 1 = Broker)                                    |
| X.8      | Years with insurance entity                                                 |
| X.12     | Number of policies canceled for nonpayment (current year)                   |
| X.14     | Net premium amount (current year)                                           |
| X.15     | Total cost of claims (current year)                                         |
| X.16     | Total number of claims (current year)                                       |
| X.17     | Total claims filed historically                                             |
| X.18     | Claims-to-duration ratio                                                    |
| X.19     | Vehicle risk type (1: Motorbike, 2: Van, 3: Passenger Car, 4: Agricultural) |
| X.20     | Region type (0 = Rural, 1 = Urban)                                          |
| X.22     | Vehicle registration year                                                   |
| X.23     | Vehicle power (horsepower)                                                  |
| X.25     | Market value of vehicle (as of 12/31/2019)                                  |
| X.27     | Energy source (P = Petrol, D = Diesel)                                      |
| X.28     | Vehicle weight (kg)                                                         |

(See full project document for all variable details.)

---

## ⚙️ Approaches Considered

* **Generalized Linear Models (GLM)** with Tweedie distribution
* **Tree-based models**: Random Forest, Gradient Boosting (XGBoost, LightGBM)
* **Neural networks** for nonlinear relationships
* **Model interpretation** using SHAP values

---

## 📈 Evaluation Metrics

* **MSE** → For LC and HALC predictions
* **ROC-AUC** → For claim status (CS) classification

---

## 🚀 Deliverables

* **Prediction file (`group_x_prediction.csv`)** containing three columns:

  * `LC`
  * `HALC`
  * `CS`

* **Project report** summarizing methodology, results, and business implications.


