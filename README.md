
# CreditWise Loan System

> An intelligent, ML-powered loan approval prediction system for SecureTrust Bank — automating credit decisions with speed, accuracy, and fairness.

---

## Problem Statement

**SecureTrust Bank** processes hundreds of loan applications daily across urban and rural India. Their manual loan verification process was:
- Time-consuming
- Biased and inconsistent
- Causing good customers to be rejected
- Approving high-risk customers, leading to financial losses

**CreditWise** solves this with a Machine Learning system that predicts loan approval outcomes before final human verification.

---

## Objective

Build a binary classification model to predict whether a loan application should be:
- **Approved** (1)
- **Rejected** (0)

---

## Project Structure

```
creditwise-loan-system/
│
├── credit_wise.ipynb          # Main Jupyter Notebook
├── loan_approval_data.csv     # Dataset (not included)
└── README.md                  #Project documentation
```

---

## Dataset Description

Each row represents a **loan applicant** with 20 features:

| Column | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Age` | Applicant's age |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of active loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration in months |
| `Loan_Purpose` | Home / Education / Personal / Business |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt / Private / Self |
| `Loan_Approved` | **Target** — 1 = Approved, 0 = Rejected |

---

##  Workflow

### 1. Data Loading & Exploration
- Load dataset using `pandas`
- Inspect shape, data types, and null values via `df.info()` and `df.isnull().sum()`

### 2. Data Preprocessing
- **Numerical imputation**: Missing values filled with column mean (`SimpleImputer(strategy="mean")`)
- **Categorical imputation**: Missing values filled with most frequent value (`strategy="most_frequent"`)
- **Drop** `Applicant_ID` (non-predictive unique key)

### 3. Exploratory Data Analysis (EDA)
- Class balance pie chart for `Loan_Approved`

### 4. Encoding
- **Label Encoding**: `Education_Level`, `Loan_Approved`
- **One-Hot Encoding** (drop first): `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`

### 5. Feature Engineering
- Added squared features: `DTI_Ratio_sq`, `Credit_Score_sq`
- Dropped original `DTI_Ratio` and `Credit_Score` after transformation

### 6. Train-Test Split & Scaling
- 80/20 split (`random_state=42`)
- `StandardScaler` applied to training and test sets

### 7. Model Training & Evaluation
Three classifiers trained and evaluated:

| Model | Key Metric Focus |
| Logistic Regression | Baseline linear classifier |
 Distance-based classifier |

**Evaluation Metrics Used**: Precision, Recall, F1 Score, Accuracy, Confusion Matrix

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualizations |
| Scikit-learn | ML models, preprocessing, evaluation |
| Jupyter Notebook | Development environment |

---

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Run the Notebook
```bash
jupyter notebook credit_wise.ipynb
```

### Required File
Place `loan_approval_data.csv` in the same directory as the notebook before running.

---

##  Key Insights

- **Credit Score** and **DTI Ratio** are among the most correlated features with loan approval
- Squaring these features (feature engineering) helps capture non-linear relationships
- Class imbalance should be monitored — the pie chart reveals the approval/rejection ratio
- Applicants with higher savings and lower DTI ratios are more likely to be approved

---

##  License

This project is for educational purposes as part of a Machine Learning assignment.

---

