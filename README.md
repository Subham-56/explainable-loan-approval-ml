# Loan Approval Prediction System

## Problem Statement

Banks need reliable loan approval predictions, but model behavior should still be interpretable enough for analysis and decision support.

This project builds a machine learning pipeline that predicts whether a loan application is approved and examines the main factors influencing those predictions.

## Dataset

- Source: Loan Prediction dataset
- Each row represents one loan applicant
- Original target: `Loan_Status`
- Derived binary target: `loan_approved` where `1 = Approved` and `0 = Rejected`
- Dataset size used in the project: 614 rows

## Project Workflow

The project is organized across four notebooks:

- `Notebooks/01_data_and_label.ipynb`: loads the raw dataset and creates the `loan_approved` target
- `Notebooks/02_eda.ipynb`: explores distributions and approval patterns
- `Notebooks/03_feature_engineering.ipynb`: creates `TotalIncome` and `Income_Loan_Ratio`, then saves `Data/featured_loan_data.csv`
- `Notebooks/04_modeling.ipynb`: builds preprocessing pipelines, trains models, evaluates them, and saves outputs

## Features and Preprocessing

- Engineered features:
  - `TotalIncome = ApplicantIncome + CoapplicantIncome`
  - `Income_Loan_Ratio = TotalIncome / LoanAmount`
- Numerical data is handled with median imputation and standard scaling
- Categorical data is handled with most-frequent imputation and one-hot encoding
- The preprocessing steps are included inside scikit-learn pipelines

## Models Trained

- Logistic Regression as a baseline model
- Random Forest Classifier as the final saved model

## Evaluation

The notebooks evaluate models using classification reports and confusion matrices.

Observed test-set results in the notebook:

- Logistic Regression:
  - Accuracy: `0.84`
- Random Forest:
  - Accuracy: `0.85`

## Explainability

The current project includes:

- Random Forest feature importance analysis
- Simple manual what-if checks using `predict_proba` on modified sample inputs

This repository does not currently include per-loan explanation methods such as SHAP or LIME, and it does not expose explanations through an application interface.

## Saved Outputs

- Featured dataset: `Data/featured_loan_data.csv`
- Trained model pipeline: `Models/loan_approval_model.pkl`
- Prediction output with approval probability: `Models/loan_predictions.csv`

## Limitations

- The dataset is relatively small
- Explainability is limited to global feature importance and notebook-based what-if analysis
- There is no deployed web app or API in this repository
- Some engineered `Income_Loan_Ratio` values remain missing in the saved feature dataset and are handled later by the model preprocessing pipeline

## Future Improvements

- Recompute engineered features after missing-value handling so saved feature files are fully consistent
- Add stronger explainability with SHAP or LIME
- Add fairness and bias evaluation
- Build a simple web app or API for inference
