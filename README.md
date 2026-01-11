# Explainable Loan Approval System

# Problem Statement

- Banks need to make reliable loan approval decisions while being able to explain why an application was approved or rejected.
- This project builds an ML system that predicts loan approval and provides transparent reasoning behind decisions.

# Dataset

- Loan Prediction dataset
- Each record represents a loan applicant
- Target variable: loan_approved (1 = Approved, 0 = Rejected)

# Approach

- Performed EDA to understand key approval drivers
- Engineered financial features such as TotalIncome and Income-to-Loan Ratio
- Built a preprocessing pipeline using scikit-learn to handle missing values, encoding, and scaling
- Trained:
    - Logistic Regression (baseline)
    - Random Forest (final model)
- Evaluated using precision, recall, F1-score, and confusion matrix

# Explainability

- Used feature importance from Random Forest
- Found that Credit History, Income-to-Loan Ratio, and Total Income are the strongest decision factors
- Performed simple what-if analysis to show how changes in income affect approval probability

# Outcome

- Built an interpretable ML system aligned with real-world banking logic
- Model outputs are transparent and suitable for decision support

# Limitations & Future Work

- Dataset size is limited
- Can be extended with fairness analysis and web-based interface