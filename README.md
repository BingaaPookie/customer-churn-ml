# Project 2 — Customer Churn Prediction (Machine Learning)

## Goal
Train classification models to predict whether a telecom customer will churn.

## Models
- Logistic Regression
- Random Forest
- Gradient Boosting

The project compares the models and selects the best one using **F1-score**, because churn is an imbalanced binary classification problem and F1 balances precision and recall.

## Dataset
Same Telco Customer Churn dataset as Project 1, 7,043 rows, target `Churn`.

Source CSV:
https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Preprocessing
- `TotalCharges` converted to numeric.
- Missing numeric values: median imputation.
- Missing categorical values: most-frequent imputation.
- Categorical variables: one-hot encoding with `handle_unknown="ignore"`.
- Numeric variables: standard scaling.

The preprocessor is stored inside the same scikit-learn pipeline as the model, so training and Streamlit use exactly the same transformation steps.

## Train
```powershell
python train.py
```
This creates:
- `models/best_model.joblib`
- `models/metadata.json`
- `models/metrics.json`

## Run
```powershell
streamlit run app.py
```


