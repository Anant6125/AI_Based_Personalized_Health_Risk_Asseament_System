AI-Driven Personalized Health Risk Assessment

One Streamlit form that predicts risk of Diabetes, Heart Disease, Hypertension and Stroke using 4 Random Forest models (scikit-learn).

## Archivos- app.py : Streamlit web app (single input form, gages, comparison graph)

train_all.py: train the 4 models and save .pkl files

*_model.pkl : trained model

- requirements.txt: dependencies ## Run pip setup -r requirements.txt streamlit run app.py

## Test results (20% held-out data)

Diabetes : Acc 75.97%, ROC-AUC 0.829

Heart Disease Acc 81.97% ROC-AUC 0.906

Hypertension: Acc 65.52%, ROC-AUC 0.816.

Stroke: Acc 86.86 %, ROC-AUC 0.787

## To do SHAP explainability, model comparison (LogReg/XGBoost), imbalance handling, deployment.

Disclaimer: risk estimate only, not a medical diagnosis.
Note: in the heart dataset, label 0 = heart disease and 1 = no disease.
For the disease class: precision 0.95, recall 0.64, F1 0.77 (accuracy 81.97%, ROC-AUC 0.906).
