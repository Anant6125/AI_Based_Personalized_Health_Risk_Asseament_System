import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Training all 4 models... this will take a few seconds.\n")

# ============ DIABETES ============
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("Diabetes accuracy:", round(model.score(X_test, y_test) * 100, 2))
joblib.dump(model, "diabetes_model.pkl")

# ============ HEART DISEASE ============
url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
data = pd.read_csv(url)
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("Heart Disease accuracy:", round(model.score(X_test, y_test) * 100, 2))
joblib.dump(model, "heart_model.pkl")

# ============ HYPERTENSION ============
url = "https://raw.githubusercontent.com/hrachsakanyan/diabetes-prediction-model/main/diabetes_prediction_dataset.csv"
data = pd.read_csv(url)
data["gender"] = data["gender"].map({"Male": 1, "Female": 0, "Other": 0})
smoke_map = {
    "never": 0,          # never smoked
    "former": 1,         # formerly smoked
    "not current": 1,    # formerly smoked (quit)
    "ever": 1,           # formerly smoked
    "current": 2,        # smokes
    "No Info": 3,        # Unknown
}
data["smoking_history"] = data["smoking_history"].map(smoke_map)
X = data[["gender", "age", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]]
y = data["hypertension"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("Hypertension accuracy:", round(model.score(X_test, y_test) * 100, 2))
joblib.dump(model, "hypertension_model.pkl")

# ============ STROKE ============
url = "https://raw.githubusercontent.com/YuvrazError/Healthcare-Dataset-Analysis/main/healthcare-dataset-stroke-data.csv"
data = pd.read_csv(url)
data = data.dropna(subset=["bmi"])
data = data[data["gender"] != "Other"]
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
data["ever_married"] = data["ever_married"].map({"Yes": 1, "No": 0})
data["smoking_status"] = data["smoking_status"].map({
    "never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3
})
X = data[["gender", "age", "hypertension", "heart_disease", "ever_married",
          "avg_glucose_level", "bmi", "smoking_status"]]
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("Stroke accuracy:", round(model.score(X_test, y_test) * 100, 2))
joblib.dump(model, "stroke_model.pkl")

print("\nAll 4 models trained and saved successfully!")