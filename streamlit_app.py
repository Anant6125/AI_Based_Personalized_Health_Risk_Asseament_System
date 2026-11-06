"""AI-Based Personalized Health Risk Assessment - full website (Streamlit).
Pages: Home | Risk Assessment | Model Performance | Explainability | About
Team: Anant Raj, Aadesh Agarwal, Abhinav Sahai Kamthan, Divyansh Saxena  |  Guide: Ms. Shruti Sharma
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI Health Risk Assessment", page_icon="🩺", layout="wide")

GITHUB = "https://github.com/Anant6125/AI_Based_Personalized_Health_Risk_Asseament_System"
MODEL_FILES = {"diabetes": "diabetes_model.pkl", "heart": "heart_model.pkl",
               "hypertension": "hypertension_model.pkl", "stroke": "stroke_model.pkl"}

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.hero { background: linear-gradient(90deg, #11998e, #38ef7d); padding: 38px; border-radius: 14px;
        text-align: center; margin-bottom: 26px; }
.hero h1 { color: white; margin: 0; font-size: 36px; }
.hero p { color: #eafff0; margin-top: 10px; font-size: 17px; }
.card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
        padding: 18px 20px; margin-bottom: 14px; }
.card h4 { margin: 0 0 6px 0; color: #38ef7d; }
.card p { margin: 0; color: #e8f1f2; font-size: 15px; }
.stat { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; text-align: center; }
.stat .num { font-size: 30px; font-weight: bold; color: #38ef7d; }
.stat .lab { font-size: 14px; color: #cfe3e6; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color: rgba(255,255,255,0.04); border-radius: 12px; padding: 10px; }
div.stButton > button { background: linear-gradient(90deg, #11998e, #38ef7d); color: white; font-weight: bold;
        padding: 12px 30px; border-radius: 10px; border: none; font-size: 17px; width: 100%; }
div.stButton > button:hover { opacity: 0.85; color: white; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers (work on old and new Streamlit)
def show_chart(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_image(path, caption=None):
    try:
        st.image(path, caption=caption, width="stretch")
    except TypeError:
        st.image(path, caption=caption, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def card(title, text):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def stat(num, label):
    st.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lab">{label}</div></div>', unsafe_allow_html=True)


def go_to(page):
    st.session_state["page"] = page


# ------------------------------------------------------------------ models
def train_models():
    """Fallback: retrain the four models (same settings as train_all.py) if the .pkl files cannot be used."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    def fit(X, y):
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42).fit(Xtr, ytr)

    d = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    m1 = fit(d.drop("Outcome", axis=1), d["Outcome"])
    d = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
    m2 = fit(d.drop("target", axis=1), d["target"])
    d = pd.read_csv("https://raw.githubusercontent.com/hrachsakanyan/diabetes-prediction-model/main/diabetes_prediction_dataset.csv")
    d["gender"] = d["gender"].map({"Male": 1, "Female": 0, "Other": 0})
    d["smoking_history"] = d["smoking_history"].map({"never": 0, "former": 1, "not current": 1, "ever": 1, "current": 2, "No Info": 3})
    m3 = fit(d[["gender", "age", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]], d["hypertension"])
    d = pd.read_csv("https://raw.githubusercontent.com/YuvrazError/Healthcare-Dataset-Analysis/main/healthcare-dataset-stroke-data.csv")
    d = d.dropna(subset=["bmi"])
    d = d[d["gender"] != "Other"]
    d["gender"] = d["gender"].map({"Male": 1, "Female": 0})
    d["ever_married"] = d["ever_married"].map({"Yes": 1, "No": 0})
    d["smoking_status"] = d["smoking_status"].map({"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3})
    m4 = fit(d[["gender", "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi", "smoking_status"]], d["stroke"])
    return {"diabetes": m1, "heart": m2, "hypertension": m3, "stroke": m4}


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        models = {k: joblib.load(v) for k, v in MODEL_FILES.items()}
        # smoke test: if a saved model is not compatible with the installed scikit-learn, retrain
        models["diabetes"].predict_proba(np.zeros((1, 8)))
        models["heart"].predict_proba(np.zeros((1, 13)))
        models["hypertension"].predict_proba(np.zeros((1, 7)))
        models["stroke"].predict_proba(np.zeros((1, 8)))
        return models
    except Exception:
        return train_models()


# "Typical person" reference values (median / most common value of the training data), for the what-if explanation.
REF = {
    "diabetes": [3.0, 117.0, 72.0, 23.0, 40.5, 32.3, 0.3825, 29.0],
    "heart": [55.5, 1.0, 0.0, 130.0, 240.0, 0.0, 1.0, 152.0, 0.0, 0.8, 2.0, 0.0, 2.0],
    "hypertension": [0.0, 43.0, 0.0, 0.0, 27.32, 5.8, 140.0],
    "stroke": [0.0, 44.0, 0.0, 0.0, 1.0, 91.685, 28.1, 0.0],
}
NAMES = {
    "diabetes": ["Pregnancies", "Glucose", "Blood pressure", "Skin thickness", "Insulin", "BMI", "Pedigree function", "Age"],
    "heart": ["Age", "Sex", "Chest pain type", "Resting BP", "Cholesterol", "Fasting sugar", "Rest ECG", "Max heart rate",
              "Exercise angina", "Oldpeak", "Slope", "Major vessels", "Thal"],
    "hypertension": ["Sex", "Age", "Heart disease", "Smoking", "BMI", "HbA1c", "Glucose"],
    "stroke": ["Sex", "Age", "Hypertension", "Heart disease", "Ever married", "Glucose", "BMI", "Smoking"],
}


def factor_impact(model, x, ref, class_index):
    """Change in risk (percentage points) when ONE input is replaced by the typical value."""
    base = model.predict_proba(np.array([x]))[0][class_index]
    out = []
    for i in range(len(x)):
        x2 = list(x)
        x2[i] = ref[i]
        out.append((base - model.predict_proba(np.array([x2]))[0][class_index]) * 100)
    return out


def make_gauge(title, value):
    color = "#ff4b4b" if value >= 50 else "#21c354"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 26}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)",
               'steps': [{'range': [0, 50], 'color': "#1e3d2f"}, {'range': [50, 100], 'color': "#3d1e1e"}],
               'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 50}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig


def impact_chart(title, names, impacts, top=5):
    order = sorted(range(len(names)), key=lambda i: abs(impacts[i]), reverse=True)[:top][::-1]
    fig = go.Figure(go.Bar(x=[impacts[i] for i in order], y=[names[i] for i in order], orientation="h",
                           marker_color=["#ff4b4b" if impacts[i] > 0 else "#21c354" for i in order],
                           text=[f"{impacts[i]:+.1f}" for i in order], textposition="outside"))
    fig.update_layout(title={'text': title, 'font': {'size': 16}}, height=300, margin=dict(l=20, r=40, t=50, b=30),
                      xaxis_title="Change in risk (percentage points)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig


# ------------------------------------------------------------------ pages
def page_home():
    st.markdown("""<div class="hero"><h1>🩺 AI Based Personalized Health Risk Assessment</h1>
    <p>One form. One click. Your estimated risk for diabetes, heart disease, hypertension and stroke.</p></div>""",
                unsafe_allow_html=True)
    st.button("Start the assessment", on_click=go_to, args=("Risk Assessment",))
    st.write("")
    c = st.columns(4)
    for col, (t, tx) in zip(c, [("Diabetes", "Based on glucose, BMI, age, insulin and family history."),
                                ("Heart Disease", "Based on 13 clinical readings such as cholesterol, heart rate and ECG."),
                                ("Hypertension", "Based on age, BMI, HbA1c, glucose and smoking history."),
                                ("Stroke", "Based on age, glucose, BMI, smoking and existing conditions.")]):
        with col:
            card(t, tx)
    st.subheader("How it works")
    h = st.columns(3)
    for col, (t, tx) in zip(h, [("1. Enter your details", "Fill in one short form. Shared details like age and BMI are asked only once."),
                                ("2. Four models run", "A separate Random Forest model is used for each disease."),
                                ("3. See your results", "Gauges, High or Low labels, a comparison chart and the factors behind each result.")]):
        with col:
            card(t, tx)
    st.subheader("Model snapshot")
    s = st.columns(4)
    for col, (n, l) in zip(s, [("4", "diseases in one tool"), ("81.97%", "heart disease accuracy"),
                               ("0.906", "best ROC-AUC (heart)"), ("~2 s", "time per prediction")]):
        with col:
            stat(n, l)
    st.info("This tool gives a risk estimate for awareness only. It is not a medical diagnosis. Please consult a doctor.")


def page_assess():
    st.markdown("""<div class="hero"><h1>Risk Assessment</h1><p>Fill the form once and see your risk for 4 major diseases</p></div>""",
                unsafe_allow_html=True)
    with st.container(border=True):
        st.header("Basic Info")
        age = st.number_input("Age", 1, 120, 30)
        sex = st.radio("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
    with st.container(border=True):
        st.header("Diabetes Info")
        preg = st.number_input("Pregnancies (put 0 if male)", 0, 20, 1)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    with st.container(border=True):
        st.header("Heart Info")
        cp = st.slider("Chest Pain Type", 0, 3, 1, help="Coded 0 to 3 as in the training dataset (not a severity scale).")
        trestbps = st.number_input("Resting BP", 0, 250, 120)
        chol = st.number_input("Cholesterol", 0, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120?", ["Yes", "No"])
        restecg = st.slider("Rest ECG", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate", 0, 250, 150)
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.slider("Slope", 0, 2, 1)
        ca = st.slider("Major Vessels", 0, 3, 0)
        thal = st.slider("Thal", 0, 3, 2)
    with st.container(border=True):
        st.header("Hypertension Info")
        hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
    with st.container(border=True):
        st.header("Extra Info for Stroke")
        hyp = st.radio("Hypertension?", ["Yes", "No"])
        heart_disease = st.radio("Heart Disease?", ["Yes", "No"])
        married = st.radio("Ever Married?", ["Yes", "No"])
        smoke = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("Predict All 4 Diseases"):
        models = load_models()
        s = 1 if sex == "Male" else 0
        f = 1 if fbs == "Yes" else 0
        e = 1 if exang == "Yes" else 0
        h = 1 if hyp == "Yes" else 0
        hd = 1 if heart_disease == "Yes" else 0
        m = 1 if married == "Yes" else 0
        smoke_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
        sm = smoke_map[smoke]
        x1 = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        x2 = [age, s, cp, trestbps, chol, f, restecg, thalach, e, oldpeak, slope, ca, thal]
        x3 = [s, age, hd, sm, bmi, hba1c, glucose]
        x4 = [s, age, h, hd, m, glucose, bmi, sm]
        risks = {
            "Diabetes": round(models["diabetes"].predict_proba(np.array([x1]))[0][1] * 100, 2),
            "Heart Disease": round(models["heart"].predict_proba(np.array([x2]))[0][0] * 100, 2),  # class 0 = heart disease in this dataset
            "Hypertension": round(models["hypertension"].predict_proba(np.array([x3]))[0][1] * 100, 2),
            "Stroke": round(models["stroke"].predict_proba(np.array([x4]))[0][1] * 100, 2),
        }
        impacts = {
            "Diabetes": factor_impact(models["diabetes"], x1, REF["diabetes"], 1),
            "Heart Disease": factor_impact(models["heart"], x2, REF["heart"], 0),
            "Hypertension": factor_impact(models["hypertension"], x3, REF["hypertension"], 1),
            "Stroke": factor_impact(models["stroke"], x4, REF["stroke"], 1),
        }
        st.session_state["result"] = {"risks": risks, "impacts": impacts}

    res = st.session_state.get("result")
    if res:
        risks, impacts = res["risks"], res["impacts"]
        st.write("---")
        st.subheader("Results")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(make_gauge("Diabetes", risks["Diabetes"]), key="g1")
        with c2:
            show_chart(make_gauge("Heart Disease", risks["Heart Disease"]), key="g2")
        c3, c4 = st.columns(2)
        with c3:
            show_chart(make_gauge("Hypertension", risks["Hypertension"]), key="g3")
        with c4:
            show_chart(make_gauge("Stroke", risks["Stroke"]), key="g4")
        for name, val in risks.items():
            (st.error if val >= 50 else st.success)(f"{name}: {'High' if val >= 50 else 'Low'} Risk - {val}%")
        if any(v >= 50 for v in risks.values()):
            st.warning("One or more estimates are High. This is not a diagnosis, but it is a good reason to talk to a doctor and get a proper check-up.")

        st.write("---")
        st.subheader("Risk Comparison - All 4 Diseases")
        vals = list(risks.values())
        bar = go.Figure(go.Bar(x=list(risks.keys()), y=vals, marker_color=["#ff4b4b" if v >= 50 else "#21c354" for v in vals],
                               text=[f"{v}%" for v in vals], textposition="outside"))
        bar.update_layout(yaxis_range=[0, 100], yaxis_title="Risk %", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=400)
        show_chart(bar, key="bar")

        st.write("---")
        st.subheader("Why this result? (top factors)")
        st.caption("Each bar shows how much the risk changes when that input is replaced by the value of a typical person in the "
                   "training data. Red bars push your risk up, green bars push it down. This is a what-if explanation, not a diagnosis.")
        k1, k2 = st.columns(2)
        with k1:
            show_chart(impact_chart("Diabetes", NAMES["diabetes"], impacts["Diabetes"]), key="i1")
            show_chart(impact_chart("Hypertension", NAMES["hypertension"], impacts["Hypertension"]), key="i3")
        with k2:
            show_chart(impact_chart("Heart Disease", NAMES["heart"], impacts["Heart Disease"]), key="i2")
            show_chart(impact_chart("Stroke", NAMES["stroke"], impacts["Stroke"]), key="i4")

    st.write("---")
    st.write("Disclaimer: this is just a risk estimate from ML models, not an actual medical diagnosis. Please consult a doctor.")


PERF = pd.DataFrame({
    "Disease": ["Diabetes", "Heart Disease", "Hypertension", "Stroke"],
    "Records": ["768", "303", "100,000", "~4,900"],
    "Positive cases (%)": [34.9, 45.5, 7.5, 4.3],
    "Accuracy (%)": [75.97, 81.97, 65.52, 86.86],
    "Precision": [0.66, 0.95, 0.16, 0.10],
    "Recall": [0.65, 0.64, 0.83, 0.26],
    "F1": [0.65, 0.77, 0.27, 0.15],
    "ROC-AUC": [0.829, 0.906, 0.816, 0.787],
})
COMPARE = pd.DataFrame([
    ["Diabetes", "Random Forest (used)", 75.97, 0.66, 0.65, 0.65, 0.829], ["Diabetes", "Logistic Regression", 73.38, 0.60, 0.70, 0.65, 0.825],
    ["Diabetes", "XGBoost", 74.68, 0.62, 0.70, 0.66, 0.824], ["Diabetes", "RF + SMOTE", 75.97, 0.63, 0.74, 0.68, 0.834],
    ["Diabetes", "RF + threshold 0.43", 75.32, 0.62, 0.78, 0.69, 0.829],
    ["Heart Disease", "Random Forest (used)", 81.97, 0.95, 0.64, 0.77, 0.906], ["Heart Disease", "Logistic Regression", 78.69, 0.83, 0.68, 0.75, 0.871],
    ["Heart Disease", "XGBoost", 77.05, 0.79, 0.68, 0.73, 0.846], ["Heart Disease", "RF + SMOTE", 80.33, 0.86, 0.68, 0.76, 0.891],
    ["Heart Disease", "RF + threshold 0.46", 78.69, 0.86, 0.64, 0.73, 0.906],
    ["Hypertension", "Random Forest (used)", 65.52, 0.16, 0.83, 0.27, 0.816], ["Hypertension", "Logistic Regression", 70.11, 0.17, 0.77, 0.28, 0.809],
    ["Hypertension", "XGBoost", 66.46, 0.16, 0.84, 0.27, 0.818], ["Hypertension", "RF + SMOTE", 71.25, 0.17, 0.72, 0.27, 0.791],
    ["Hypertension", "RF + threshold 0.65", 80.61, 0.21, 0.59, 0.31, 0.816],
    ["Stroke", "Random Forest (used)", 86.86, 0.10, 0.26, 0.15, 0.787], ["Stroke", "Logistic Regression", 73.93, 0.10, 0.67, 0.18, 0.800],
    ["Stroke", "XGBoost", 84.83, 0.15, 0.55, 0.24, 0.793], ["Stroke", "RF + SMOTE", 79.63, 0.10, 0.48, 0.17, 0.768],
    ["Stroke", "RF + threshold 0.45", 82.99, 0.10, 0.36, 0.15, 0.787],
], columns=["Disease", "Method", "Accuracy (%)", "Precision", "Recall", "F1", "ROC-AUC"])


def page_perf():
    st.markdown("""<div class="hero"><h1>Model Performance</h1><p>How well do the four models work on unseen test data?</p></div>""", unsafe_allow_html=True)
    st.write("Each model is a Random Forest (300 trees, depth 8, balanced class weights) evaluated on a held-out 20% test set "
             "(stratified split, random_state 42). Precision, recall and F1 are for the disease class.")
    show_table(PERF)
    fig = go.Figure()
    for col, color in [("Accuracy (%)", "#4c72b0"), ("Recall", "#dd8452"), ("ROC-AUC", "#55a868")]:
        vals = [v * 100 if col != "Accuracy (%)" else v for v in PERF[col]]
        fig.add_bar(name=col if col != "Recall" else "Recall (%)", x=PERF["Disease"], y=vals, marker_color=color,
                    text=[f"{v:.0f}" for v in vals], textposition="outside")
    fig.update_layout(barmode="group", height=420, yaxis_range=[0, 110], paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, legend=dict(orientation="h", y=-0.15))
    show_chart(fig, key="perf")
    st.warning("Accuracy alone can mislead. Only 4.3% of the stroke records are positive, so the stroke model shows 86.86% accuracy "
               "but finds only 26% of true cases (recall 0.26). The hypertension data (7.5% positive) behaves similarly.")
    st.subheader("Imbalance handling and model comparison")
    st.write("We compared the Random Forest used in this app with Logistic Regression, XGBoost, SMOTE oversampling and a tuned decision "
             "threshold (chosen on a validation part of the training data, never on the test set).")
    choice = st.selectbox("Disease", ["All"] + list(PERF["Disease"]))
    show_table(COMPARE if choice == "All" else COMPARE[COMPARE["Disease"] == choice])
    st.caption("Stroke recall rises to 0.67 with Logistic Regression, but precision stays about 0.10, so the stroke estimate is still not "
               "reliable for individuals. For heart disease, no alternative beats the Random Forest.")


def page_explain():
    st.markdown("""<div class="hero"><h1>Explainability (SHAP)</h1><p>Which inputs push each model's prediction up or down?</p></div>""", unsafe_allow_html=True)
    st.write("SHAP assigns every input a contribution to the model output. In each plot, every dot is one test record, the colour shows the "
             "input value (red = high, blue = low), and the position shows its effect on the prediction.")
    items = [("Diabetes", "diabetes", "Glucose, BMI and age have the strongest effect: high glucose pushes the risk up."),
             ("Heart Disease", "heart", "Chest pain type, thalassemia result and number of major vessels lead. In this dataset label 0 means heart disease."),
             ("Hypertension", "hypertension", "Age is the strongest input, followed by BMI. Higher HbA1c and heart disease push the risk up."),
             ("Stroke", "stroke", "Age is the strongest input, then average glucose and BMI. Existing hypertension raises the risk.")]
    if not all(os.path.exists(f"shap_images/{k}.png") for _, k, _ in items):
        st.info("SHAP images were not found in the shap_images folder.")
    else:
        for i in range(0, 4, 2):
            cols = st.columns(2)
            for col, (name, key, note) in zip(cols, items[i:i + 2]):
                with col:
                    show_image(f"shap_images/{key}.png", caption=name)
                    st.caption(note)
    st.info("SHAP values were computed offline on the test data. The 'Why this result?' charts on the Risk Assessment page use a "
            "simpler what-if method for your own inputs.")


def page_about():
    st.markdown("""<div class="hero"><h1>About the Project</h1><p>B.Tech Major Project, School of Computer Science and Engineering, IILM University, Greater Noida</p></div>""", unsafe_allow_html=True)
    st.subheader("Project")
    st.write("**AI-Based Personalized Health Risk Assessment System** (Project ID 26P_3CSE022). Diabetes, heart disease, hypertension and "
             "stroke often develop silently and are found late. Our tool lets a user enter basic health details once and see an early "
             "risk estimate for all four, as an aid to awareness.")
    st.subheader("Team")
    show_table(pd.DataFrame({"Name": ["Anant Raj", "Aadesh Agarwal", "Abhinav Sahai Kamthan", "Divyansh Saxena"],
                             "Roll No.": ["2410031035", "2410030324", "2410031548", "2410030172"]}))
    st.write("**Guide:** Ms. Shruti Sharma")
    st.subheader("Method")
    st.write("- Data: Pima Indians Diabetes, UCI Cleveland Heart Disease, a 100,000-record hypertension dataset and a healthcare stroke dataset.\n"
             "- Models: one Random Forest per disease (scikit-learn), 80/20 stratified split.\n"
             "- Evaluation: accuracy, precision, recall, F1 and ROC-AUC; imbalance handling; comparison with Logistic Regression and XGBoost; SHAP.\n"
             "- Web app: Streamlit and Plotly.")
    st.subheader("Limitations")
    st.write("The datasets are small, public and come from different populations; stroke and hypertension are highly imbalanced; the "
             "percentages are model scores, not calibrated probabilities; and the system has no clinical validation.")
    st.markdown(f"**Source code:** [{GITHUB}]({GITHUB})")
    st.error("Disclaimer: this website provides a risk estimate from machine learning models for awareness only. It is not a medical "
             "diagnosis or treatment advice. Please consult a doctor.")


# ------------------------------------------------------------------ navigation
PAGES = {"Home": page_home, "Risk Assessment": page_assess, "Model Performance": page_perf,
         "Explainability": page_explain, "About": page_about}
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
st.sidebar.title("🩺 Health Risk Assessment")
st.sidebar.radio("Go to", list(PAGES.keys()), key="page")
st.sidebar.caption("Awareness tool only. Not a medical diagnosis.")
PAGES[st.session_state["page"]]()
