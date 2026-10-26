import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Health Risk Assessment", page_icon="🩺", layout="wide")

# ---- custom styling so it looks like a real website, not a plain form ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.hero {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    padding: 35px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 30px;
}
.hero h1 {
    color: white;
    margin: 0;
    font-size: 34px;
}
.hero p {
    color: #eafff0;
    margin-top: 10px;
    font-size: 16px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 10px;
}
div.stButton > button {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: white;
    font-weight: bold;
    padding: 12px 30px;
    border-radius: 10px;
    border: none;
    font-size: 17px;
    width: 100%;
}
div.stButton > button:hover {
    opacity: 0.85;
    color: white;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- hero banner at top ----
st.markdown("""
<div class="hero">
<h1>🩺 AI Based Personalized Health Risk Assessment</h1>
<p>Fill this form once and instantly see your risk for 4 major diseases</p>
</div>
""", unsafe_allow_html=True)


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
    cp = st.slider("Chest Pain Type", 0, 3, 1)
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

predict_btn = st.button("Predict All 4 Diseases")


def make_gauge(title, value):
    color = "#ff4b4b" if value >= 50 else "#21c354"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 26}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 50], 'color': "#1e3d2f"},
                {'range': [50, 100], 'color': "#3d1e1e"}
            ],
            'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 50}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20),
                       paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig


if predict_btn:

    s = 1 if sex == "Male" else 0
    f = 1 if fbs == "Yes" else 0
    e = 1 if exang == "Yes" else 0
    h = 1 if hyp == "Yes" else 0
    hd = 1 if heart_disease == "Yes" else 0
    m = 1 if married == "Yes" else 0

    if smoke == "never smoked":
        sm = 0
    elif smoke == "formerly smoked":
        sm = 1
    elif smoke == "smokes":
        sm = 2
    else:
        sm = 3

    st.write("---")
    st.subheader("Results")

    model1 = joblib.load("diabetes_model.pkl")
    x1 = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prob1 = model1.predict_proba(x1)
    risk_diabetes = round(prob1[0][1] * 100, 2)

    model2 = joblib.load("heart_model.pkl")
    x2 = np.array([[age, s, cp, trestbps, chol, f, restecg, thalach, e, oldpeak, slope, ca, thal]])
    prob2 = model2.predict_proba(x2)
    risk_heart = round(prob2[0][0] * 100, 2)

    model3 = joblib.load("hypertension_model.pkl")
    smoke_map_hyp = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
    sm_hyp = smoke_map_hyp[smoke]
    x3 = np.array([[s, age, hd, sm_hyp, bmi, hba1c, glucose]])
    prob3 = model3.predict_proba(x3)
    risk_hypertension = round(prob3[0][1] * 100, 2)

    model4 = joblib.load("stroke_model.pkl")
    x4 = np.array([[s, age, h, hd, m, glucose, bmi, sm]])
    prob4 = model4.predict_proba(x4)
    risk_stroke = round(prob4[0][1] * 100, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_gauge("Diabetes", risk_diabetes), use_container_width=True)
    with col2:
        st.plotly_chart(make_gauge("Heart Disease", risk_heart), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_gauge("Hypertension", risk_hypertension), use_container_width=True)
    with col4:
        st.plotly_chart(make_gauge("Stroke", risk_stroke), use_container_width=True)

    labels = {
        "Diabetes": risk_diabetes,
        "Heart Disease": risk_heart,
        "Hypertension": risk_hypertension,
        "Stroke": risk_stroke,
    }
    for name, val in labels.items():
        if val >= 50:
            st.error(f"{name}: High Risk - {val}%")
        else:
            st.success(f"{name}: Low Risk - {val}%")

    st.write("---")
    st.subheader("Risk Comparison - All 4 Diseases")
    names = list(labels.keys())
    values = list(labels.values())
    colors = ["#ff4b4b" if v >= 50 else "#21c354" for v in values]

    bar_fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v}%" for v in values], textposition="outside"
    ))
    bar_fig.update_layout(
        yaxis_range=[0, 100], yaxis_title="Risk %",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}, height=400
    )
    st.plotly_chart(bar_fig, use_container_width=True)


st.write("---")
st.write("Disclaimer: this is just a risk estimate from ML models, not an actual medical diagnosis. please consult a doctor.")