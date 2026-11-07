"""AI-Based Personalized Health Risk Assessment - full website (Streamlit).
Pages: Home | Risk Assessment | Model Performance | Explainability | About
Team: Anant Raj, Aadesh Agarwal, Abhinav Sahai Kamthan, Divyansh Saxena  |  Guide: Ms. Shruti Sharma
"""
import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import shap
    HAVE_SHAP = True
except Exception:          # the app still works without the shap package (falls back to what-if factors)
    HAVE_SHAP = False

st.set_page_config(page_title="AI Health Risk Assessment", page_icon="🩺", layout="wide")

GITHUB = "https://github.com/Anant6125/AI_Based_Personalized_Health_Risk_Asseament_System"
MODEL_FILES = {"diabetes": "diabetes_model.pkl", "heart": "heart_model.pkl",
               "hypertension": "hypertension_model.pkl", "stroke": "stroke_model.pkl"}

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.hero { background: linear-gradient(90deg, #11998e, #38ef7d); padding: 40px; border-radius: 16px;
        text-align: center; margin-bottom: 26px; box-shadow: 0 8px 30px rgba(0,0,0,0.35); }
.hero h1 { color: white; margin: 0; font-size: 38px; }
.hero p { color: #eafff0; margin-top: 10px; font-size: 18px; }
.badge { display:inline-block; background: rgba(255,255,255,0.18); color:#fff; padding:4px 12px; border-radius:20px;
         font-size: 13px; margin: 10px 4px 0 4px; }
.card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
        padding: 18px 20px; margin-bottom: 14px; }
.card h4 { margin: 0 0 6px 0; color: #38ef7d; }
.card p { margin: 0; color: #e8f1f2; font-size: 15px; }
.stat { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; text-align: center; }
.stat .num { font-size: 30px; font-weight: bold; color: #38ef7d; }
.stat .lab { font-size: 14px; color: #cfe3e6; }
.tile { border-radius: 14px; padding: 18px; text-align: center; border: 1px solid rgba(255,255,255,0.12); }
.tile .nm { font-size: 15px; color: #dcecee; }
.tile .pc { font-size: 36px; font-weight: bold; margin: 4px 0; }
.tile .lv { display:inline-block; padding: 3px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; color: white; }
.flow { display:flex; align-items:stretch; gap: 8px; flex-wrap: wrap; margin: 8px 0 18px 0; }
.flow .box { flex: 1; min-width: 150px; background: rgba(255,255,255,0.07); border: 1px solid rgba(56,239,125,0.35);
             border-radius: 12px; padding: 14px; text-align:center; color:#eaf6f7; font-size: 14px; }
.flow .box b { color:#38ef7d; display:block; margin-bottom: 4px; font-size: 15px; }
.flow .arr { align-self:center; color:#38ef7d; font-size: 26px; font-weight: bold; }
.chip { display:inline-block; background: rgba(56,239,125,0.12); border: 1px solid rgba(56,239,125,0.4); color:#bff5d3;
        padding: 5px 14px; border-radius: 20px; font-size: 14px; margin: 4px; }
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


def tile(name, value):
    high = value >= 50
    color = "#ff4b4b" if high else "#21c354"
    bg = "rgba(255,75,75,0.12)" if high else "rgba(33,195,84,0.12)"
    st.markdown(f'<div class="tile" style="background:{bg}"><div class="nm">{name}</div>'
                f'<div class="pc" style="color:{color}">{value:.1f}%</div>'
                f'<span class="lv" style="background:{color}">{"HIGH RISK" if high else "LOW RISK"}</span></div>',
                unsafe_allow_html=True)


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
        models["diabetes"].predict_proba(np.zeros((1, 8)))
        models["heart"].predict_proba(np.zeros((1, 13)))
        models["hypertension"].predict_proba(np.zeros((1, 7)))
        models["stroke"].predict_proba(np.zeros((1, 8)))
        return models
    except Exception:
        return train_models()


@st.cache_resource(show_spinner="Preparing SHAP explainers...")
def load_explainers():
    models = load_models()
    if not HAVE_SHAP:
        return None
    try:
        return {k: shap.TreeExplainer(m) for k, m in models.items()}
    except Exception:
        return None


# "Typical person" reference values, used only as a fallback what-if explanation when SHAP is unavailable.
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
SMOKE_NAMES = {0: "never smoked", 1: "formerly smoked", 2: "smokes", 3: "unknown"}


def fmt(name, v):
    if name == "Sex":
        return "Male" if v == 1 else "Female"
    if name in ("Fasting sugar", "Exercise angina", "Heart disease", "Hypertension", "Ever married"):
        return "Yes" if v == 1 else "No"
    if name == "Smoking":
        return SMOKE_NAMES.get(int(v), str(v))
    return f"{v:g}"


def factor_impact(model, x, ref, class_index):
    base = model.predict_proba(np.array([x]))[0][class_index]
    out = []
    for i in range(len(x)):
        x2 = list(x)
        x2[i] = ref[i]
        out.append((base - model.predict_proba(np.array([x2]))[0][class_index]) * 100)
    return out


def shap_contrib(explainer, x, class_index):
    """Real SHAP values (percentage points of risk) for one person."""
    sv = explainer.shap_values(np.array([x]))
    if isinstance(sv, list):
        arr = sv[class_index][0]
    elif getattr(sv, "ndim", 2) == 3:
        arr = sv[0, :, class_index]
    else:
        arr = sv[0]
    return list(np.asarray(arr) * 100)


def explanation_chart(title, names, values, impacts, top=6):
    labels = [f"{n} = {fmt(n, v)}" for n, v in zip(names, values)]
    order = sorted(range(len(names)), key=lambda i: abs(impacts[i]), reverse=True)[:top][::-1]
    fig = go.Figure(go.Bar(x=[impacts[i] for i in order], y=[labels[i] for i in order], orientation="h",
                           marker_color=["#ff4b4b" if impacts[i] > 0 else "#21c354" for i in order],
                           text=[f"{impacts[i]:+.1f}" for i in order], textposition="outside", cliponaxis=False))
    shown = [impacts[i] for i in order]
    lo, hi = min(0, min(shown)), max(0, max(shown))
    pad = (hi - lo) * 0.18 or 1
    fig.update_layout(title={'text': title, 'font': {'size': 16}}, height=330, margin=dict(l=20, r=40, t=50, b=30),
                      xaxis_range=[lo - pad, hi + pad], xaxis_title="Effect on risk (percentage points)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig


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


# ------------------------------------------------------------------ sample profiles
DEFAULTS = dict(age=30, sex="Male", bmi=25.0, glucose=120, preg=1, bp=70, skin=20, insulin=80, dpf=0.5, cp=1, trestbps=120,
                chol=200, fbs="Yes", restecg=1, thalach=150, exang="Yes", oldpeak=1.0, slope=1, ca=0, thal=2, hba1c=5.5,
                hyp="Yes", hd="Yes", married="Yes", smoke="never smoked")
HEALTHY = dict(DEFAULTS, age=25, sex="Female", cp=0, chol=180, fbs="No", thalach=180, exang="No", oldpeak=0.0,
               hyp="No", hd="No", married="No")
RISKY = dict(DEFAULTS, age=60, sex="Male", glucose=220, cp=3, chol=300, fbs="No", thalach=110, exang="Yes", oldpeak=3.0,
             ca=3, hyp="Yes", hd="Yes", married="Yes", smoke="smokes")


def set_profile(profile):
    for k, v in profile.items():
        st.session_state[k] = v
    st.session_state["_form"] = dict(profile)
    st.session_state.pop("result", None)


# ------------------------------------------------------------------ pages
def page_home():
    st.markdown("""<div class="hero"><h1>🩺 AI Based Personalized Health Risk Assessment</h1>
    <p>One form. One click. Your estimated risk for diabetes, heart disease, hypertension and stroke, with an explanation.</p>
    <span class="badge">4 Random Forest models</span><span class="badge">SHAP explainability</span>
    <span class="badge">Imbalance-aware evaluation</span><span class="badge">Streamlit web app</span></div>""",
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.button("Start the assessment", on_click=go_to, args=("Risk Assessment",), key="cta1")
    with b2:
        st.button("See how well the models work", on_click=go_to, args=("Model Performance",), key="cta2")
    st.write("")
    c = st.columns(4)
    for col, (t, tx) in zip(c, [("Diabetes", "Based on glucose, BMI, age, insulin and family history."),
                                ("Heart Disease", "Based on 13 clinical readings such as cholesterol, heart rate and ECG."),
                                ("Hypertension", "Based on age, BMI, HbA1c, glucose and smoking history."),
                                ("Stroke", "Based on age, glucose, BMI, smoking and existing conditions.")]):
        with col:
            card(t, tx)
    st.subheader("How it works")
    st.markdown("""<div class="flow">
    <div class="box"><b>1. Input form</b>One short form for all four diseases</div><div class="arr">&#10142;</div>
    <div class="box"><b>2. Encoding</b>Sex, smoking and marital status mapped like in training</div><div class="arr">&#10142;</div>
    <div class="box"><b>3. Four models</b>A separate Random Forest for each disease</div><div class="arr">&#10142;</div>
    <div class="box"><b>4. Risk score</b>Probability in %, High if 50% or more</div><div class="arr">&#10142;</div>
    <div class="box"><b>5. Explanation</b>SHAP shows what pushed your risk up or down</div></div>""", unsafe_allow_html=True)
    st.subheader("Model snapshot")
    s = st.columns(4)
    for col, (n, l) in zip(s, [("4", "diseases in one tool"), ("81.97%", "heart disease accuracy"),
                               ("0.906", "best ROC-AUC (heart)"), ("~2 s", "time per prediction")]):
        with col:
            stat(n, l)
    st.subheader("Built with")
    st.markdown("".join(f'<span class="chip">{t}</span>' for t in
                        ["Python", "scikit-learn", "Random Forest", "SHAP", "XGBoost (comparison)", "SMOTE", "Pandas", "NumPy", "Plotly", "Streamlit"]),
                unsafe_allow_html=True)
    st.write("")
    st.info("This tool gives a risk estimate for awareness only. It is not a medical diagnosis. Please consult a doctor.")


def build_report(inputs, risks):
    lines = ["AI BASED PERSONALIZED HEALTH RISK ASSESSMENT - REPORT",
             f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", "", "RESULTS"]
    for k, v in risks.items():
        lines.append(f"- {k}: {'HIGH' if v >= 50 else 'LOW'} risk ({v}%)")
    lines += ["", "INPUTS"] + [f"- {k}: {v}" for k, v in inputs.items()]
    lines += ["", "Note: this is a risk estimate from machine learning models for awareness only. It is not a medical "
              "diagnosis. Please consult a doctor.", "Project ID 26P_3CSE022 | IILM University, Greater Noida"]
    return "\n".join(lines)


def page_assess():
    st.markdown("""<div class="hero"><h1>Risk Assessment</h1><p>Fill the form once and see your risk for 4 major diseases</p></div>""",
                unsafe_allow_html=True)
    # keep the form values when the user visits another page and comes back
    for k, v in {**DEFAULTS, **st.session_state.get("_form", {})}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    q1, q2, q3 = st.columns([1, 1, 2])
    with q1:
        st.button("Try a healthy sample", on_click=set_profile, args=(HEALTHY,), key="smp1")
    with q2:
        st.button("Try a high-risk sample", on_click=set_profile, args=(RISKY,), key="smp2")
    with q3:
        st.caption("Quick demo: fill the whole form with a sample profile, then press Predict.")

    with st.container(border=True):
        st.header("Basic Info")
        age = st.number_input("Age", 1, 120, key="age")
        sex = st.radio("Sex", ["Male", "Female"], key="sex")
        bmi = st.number_input("BMI", 0.0, 70.0, key="bmi")
        glucose = st.number_input("Glucose Level", 0, 300, key="glucose")
    with st.container(border=True):
        st.header("Diabetes Info")
        preg = st.number_input("Pregnancies (put 0 if male)", 0, 20, key="preg")
        bp = st.number_input("Blood Pressure", 0, 200, key="bp")
        skin = st.number_input("Skin Thickness", 0, 100, key="skin")
        insulin = st.number_input("Insulin", 0, 900, key="insulin")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, key="dpf")
    with st.container(border=True):
        st.header("Heart Info")
        cp = st.slider("Chest Pain Type", 0, 3, key="cp", help="Coded 0 to 3 as in the training dataset (not a severity scale).")
        trestbps = st.number_input("Resting BP", 0, 250, key="trestbps")
        chol = st.number_input("Cholesterol", 0, 600, key="chol")
        fbs = st.radio("Fasting Blood Sugar > 120?", ["Yes", "No"], key="fbs")
        restecg = st.slider("Rest ECG", 0, 2, key="restecg")
        thalach = st.number_input("Max Heart Rate", 0, 250, key="thalach")
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"], key="exang")
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, key="oldpeak")
        slope = st.slider("Slope", 0, 2, key="slope")
        ca = st.slider("Major Vessels", 0, 3, key="ca")
        thal = st.slider("Thal", 0, 3, key="thal")
    with st.container(border=True):
        st.header("Hypertension Info")
        hba1c = st.number_input("HbA1c Level", 3.0, 15.0, key="hba1c")
    with st.container(border=True):
        st.header("Extra Info for Stroke")
        hyp = st.radio("Hypertension?", ["Yes", "No"], key="hyp")
        heart_disease = st.radio("Heart Disease?", ["Yes", "No"], key="hd")
        married = st.radio("Ever Married?", ["Yes", "No"], key="married")
        smoke = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"], key="smoke")

    st.session_state["_form"] = {k: st.session_state[k] for k in DEFAULTS}

    if st.button("Predict All 4 Diseases", key="predict"):
        models = load_models()
        s = 1 if sex == "Male" else 0
        f = 1 if fbs == "Yes" else 0
        e = 1 if exang == "Yes" else 0
        h = 1 if hyp == "Yes" else 0
        hd = 1 if heart_disease == "Yes" else 0
        m = 1 if married == "Yes" else 0
        sm = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}[smoke]
        xs = {"diabetes": [preg, glucose, bp, skin, insulin, bmi, dpf, age],
              "heart": [age, s, cp, trestbps, chol, f, restecg, thalach, e, oldpeak, slope, ca, thal],
              "hypertension": [s, age, hd, sm, bmi, hba1c, glucose],
              "stroke": [s, age, h, hd, m, glucose, bmi, sm]}
        cls = {"diabetes": 1, "heart": 0, "hypertension": 1, "stroke": 1}   # heart: class 0 = heart disease in this dataset
        titles = {"diabetes": "Diabetes", "heart": "Heart Disease", "hypertension": "Hypertension", "stroke": "Stroke"}
        risks = {titles[k]: round(models[k].predict_proba(np.array([xs[k]]))[0][cls[k]] * 100, 2) for k in xs}
        explainers = load_explainers()
        method = "SHAP"
        contrib = {}
        for k in xs:
            try:
                if explainers is None:
                    raise RuntimeError("no shap")
                contrib[titles[k]] = shap_contrib(explainers[k], xs[k], cls[k])
            except Exception:
                method = "what-if"
                contrib[titles[k]] = factor_impact(models[k], xs[k], REF[k], cls[k])
        inputs = {"Age": age, "Sex": sex, "BMI": bmi, "Glucose": glucose, "Pregnancies": preg, "Blood pressure": bp,
                  "Skin thickness": skin, "Insulin": insulin, "Pedigree function": dpf, "Chest pain type": cp,
                  "Resting BP": trestbps, "Cholesterol": chol, "Fasting sugar > 120": fbs, "Rest ECG": restecg,
                  "Max heart rate": thalach, "Exercise angina": exang, "Oldpeak": oldpeak, "Slope": slope,
                  "Major vessels": ca, "Thal": thal, "HbA1c": hba1c, "Hypertension": hyp, "Heart disease": heart_disease,
                  "Ever married": married, "Smoking": smoke}
        st.session_state["result"] = {"risks": risks, "contrib": contrib, "xs": xs, "method": method,
                                      "titles": titles, "report": build_report(inputs, risks)}

    res = st.session_state.get("result")
    if res:
        risks = res["risks"]
        st.write("---")
        st.subheader("Your results")
        tcols = st.columns(4)
        for col, (name, val) in zip(tcols, risks.items()):
            with col:
                tile(name, val)
        if any(v >= 50 for v in risks.values()):
            st.warning("One or more estimates are High. This is not a diagnosis, but it is a good reason to talk to a doctor and get a proper check-up.")
        else:
            st.success("All four estimates are Low. Keep up healthy habits and regular check-ups.")
        tab1, tab2, tab3, tab4 = st.tabs(["Gauges", "Comparison", "Why this result? (SHAP)", "Overall SHAP plots"])
        with tab1:
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
        with tab2:
            vals = list(risks.values())
            bar = go.Figure(go.Bar(x=list(risks.keys()), y=vals, marker_color=["#ff4b4b" if v >= 50 else "#21c354" for v in vals],
                                   text=[f"{v}%" for v in vals], textposition="outside"))
            bar.update_layout(yaxis_range=[0, 100], yaxis_title="Risk %", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=420)
            show_chart(bar, key="bar")
        with tab3:
            if res["method"] == "SHAP":
                st.caption("Real SHAP values for your own inputs (TreeSHAP on the Random Forest models). Red bars push your "
                           "risk up, green bars push it down, in percentage points. The bars add up to the difference between "
                           "your result and the model's average output.")
            else:
                st.caption("SHAP is not available on this server, so a what-if explanation is shown: how much the risk changes "
                           "when one input is replaced by the value of a typical person.")
            key_of = {v: k for k, v in res["titles"].items()}
            k1, k2 = st.columns(2)
            for col, names_ in zip((k1, k2), (("Diabetes", "Hypertension"), ("Heart Disease", "Stroke"))):
                with col:
                    for nm in names_:
                        k = key_of[nm]
                        show_chart(explanation_chart(nm, NAMES[k], res["xs"][k], res["contrib"][nm]), key=f"x_{k}")
        with tab4:
            files = {"Diabetes": "diabetes", "Heart Disease": "heart", "Hypertension": "hypertension", "Stroke": "stroke"}
            if all(os.path.exists(f"shap_images/{v}.png") for v in files.values()):
                st.caption("SHAP summary plots computed on the test data: each dot is one test record, the colour is the input "
                           "value (red = high) and the position is its effect on the prediction.")
                sc1, sc2 = st.columns(2)
                for i, (label, key) in enumerate(files.items()):
                    with (sc1 if i % 2 == 0 else sc2):
                        show_image(f"shap_images/{key}.png", caption=label)
            else:
                st.info("SHAP images were not found in the shap_images folder.")
        st.download_button("Download my report (.txt)", res["report"], file_name="health_risk_report.txt", key="dl")

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
    s = st.columns(4)
    for col, (n, l) in zip(s, [("75.97%", "diabetes accuracy"), ("81.97%", "heart disease accuracy"),
                               ("65.52%", "hypertension accuracy"), ("86.86%", "stroke accuracy")]):
        with col:
            stat(n, l)
    st.write("")
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
    choice = st.selectbox("Disease", ["All"] + list(PERF["Disease"]), key="cmp_choice")
    sub = COMPARE if choice == "All" else COMPARE[COMPARE["Disease"] == choice]
    show_table(sub)
    metric = st.radio("Compare methods by", ["Recall", "F1", "ROC-AUC", "Accuracy (%)"], horizontal=True, key="cmp_metric")
    cf = go.Figure()
    for d_name, g in sub.groupby("Disease", sort=False):
        cf.add_bar(name=d_name, x=g["Method"], y=g[metric])
    cf.update_layout(barmode="group", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font={'color': "white"}, yaxis_title=metric)
    show_chart(cf, key="cmpchart")
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
    st.info("These plots show the models overall. For your own inputs, open the Risk Assessment page and use the "
            "'Why this result? (SHAP)' tab after pressing Predict.")


def page_about():
    st.markdown("""<div class="hero"><h1>About the Project</h1><p>B.Tech Major Project, School of Computer Science and Engineering, IILM University, Greater Noida</p></div>""", unsafe_allow_html=True)
    st.subheader("Project")
    st.write("**AI-Based Personalized Health Risk Assessment System** (Project ID 26P_3CSE022). Diabetes, heart disease, hypertension and "
             "stroke often develop silently and are found late. Our tool lets a user enter basic health details once and see an early "
             "risk estimate for all four, with an explanation, as an aid to awareness.")
    st.subheader("Team")
    show_table(pd.DataFrame({"Name": ["Anant Raj", "Aadesh Agarwal", "Abhinav Sahai Kamthan", "Divyansh Saxena"],
                             "Roll No.": ["2410031035", "2410030324", "2410031548", "2410030172"]}))
    st.write("**Guide:** Ms. Shruti Sharma")
    st.subheader("System architecture")
    st.markdown("""<div class="flow">
    <div class="box"><b>Public datasets</b>Pima Diabetes, UCI Heart, hypertension, stroke</div><div class="arr">&#10142;</div>
    <div class="box"><b>Preprocessing</b>Encoding, cleaning, 80/20 split</div><div class="arr">&#10142;</div>
    <div class="box"><b>Training</b>Random Forest per disease, saved with joblib</div><div class="arr">&#10142;</div>
    <div class="box"><b>Evaluation</b>Accuracy, precision, recall, ROC-AUC, SHAP</div><div class="arr">&#10142;</div>
    <div class="box"><b>Web app</b>Streamlit form, gauges, explanations</div></div>""", unsafe_allow_html=True)
    st.subheader("What we did")
    with st.expander("Project journey (Aug to Nov 2026)", expanded=False):
        st.write("- **Phase I:** literature survey, objectives, feasibility, first four models and prototype.\n"
                 "- **Phase II:** planning and supervisor review.\n"
                 "- **Phase III:** single Streamlit form, gauges and charts, testing, metrics, imbalance handling, model comparison, SHAP.\n"
                 "- **Next:** research paper, final report and final presentation.")
    with st.expander("Key lessons"):
        st.write("- The first hypertension model used a proxy label and showed high accuracy but almost zero recall; it was found by testing "
                 "with known profiles and replaced with a model trained on a real hypertension label.\n"
                 "- In the heart dataset, label 0 means disease; SHAP plots helped us notice this and the metrics are reported for the disease class.\n"
                 "- Accuracy alone is misleading on imbalanced data such as stroke (4.3% positive).")
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
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech Major Project 26P_3CSE022\n\nIILM University, Greater Noida")
st.sidebar.caption("Awareness tool only. Not a medical diagnosis.")
PAGES[st.session_state["page"]]()
