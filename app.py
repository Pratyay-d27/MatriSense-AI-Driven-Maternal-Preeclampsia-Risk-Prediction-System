import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# =============================
# 1. LOAD MODELS
# =============================
@st.cache_resource
def load_maternal_model():
    dataset = pd.read_csv("Maternal_Health_Risk_Data_Set.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    lda = LDA(n_components=2)
    X_reduced = lda.fit_transform(X_scaled, y)

    clf = RandomForestClassifier(n_estimators=45, criterion='entropy', random_state=0)
    clf.fit(X_reduced, y)

    # Store training data in original scale for readable LIME explanation
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=dataset.columns[:-1],
        class_names=le.classes_,
        mode='classification'
    )
    return clf, sc, lda, le, lime_explainer, dataset

@st.cache_resource
def load_preeclampsia_model():
    df = pd.read_csv("dataframe.csv")
    df = df.drop(columns=["id"])
    X = df.drop(columns=["preeclampsia_onset"])
    y = df["preeclampsia_onset"]

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_scaled, y, batch_size=32, epochs=20, verbose=0)

    shap_explainer = shap.KernelExplainer(ann.predict, X_scaled[:50])
    return ann, sc, X.columns, shap_explainer

# =============================
# 2. UI
# =============================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #5a7fbf, #a8c8e9) !important; background-attachment: fixed !important; background-size: cover !important; font-family: "Bookman Old Style", serif; color: #f0f8ff; }
*{ font-weight: 700 !important; }
.block-container { background-color: transparent !important; }
[data-testid="stSidebar"] { background-color: rgba(44, 62, 80, 0.85) !important; color: #f0f8ff !important; }
[data-testid="stHeader"], [data-testid="stToolbar"] { background-color: transparent !important; }
h1, h2, h3, h4, h5, h6 { font-family: "Bookman Old Style", serif; color: #f0f8ff; text-shadow: 0 2px 6px rgba(0,0,0,0.3); }
.stButton>button { background: linear-gradient(90deg, #1abc9c, #16a085); color: #f0f8ff; font-weight: bold; border-radius: 50px; padding: 12px 45px; font-size: 17px; box-shadow: 0px 4px 12px rgba(0,0,0,0.35); transition: 0.4s; }
.stButton>button:hover { transform: scale(1.05); background: linear-gradient(90deg, #16a085, #1abc9c); }
.result-box { text-align: center; border-radius: 18px; padding: 25px; font-weight: 700; box-shadow: 0 6px 25px rgba(0,0,0,0.3); color: #f0f8ff; }
.risk-low {background: linear-gradient(135deg, #27ae60, #2ecc71);}
.risk-medium {background: linear-gradient(135deg, #f39c12, #f1c40f);}
.risk-high {background: linear-gradient(135deg, #e06666, #f1948a);}
.glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 18px; backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 25px; transition: transform 0.3s ease-in-out; }
.glass-card:hover { transform: translateY(-3px); }
</style>
""", unsafe_allow_html=True)

# =============================
# 3. SESSION STATE INIT
# =============================
if 'maternal_predicted' not in st.session_state:
    st.session_state['maternal_predicted'] = False
    st.session_state['risk_label'] = ''
    st.session_state['maternal_reduced'] = None

# =============================
# 4. UI LAYOUT
# =============================
st.set_page_config(page_title="Maternal & Preeclampsia AI", layout="wide", page_icon="🧬")
st.markdown("<h1 style='text-align:center;'>🩺 MatriSense: AI Powered Maternal & Preeclampsia Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Empowering clinical insights through intelligent diagnostics</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load models
clf, sc_maternal, lda, le_maternal, lime_explainer, dataset = load_maternal_model()
ann, sc_pre, pre_features, shap_explainer = load_preeclampsia_model()

with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Maternal Health Risk Evaluation")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=25)
        sbp = st.number_input("Systolic BP", min_value=80, max_value=200, value=130)
    with col2:
        dbp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
        bs = st.number_input("Blood Sugar", min_value=0.0, max_value=25.0, value=15.0)
    with col3:
        body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=110.0, value=98.0)
        hr = st.number_input("Heart Rate", min_value=50, max_value=150, value=86)

    maternal_features = np.array([[age, sbp, dbp, bs, body_temp, hr]])

    # =============================
    # Maternal Risk Prediction Button
    # =============================
    if st.button("Predict Maternal Risk ⚡"):
        maternal_scaled = sc_maternal.transform(maternal_features)
        maternal_reduced = lda.transform(maternal_scaled)
        risk_pred = clf.predict(maternal_reduced)
        risk_label = le_maternal.inverse_transform(risk_pred)[0]

        st.session_state['maternal_predicted'] = True
        st.session_state['risk_label'] = risk_label
        st.session_state['maternal_reduced'] = maternal_reduced

    # =============================
    # Display Maternal Prediction if already done
    # =============================
    if st.session_state['maternal_predicted']:
        risk_label = st.session_state['risk_label']
        maternal_reduced = st.session_state['maternal_reduced']

        css_class = 'risk-low'
        if risk_label == "medium risk": css_class = 'risk-medium'
        elif risk_label == "high risk": css_class = 'risk-high'

        st.markdown(f"<div class='result-box {css_class}'><h2>{risk_label.upper()}</h2></div>", unsafe_allow_html=True)
        st.progress(float(max(clf.predict_proba(maternal_reduced)[0])))

        #LIME explanation 
        exp = lime_explainer.explain_instance(
            maternal_features.flatten(),
            lambda x: clf.predict_proba(lda.transform(sc_maternal.transform(x))),
            num_features=5
        )
        st.markdown("### 🔍 Explanation of Predicted Risk")
        st.json(dict(exp.as_list()))

        # =============================
        # Preeclampsia Prediction Section
        # =============================
        if risk_label in ["medium risk", "high risk"]:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Preeclampsia Prediction: Since Risk is Not low")
            user_inputs = {}
            with st.expander("📋 Enter Preeclampsia Features"):
                for feature in pre_features:
                    user_inputs[feature] = st.text_input(f"{feature}", value="0")

            if st.button("Predict Preeclampsia 🧬"):
                input_values = np.array([float(user_inputs[f]) for f in pre_features]).reshape(1, -1)
                input_scaled = sc_pre.transform(input_values)
                pred = ann.predict(input_scaled)[0][0]

                preeclampsia_result = "High Risk of Preeclampsia" if pred > 0.5 else "No Significant Risk"
                box_color = "risk-high" if pred > 0.5 else "risk-low"

                st.markdown(f"<div class='result-box {box_color}'><h2>{preeclampsia_result}</h2></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
