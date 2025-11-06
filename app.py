import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# --- Load model ---
MODEL_PATH = "heartdisease.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter patient features below. The model returns a binary prediction and (if available) the predicted probability."
)

# --- Input form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=54, step=1)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[
                "0 - Typical Angina",
                "1 - Atypical Angina",
                "2 - Non-anginal Pain",
                "3 - Asymptomatic",
                "numeric (0..3)"
            ],
            index=0,
        )
        trestbps = st.number_input("Resting Blood Pressure (trestbps, mm Hg)", min_value=50, max_value=250, value=140, step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=230, step=1)

    with col2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=["No (0)", "Yes (1)"])
        restecg = st.selectbox(
            "Resting electrocardiographic results (restecg)",
            options=[
                "0 - Normal",
                "1 - Having ST-T wave abnormality",
                "2 - Showing probable or definite left ventricular hypertrophy",
                "numeric (0..2)"
            ],
            index=0,
        )
        thalach = st.number_input("Max heart rate achieved (thalach)", min_value=50, max_value=250, value=150, step=1)
        exang = st.selectbox("Exercise induced angina (exang)", options=["No (0)", "Yes (1)"])

    col3, col4 = st.columns(2)
    with col3:
        oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")
        slope = st.selectbox(
            "Slope of the peak exercise ST segment (slope)",
            options=[
                "0 - Upsloping",
                "1 - Flat",
                "2 - Downsloping",
                "numeric (0..2)"
            ],
            index=0,
        )
    with col4:
        ca = st.number_input("Number of major vessels colored by fluoroscopy (ca)", min_value=0, max_value=4, value=0, step=1)
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[
                "1 - Normal",
                "2 - Fixed defect",
                "3 - Reversible defect",
                "numeric (1..3)"
            ],
            index=0,
        )

    submitted = st.form_submit_button("Predict")

# --- Preprocess inputs to numeric values matching the model's expected order ---
def parse_choice(value):
    # If the user selected one of the descriptive strings, extract the numeric at start.
    if isinstance(value, str):
        try:
            return int(value.split()[0])
        except Exception:
            return value
    return value

# map sex to numeric commonly used: Male=1, Female=0
sex_val = 1 if sex.lower().startswith("m") else 0
cp_val = parse_choice(cp)
fbs_val = 1 if str(fbs).lower().startswith("y") or "1" in str(fbs) else 0
restecg_val = parse_choice(restecg)
exang_val = 1 if str(exang).lower().startswith("y") or "1" in str(exang) else 0
slope_val = parse_choice(slope)
thal_val = parse_choice(thal)

# Build single-row DataFrame in the order the user supplied:
feature_order = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeack','slope','ca','thal']

# Note: user provided column name "oldpeack" — common dataset uses "oldpeak". Keep the same column name order.
row = {
    'age': age,
    'sex': sex_val,
    'cp': cp_val,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_val,
    'restecg': restecg_val,
    'thalach': thalach,
    'exang': exang_val,
    # Keep same spelling user gave in the column list:
    'oldpeack': oldpeak,
    'slope': slope_val,
    'ca': ca,
    'thal': thal_val
}

input_df = pd.DataFrame([row], columns=feature_order)

if submitted:
    st.subheader("Input summary")
    st.dataframe(input_df.T.rename(columns={0: "value"}))

    # Predict
    try:
        # Some models expect numpy array, some expect DataFrame — try both
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[:, 1]
        else:
            proba = None

        pred = model.predict(input_df)
        pred_label = int(pred[0]) if isinstance(pred, (list, np.ndarray)) else int(pred)

        if proba is not None:
            prob_val = float(proba[0])
            st.success(f"Prediction: **{pred_label}**  (1 = disease, 0 = no disease)")
            st.info(f"Predicted probability of heart disease: **{prob_val*100:.2f}%**")
        else:
            st.success(f"Prediction: **{pred_label}**  (1 = disease, 0 = no disease)")
            st.warning("Model does not provide probabilities (no `predict_proba`).")

        # Short plain-language interpretation
        if pred_label == 1:
            st.write("Interpretation: The model indicates a **higher risk** of heart disease. Please consult a clinician for diagnostic testing and clinical correlation.")
        else:
            st.write("Interpretation: The model indicates a **lower risk** of heart disease. This is not a medical diagnosis. If there are symptoms or concerns, consult a clinician.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.markdown("---")
st.caption(
    """
    Notes:
    - This app expects the model `heartdisease.pkl` to accept features in this exact column order:
      `['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeack','slope','ca','thal']`.
    - The input widgets use common encodings (sex: Male=1, Female=0; fbs/exang: 1 = Yes, 0 = No). 
    - If your model expects different encodings (e.g. one-hot, different integer mapping), adjust the preprocessing section accordingly.
    - This tool is for demonstration only and not a substitute for professional medical advice.
    """
)
