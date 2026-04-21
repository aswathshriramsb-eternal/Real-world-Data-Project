# =========================
# ALL IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("❤️ Heart Disease Prediction System")
st.write("If you see this, app is running ✅")

# =========================
# LOAD FILES (SAFE)
# =========================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("cleaned_data.csv")

    st.success("✅ Files Loaded Successfully")

except Exception as e:
    st.error("❌ Error loading files")
    st.code(str(e))
    st.stop()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Patient Input")

features = df.columns[:-1]

# Create default values
input_dict = {}
for col in features:
    input_dict[col] = df[col].median()

# Override important ones
if "age" in input_dict:
    input_dict["age"] = st.sidebar.slider("Age", 20, 80, int(df["age"].median()))

if "sex" in input_dict:
    input_dict["sex"] = st.sidebar.selectbox("Sex", [0, 1])

if "cp" in input_dict:
    input_dict["cp"] = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])

if "trestbps" in input_dict:
    input_dict["trestbps"] = st.sidebar.slider("Resting BP", 90, 200, int(df["trestbps"].median()))

if "chol" in input_dict:
    input_dict["chol"] = st.sidebar.slider("Cholesterol", 100, 400, int(df["chol"].median()))

if "fbs" in input_dict:
    input_dict["fbs"] = st.sidebar.selectbox("Fasting Blood Sugar", [0,1])

if "thalach" in input_dict:
    input_dict["thalach"] = st.sidebar.slider("Max Heart Rate", 70, 210, int(df["thalach"].median()))

# Convert input to dataframe
input_df = pd.DataFrame([input_dict])

# =========================
# SCALE INPUT
# =========================
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error("Scaling Error")
    st.code(str(e))
    st.stop()

# =========================
# PREDICTION
# =========================
st.subheader("Prediction")

if st.button("Predict"):
    try:
        pred = model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error("Prediction Error")
        st.code(str(e))

# =========================
# VISUALIZATION
# =========================
st.subheader("Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Target Distribution")
    st.bar_chart(df.iloc[:, -1].value_counts())

with col2:
    if "age" in df.columns:
        st.write("Age Distribution")
        st.line_chart(df["age"])

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")

try:
    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))

except:
    st.warning("Feature importance not available")

# =========================
# DATA VIEW
# =========================
if st.checkbox("Show Dataset"):
    st.write(df.head())