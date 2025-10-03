import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Employee-Attrition - Employee-Attrition.csv")

st.set_page_config(page_title="Dashboard", layout='wide')

st.sidebar.title("Employee Attrition Analysis")
page = st.sidebar.radio("", ["Home", "Predict Employee Attrition"])

if page == "Home":
    st.title("Dashboard Home")
    st.header("Employee Insights Dashboard")

    col1, col2, col3 = st.columns(3)

    total_employees = len(df)
    attrition_yes = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_yes / total_employees) * 100
    performance_rate = df["PerformanceRating"].mean()

    with col1:
        st.subheader("High-Risk Employees")
        st.metric("Attrition", f"{attrition_rate:.2f}%")
        high_risk_employees = df[df["Attrition"] == "Yes"]
        st.dataframe(high_risk_employees[["EmployeeNumber", "Attrition"]])

    with col2:
        st.subheader("High Job Satisfaction")
        high_satisfaction = df[df['JobSatisfaction'] >= 4]
        st.dataframe(high_satisfaction[['EmployeeNumber', 'JobSatisfaction']])

    with col3:
        st.subheader("High Performance Score")
        if "PerformanceRating" in df.columns:
            high_perf = df.sort_values('PerformanceRating', ascending=False).head(5)
            st.dataframe(high_perf[['EmployeeNumber', 'PerformanceRating']])

    st.write("### Employee Data Table")
    st.dataframe(df)


elif page == "Predict Employee Attrition":
    st.title("Predict Employee Attrition")

    # Load trained objects
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_order = joblib.load("feature_order.pkl")
    gbc = joblib.load("gbc_model.pkl")   # Gradient Boosting model
    scaler = joblib.load("scaler_gb.pkl")  # Scaler used during trainin

    categorical_cols_encoder = encoder.feature_names_in_.tolist()

    frequent_values = {
        'BusinessTravel': 'Travel_Rarely',
        'OverTime': 'No',
        'Department': 'Sales',
        'EducationField': 'Life Sciences',
        'Gender': 'Male',
        'JobRole': 'Sales Executive',
        'TenureCategory': 'Experienced',
        'MaritalStatus': 'Married'
    }

    user_input = {}

    for col in categorical_cols_encoder:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
        else:
            options = [frequent_values.get(col, "missing")]
        default_value = frequent_values.get(col, options[0])
        user_input[col] = st.selectbox(col, options, index=options.index(default_value) if default_value in options else 0)

    numeric_cols = [col for col in feature_order if col not in categorical_cols_encoder]
    for col in numeric_cols:
        default_val = float(df[col].mean()) if col in df.columns else 0.0
        user_input[col] = st.number_input(col, value=default_val)
  
    
    df_input = pd.DataFrame([user_input])

    for col in categorical_cols_encoder:
        if col not in df_input.columns:
            df_input[col] = frequent_values.get(col, "missing")

    df_categorical = df_input[categorical_cols_encoder]
    df_categorical_encoded = encoder.transform(df_categorical)
    df_categorical_encoded = pd.DataFrame(df_categorical_encoded, columns=encoder.get_feature_names_out())

    df_numerical = df_input[numeric_cols]
    df_final = pd.concat([df_numerical.reset_index(drop=True), df_categorical_encoded.reset_index(drop=True)], axis=1)

    df_final = df_final.reindex(columns=feature_order, fill_value=0)

    df_scaled = scaler.transform(df_final)
    df_final = df_final.reindex(columns=feature_order, fill_value=0)

    df_scaled = scaler.transform(df_final)

    # Display raw and scaled data
    st.write("Original Input Data:", user_input)
    st.write("### Scaled Input Data")
    st.dataframe(pd.DataFrame([df_scaled[0]], columns=df_final.columns))

    st.write("### Predict Employee Attrition")

# Threshold slider
threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.35, 0.01)

if st.button("Predict Attrition"):
    probability = gbc.predict_proba(df_scaled)[0][1]
    prediction = 1 if probability > threshold else 0

    if prediction == 1:
        st.error(f"🚨 Prediction: Attrition\nProbability")
    else:
        st.success(f"✅ Prediction: No Attrition\nProbability")
