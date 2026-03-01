import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from database import init_db, save_diabetes_prediction, save_disease_prediction, get_diabetes_history, get_disease_history

st.set_page_config(page_title="HealthGuard AI", page_icon="🩺", layout="wide")
init_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Prediction Type:", ["Diabetes Prediction", "Normal Disease Check"])

st.sidebar.markdown("---")
st.sidebar.title("About")

if page == "Diabetes Prediction":
    model = joblib.load("./diabetes model/diabetes_model.pkl")
    scaler = joblib.load("./diabetes model/scaler.pkl")
    
    st.sidebar.info("This model uses Logistic Regression trained on PIMA dataset.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use")
    st.sidebar.markdown("1. Enter patient details\n2. Click Predict\n3. View results")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Example")
    st.sidebar.markdown("**High Risk Case:**\n- Pregnancies: 6\n- Glucose: 148 mg/dL\n- Blood Pressure: 72 mm Hg\n- Skin Thickness: 35 mm\n- Insulin: 0 μU/mL\n- BMI: 33.6 kg/m²\n- DPF: 0.627\n- Age: 50 years")

    st.title("HealthGuard AI - Diabetes Prediction")
    st.markdown("### Enter patient details to predict diabetes risk")
    st.markdown("💡 How to use: Enter patient details → Click Predict button → View results")
    
    if st.button("View Diabetes History"):
        st.session_state['show_diabetes_history'] = not st.session_state.get('show_diabetes_history', False)
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.get('show_diabetes_history', False):
        st.subheader("Diabetes Prediction History")
        history = get_diabetes_history()
        if not history.empty:
            history['Result'] = history['prediction'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
            st.dataframe(history[['timestamp', 'age', 'glucose', 'bmi', 'Result', 'probability']], use_container_width=True)
        else:
            st.info("No prediction history available.")
        st.markdown("---")

    # Input fields in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120, help="Plasma glucose concentration")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70, help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")

    with col2:
        st.subheader("Medical Metrics")
        insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80, help="2-Hour serum insulin")
        bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, help="Body Mass Index")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, help="Genetic influence score")
        age = st.number_input("Age (years)", 1, 120, 30, help="Age in years")

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button(" Predict Diabetes Risk", use_container_width=True, type="primary")

    if predict_btn:
        user_input_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        input_df = pd.DataFrame([user_input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        save_diabetes_prediction(user_input_dict, int(prediction[0]), float(probability[0][1]))

        st.markdown("---")
        st.subheader(" Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction[0] == 1:
                st.error("### High Risk of Diabetes")
            else:
                st.success("###  Low Risk of Diabetes")
        
        with result_col2:
            st.metric("Risk Probability", f"{probability[0][1]*100:.2f}%")
        
        st.progress(probability[0][1])
        
        st.markdown("---")
        st.subheader("Patient Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_glucose = go.Figure(go.Indicator(
                mode="gauge+number",
                value=glucose,
                title={'text': "Glucose Level (mg/dL)"},
                gauge={'axis': {'range': [0, 300]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 140], 'color': "lightgreen"},
                           {'range': [140, 200], 'color': "yellow"},
                           {'range': [200, 300], 'color': "red"}]}))
            fig_glucose.update_layout(height=250)
            st.plotly_chart(fig_glucose, use_container_width=True)
            
            fig_bmi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi,
                title={'text': "BMI (kg/m²)"},
                gauge={'axis': {'range': [0, 50]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 18.5], 'color': "lightblue"},
                           {'range': [18.5, 25], 'color': "lightgreen"},
                           {'range': [25, 30], 'color': "yellow"},
                           {'range': [30, 50], 'color': "red"}]}))
            fig_bmi.update_layout(height=250)
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        with col2:
            risk_data = pd.DataFrame({
                'Factor': ['Glucose', 'BMI', 'Blood Pressure', 'Age', 'Insulin'],
                'Value': [glucose/3, bmi*4, blood_pressure, age/1.2, insulin/9]
            })
            fig_bar = px.bar(risk_data, x='Factor', y='Value', 
                            title='Risk Factors Comparison',
                            color='Value', color_continuous_scale='RdYlGn_r')
            fig_bar.update_layout(height=250)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            metrics_df = pd.DataFrame({
                'Metric': ['Pregnancies', 'Skin Thickness', 'DPF'],
                'Value': [pregnancies, skin_thickness, round(dpf, 2)]
            })
            fig_table = go.Figure(data=[go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left',
                           font=dict(color='black', size=14)),
                cells=dict(values=[metrics_df.Metric, metrics_df.Value],
                          fill_color='lavender',
                          align='left',
                          font=dict(color='black', size=12)))
            ])
            fig_table.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_table, use_container_width=True)
        
        if prediction[0] == 1:
            st.warning(" Please consult a healthcare professional for proper diagnosis and treatment.")
        else:
            st.info("💡 Maintain a healthy lifestyle to keep diabetes risk low.")

else:  # Normal Disease Check
    model = joblib.load("./Normal disease model/disease_model.pkl")
    feature_names = joblib.load("./Normal disease model/feature_names.pkl")
    scaler = joblib.load("./Normal disease model/scaler (1).pkl")
    label_encoder = joblib.load("./Normal disease model/label_encoder (1).pkl")
    
    st.sidebar.info("This model predicts diseases: Influenza, Common Cold, Eczema, Asthma.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use")
    st.sidebar.markdown("1. Enter symptoms and details\n2. Click Predict\n3. View results")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Example")
    st.sidebar.markdown("**Asthma Case:**\n- Fever: Yes\n- Cough: Yes\n- Fatigue: No\n- Difficulty Breathing: Yes\n- Age: 25\n- Gender: Male\n- Blood Pressure: Normal\n- Cholesterol: Normal")

    # Main content
    st.title("HealthGuard AI - Disease Check")
    st.markdown("### Enter symptoms and patient details")
    st.markdown("💡 How to use: Enter symptoms and patient details → Click Predict button → View results")
    
    if st.button("View Disease History"):
        st.session_state['show_disease_history'] = not st.session_state.get('show_disease_history', False)
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.get('show_disease_history', False):
        st.subheader("Disease Prediction History")
        history = get_disease_history()
        if not history.empty:
            st.dataframe(history[['timestamp', 'age', 'gender', 'disease', 'confidence']], use_container_width=True)
        else:
            st.info("No prediction history available.")
        st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤒 Symptoms")
        fever = st.radio("Fever", ["No", "Yes"], horizontal=True)
        cough = st.radio("Cough", ["No", "Yes"], horizontal=True)
        fatigue = st.radio("Fatigue", ["No", "Yes"], horizontal=True)
        difficulty_breathing = st.radio("Difficulty Breathing", ["No", "Yes"], horizontal=True)
    
    with col2:
        st.subheader("Patient Details")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        age = st.number_input("Age", 1, 120, 25)
        blood_pressure = st.radio("Blood Pressure", ["Low", "Normal", "High"], horizontal=True)
        cholesterol = st.radio("Cholesterol Level", ["Normal", "High"], horizontal=True)
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button("Predict Disease", use_container_width=True, type="primary")
    
    if predict_btn:
        user_input_dict = {
            'Age': age,
            'Fever_Yes': 1 if fever == "Yes" else 0,
            'Cough_Yes': 1 if cough == "Yes" else 0,
            'Fatigue_Yes': 1 if fatigue == "Yes" else 0,
            'Difficulty Breathing_Yes': 1 if difficulty_breathing == "Yes" else 0,
            'Gender_Male': 1 if gender == "Male" else 0,
            'Blood Pressure_Low': 1 if blood_pressure == "Low" else 0,
            'Blood Pressure_Normal': 1 if blood_pressure == "Normal" else 0,
            'Cholesterol Level_High': 1 if cholesterol == "High" else 0
        }
        
        input_df = pd.DataFrame([user_input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        disease_name = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100
        
        save_disease_prediction(age, gender, fever, cough, fatigue, difficulty_breathing,
                               blood_pressure, cholesterol, disease_name, float(confidence))
        
        st.markdown("---")
        st.subheader(" Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.success(f"### Predicted Disease: {disease_name}")
        with result_col2:
            st.metric("Confidence Level", f"{confidence:.2f}%")
        
        st.info(" Please consult a healthcare professional for proper diagnosis and treatment.")
        
        st.markdown("---")
        st.subheader("Disease Probability Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            diseases = label_encoder.classes_
            probs = probabilities[0] * 100
            
            min_len = min(len(diseases), len(probs))
            prob_df = pd.DataFrame({'Disease': diseases[:min_len], 'Probability': probs[:min_len]})
            fig_prob = px.bar(prob_df, x='Disease', y='Probability', 
                             title='Disease Probability Distribution',
                             color='Probability', color_continuous_scale='Blues')
            fig_prob.update_layout(height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            symptoms_data = {
                'Symptom': ['Fever', 'Cough', 'Fatigue', 'Breathing'],
                'Present': [1 if fever == "Yes" else 0, 
                           1 if cough == "Yes" else 0,
                           1 if fatigue == "Yes" else 0,
                           1 if difficulty_breathing == "Yes" else 0]
            }
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=symptoms_data['Present'],
                theta=symptoms_data['Symptom'],
                fill='toself'
            ))
            fig_radar.update_layout(title='Symptoms Profile',
                                   polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                   height=300)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Patient Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Age", f"{age} years")
        with summary_col2:
            st.metric("Gender", gender)
        with summary_col3:
            st.metric("Blood Pressure", blood_pressure)
        with summary_col4:
            st.metric("Cholesterol", cholesterol)
