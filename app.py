import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Cardio Risk Assessment", page_icon="icon.png")

# ========== LOAD CSS ==========
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ========== MODEL LOADING ==========
try:
    loaded_model = joblib.load('random_forest_model.joblib')
    model = loaded_model.best_estimator_
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# ========== HEADER SECTION ==========
st.markdown("""
<h1>Cardiovascular Disease Risk Assessment</h1>
<p>
This tool estimates your 10-year risk of cardiovascular disease based on established clinical metrics.<br>
<em>For medical professionals only. Always combine with clinical judgment.</em>
</p>
""", unsafe_allow_html=True)

# ========== INPUT SECTION ==========
with st.expander("PATIENT INFORMATION", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.radio("Sex", ["Male", "Female"])
    with col2:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=35)
    with col3:
        family_history = st.checkbox("Family history of early CVD")

with st.expander("CLINICAL MEASUREMENTS", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        cholesterol = st.selectbox(
            "Cholesterol Level",
            ["Normal", "Above Normal", "Well Above Normal"],
            index=1
        )
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=300, value=135)
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=200, value=83)

    with col2:
        gluc = st.selectbox(
            "Glucose Level",
            ["Normal", "Above Normal", "Well Above Normal"],
            index=1
        )
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)

with st.expander("LIFESTYLE FACTORS", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        smoke = st.checkbox("Current smoker")
    with col2:
        active = st.checkbox("Regular exercise (≥150 min/week)", value=False)

# ========== DYNAMIC THRESHOLDS ==========
if age < 40:
    HIGH_RISK_THRESHOLD = 0.35
    MODERATE_RISK_THRESHOLD = 0.15
    HIGH_RISK_LABEL = "Elevated Risk"
elif age < 60:
    HIGH_RISK_THRESHOLD = 0.5
    MODERATE_RISK_THRESHOLD = 0.25
    HIGH_RISK_LABEL = "High Risk"
else:
    HIGH_RISK_THRESHOLD = 0.6
    MODERATE_RISK_THRESHOLD = 0.35
    HIGH_RISK_LABEL = "Very High Risk"

# ========== DATA PROCESSING ==========
gender_encoded = 1 if gender == "Female" else 0
chol_encoded = ["Normal", "Above Normal", "Well Above Normal"].index(cholesterol) + 1
gluc_encoded = ["Normal", "Above Normal", "Well Above Normal"].index(gluc) + 1
smoke_encoded = 1 if smoke else 0
active_encoded = 1 if active else 0
family_history_encoded = 1 if family_history else 0

bmi = weight / ((height/100) ** 2)
map_value = ((2 * ap_lo) + ap_hi) / 3

# BMI Classification
if bmi < 18.5: 
    bmi_class = 1
    bmi_category = "Underweight"
elif 18.5 <= bmi < 25: 
    bmi_class = 2
    bmi_category = "Normal"
elif 25 <= bmi < 30: 
    bmi_class = 3
    bmi_category = "Overweight"
elif 30 <= bmi < 35: 
    bmi_class = 4
    bmi_category = "Obese (Class I)"
elif 35 <= bmi < 40: 
    bmi_class = 5
    bmi_category = "Obese (Class II)"
else: 
    bmi_class = 6
    bmi_category = "Obese (Class III)"

# BP Classification
bp_status = "Normal"
bp_impact = 0
if ap_hi >= 140 or ap_lo >= 90:
    bp_status = "Hypertensive"
    bp_impact = 0.15
elif ap_hi >= 130 or ap_lo >= 85:
    bp_status = "Borderline Elevated"
    bp_impact = 0.05

# MAP Classification
if map_value < 70: map_class = 1
elif 70 <= map_value < 80: map_class = 2
elif 80 <= map_value < 90: map_class = 3
elif 90 <= map_value < 100: map_class = 4
elif 100 <= map_value < 110: map_class = 5
elif 110 <= map_value < 120: map_class = 6
else: map_class = 7

# Age bin
age_bin = pd.cut([age], bins=[0,20,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
                labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    'gender': [gender_encoded],
    'cholesterol': [chol_encoded],
    'gluc': [gluc_encoded],
    'smoke': [smoke_encoded],
    'active': [active_encoded],
    'BMI_Class': [bmi_class],
    'MAP_Class': [map_class],
    'age_bin': [age_bin],
    'family_history': [family_history_encoded],
    'Cluster': [0]
})

try:
    input_data = input_data[model.feature_names_in_]
except Exception as e:
    st.error(f"Feature mismatch: {str(e)}")
    st.stop()

# ========== PREDICTION ==========
if st.button("Predict", type="primary"):
    try:
        probability = model.predict_proba(input_data)[0][1]
        
        # Age-based calibration
        if age < 30:
            probability *= 0.6
        elif age < 40:
            probability *= 0.8
            
        if family_history:
            probability = min(probability * 1.15, 0.8)
        
        probability = min(probability + bp_impact, 0.95)
        
        # Risk classification
        if probability >= HIGH_RISK_THRESHOLD:
            risk_level = f"🔴 {HIGH_RISK_LABEL}"
            risk_class = "risk-high"
            recommendation = "Urgent physician referral + diagnostic testing"
        elif probability >= MODERATE_RISK_THRESHOLD:
            risk_level = "🟠 Moderate Risk"
            risk_class = "risk-moderate"
            recommendation = "Lifestyle intervention + enhanced monitoring"
        else:
            risk_level = "🟢 Low Risk"
            risk_class = "risk-low"
            recommendation = "Maintain preventive care"
        
        # Display results
        st.markdown(f"""
        ## <span class="{risk_class}">Risk Assessment: {risk_level}</span>
        **10-Year Probability**: {probability:.1%}
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Clinical Recommendation**:  
        {recommendation}
        """)
        
        # Risk factor analysis
        st.subheader("Key Risk Factors")
        factors = {
            f'Age ({age} years)': True,
            'Male biological sex': gender == "Male",
            'Family history of CVD': family_history,
            f'{cholesterol} cholesterol': cholesterol != "Normal",
            f'{gluc} glucose': gluc != "Normal",
            'Current smoker': smoke,
            'Physically inactive': not active,
            f'{bmi_category} (BMI: {bmi:.1f})': bmi >= 25,
            f'BP {ap_hi}/{ap_lo} mmHg ({bp_status})': bp_status != "Normal"
        }
        
        present_factors = [f for f, exists in factors.items() if exists]
        
        if present_factors:
            cols = st.columns(2)
            for i, factor in enumerate(present_factors):
                cols[i%2].write(f"• {factor}")
        
        # Clinical context
        st.subheader("Clinical Context")
        st.markdown(f"""
        - **Blood Pressure**: {ap_hi}/{ap_lo} mmHg ({bp_status})  
        - **BMI**: {bmi:.1f} ({bmi_category})
        """)

    except Exception as e:
        st.error("Prediction failed. Please check inputs and try again.")

# ========== FOOTER ==========
st.markdown("""
---
*This tool is for clinical decision support only. Not a substitute for professional medical judgment.*
""")
