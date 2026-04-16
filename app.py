import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Predictive Insights Pro", page_icon="🤖", layout="centered")

# Custom CSS for glassmorphism and animations
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-container {
        animation: fadeIn 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model (5).pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    
    st.title("🤖 Prediction Dashboard")
    st.markdown("Enter the details below to generate a prediction.")
    
    with st.container():
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            
        with col2:
            income = st.number_input("Annual Income", min_value=0, value=50000)
            occupation = st.selectbox("Occupation", ["Professional", "Labor", "Service", "Student", "Other"])

        st.write("---")
        
        if st.button("Generate Prediction"):
            # ENCODING STEP: 
            # Your model expects numerical values. Map your strings to the numbers used in training.
            # Example mapping (Replace with your actual training logic):
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            region_map = {"North": 0, "South": 1, "East": 2, "West": 3, "Central": 4}
            occ_map = {"Professional": 0, "Labor": 1, "Service": 2, "Student": 3, "Other": 4}
            
            # Construct feature array based on model: Age, Gender, Region, Occupation, Income
            features = np.array([[
                age, 
                gender_map[gender], 
                region_map[region], 
                occ_map[occupation], 
                income
            ]])
            
            # Get Prediction
            prediction = model.predict(features)
            probability = model.predict_proba(features)
            
            # Animated result display
            st.balloons()
            st.subheader("Result:")
            if prediction[0] == "yes":
                st.success(f"Outcome: **YES** (Confidence: {np.max(probability)*100:.2f}%)")
            else:
                st.error(f"Outcome: **NO** (Confidence: {np.max(probability)*100:.2f}%)")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Ensure 'model (5).pkl' is in the same directory as this script.")
