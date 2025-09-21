import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="🏝️",
    layout="wide"
)

@st.cache_data
def load_model():
    """Load the trained model from Hugging Face"""
    try:
        model_path = hf_hub_download(
            repo_id="dr-psych/tourism_project_model",
            filename="best_tourism_model_v1.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Tourism Package Purchase Prediction")
    st.write("This application predicts whether a customer will purchase a Wellness Tourism Package based on their demographics and interaction data.")

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    st.success("Model loaded successfully!")

    # Create input form
    st.header("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=200000, value=50000, step=1000)

    with col2:
        st.subheader("Travel & Preferences")
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_children = st.number_input("Number of Children (below 5)", min_value=0, max_value=5, value=0)
        preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
        passport = st.selectbox("Has Passport", ["Yes", "No"])
        own_car = st.selectbox("Owns Car", ["Yes", "No"])

    st.subheader("Sales Interaction")
    col3, col4 = st.columns(2)

    with col3:
        type_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    with col4:
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
        duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=120, value=30)

    # Create feature vector
    # Note: This assumes your model expects these exact features in this order
    # You may need to adjust based on your actual model's feature requirements
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [1 if type_contact == "Company Invited" else 0],
        'CityTier': [city_tier],
        'Occupation': [{'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Freelancer': 3}[occupation]],
        'Gender': [1 if gender == "Male" else 0],
        'NumberOfPersonVisiting': [num_persons],
        'PreferredPropertyStar': [preferred_star],
        'MaritalStatus': [{'Single': 0, 'Married': 1, 'Divorced': 2}[marital_status]],
        'NumberOfTrips': [num_trips],
        'Passport': [1 if passport == "Yes" else 0],
        'OwnCar': [1 if own_car == "Yes" else 0],
        'NumberOfChildrenVisiting': [num_children],
        'Designation': [{'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}[designation]],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_satisfaction],
        'ProductPitched': [{'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4}[product_pitched]],
        'NumberOfFollowups': [num_followups],
        'DurationOfPitch': [duration_pitch]
    })

    # Add engineered features (matching the training pipeline)
    input_data['Income_per_person'] = input_data['MonthlyIncome'] / (input_data['NumberOfPersonVisiting'] + 1)
    input_data['Trips_per_year_ratio'] = input_data['NumberOfTrips'] / 12
    input_data['Children_ratio'] = input_data['NumberOfChildrenVisiting'] / input_data['NumberOfPersonVisiting'] if input_data['NumberOfPersonVisiting'].iloc[0] > 0 else 0
    input_data['Followup_per_pitch'] = input_data['NumberOfFollowups'] / (input_data['DurationOfPitch'] + 1)

    # Make prediction
    if st.button("Predict Purchase Decision", type="primary"):
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.header("Prediction Results")

            if prediction == 1:
                st.success("✅ WILL PURCHASE")
                st.write("This customer is likely to purchase the Wellness Tourism Package!")
            else:
                st.error("❌ WON'T PURCHASE")
                st.write("This customer is unlikely to purchase the Wellness Tourism Package.")

            st.subheader("Confidence Scores")
            col5, col6 = st.columns(2)

            with col5:
                st.metric("Won't Purchase Probability", f"{prediction_proba[0]:.2%}")

            with col6:
                st.metric("Will Purchase Probability", f"{prediction_proba[1]:.2%}")
                

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Please check if all required features are properly formatted.")

if __name__ == "__main__":
    main()
