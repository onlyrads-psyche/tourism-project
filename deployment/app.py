import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        text-align: center;
        margin: 20px 0;
    }

    .will-buy {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }

    .wont-buy {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        # Load from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="dr-psych/tourism_project",
            filename="best_model.pkl",
            repo_type="model"
        )

        scaler_path = hf_hub_download(
            repo_id="dr-psych/tourism_project",
            filename="scaler.pkl",
            repo_type="dataset"
        )

        encoders_path = hf_hub_download(
            repo_id="dr-psych/tourism_project",
            filename="label_encoders.pkl",
            repo_type="dataset"
        )

        # Load objects
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)

        return model, scaler, label_encoders

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_input(input_data, scaler, label_encoders):
    """Preprocess user input for prediction"""

    # Create feature engineering (same as training)
    input_data['Income_per_person'] = input_data['MonthlyIncome'] / (input_data['NumberOfPersonVisiting'] + 1)
    input_data['Trips_per_year_ratio'] = input_data['NumberOfTrips'] / 12
    input_data['Children_ratio'] = input_data['NumberOfChildrenVisiting'] / input_data['NumberOfPersonVisiting'] if input_data['NumberOfPersonVisiting'] > 0 else 0
    input_data['Followup_per_pitch'] = input_data['NumberOfFollowups'] / (input_data['DurationOfPitch'] + 1)

    # Handle infinite values
    for col in input_data.columns:
        if np.isinf(input_data[col]) or np.isnan(input_data[col]):
            input_data[col] = 0

    # Create DataFrame for processing
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                          'ProductPitched', 'Designation']

    for col in categorical_columns:
        if col in label_encoders and col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unseen categories
                df[col] = 0

    # Scale numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

def create_input_form():
    """Create input form for user data"""

    st.markdown('<div class="sub-header">📊 Customer Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])

    with col2:
        st.subheader("Financial & Travel Info")
        monthly_income = st.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])

    with col3:
        st.subheader("Trip Details")
        num_persons = st.slider("Number of Persons Visiting", 1, 10, 2)
        num_children = st.slider("Number of Children (below 5)", 0, 5, 0)
        preferred_star = st.slider("Preferred Property Star", 3, 5, 4)
        num_trips = st.slider("Number of Trips per Year", 0, 20, 2)

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Contact & Pitch Info")
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    with col5:
        st.subheader("Engagement Metrics")
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        num_followups = st.slider("Number of Follow-ups", 0, 10, 2)
        duration_pitch = st.slider("Duration of Pitch (minutes)", 5, 120, 30)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    # Create input dictionary
    input_data = {
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': num_persons,
        'PreferredPropertyStar': preferred_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': num_trips,
        'Passport': 1 if passport == "Yes" else 0,
        'OwnCar': 1 if own_car == "Yes" else 0,
        'NumberOfChildrenVisiting': num_children,
        'Designation': designation,
        'MonthlyIncome': monthly_income,
        'PitchSatisfactionScore': pitch_satisfaction,
        'ProductPitched': product_pitched,
        'NumberOfFollowups': num_followups,
        'DurationOfPitch': duration_pitch
    }

    return input_data

def create_prediction_visualization(prediction_proba):
    """Create visualization for prediction results"""

    # Create gauge chart for prediction probability
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_proba[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Purchase Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application"""

    # Title
    st.markdown('<div class="main-header">🏝️ Tourism Package Predictor</div>', unsafe_allow_html=True)
    st.markdown("**Predict whether a customer will purchase the Wellness Tourism Package**")

    # Load model and preprocessors
    model, scaler, label_encoders = load_model_and_preprocessors()

    if model is None:
        st.error("❌ Could not load the model. Please check the model repository.")
        return

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This application uses machine learning to predict the likelihood "
            "of a customer purchasing a Wellness Tourism Package based on their "
            "demographic and interaction data."
        )

        st.header("📈 Model Info")
        st.success("✅ Model loaded successfully!")

        st.header("🎯 How to Use")
        st.markdown("""
        1. Fill in the customer information
        2. Click 'Make Prediction'
        3. View the prediction result and probability
        """)

    # Input form
    input_data = create_input_form()

    # Prediction section
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🔮 Make Prediction", type="primary", use_container_width=True):

            # Show loading
            with st.spinner("Making prediction..."):

                # Preprocess input
                processed_data = preprocess_input(input_data, scaler, label_encoders)

                # Make prediction
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]

                # Display results
                st.markdown("### 🎯 Prediction Results")

                # Prediction box
                if prediction == 1:
                    st.markdown(
                        '<div class="prediction-box will-buy">'
                        '<h2>✅ WILL PURCHASE</h2>'
                        '<p>This customer is likely to purchase the Wellness Tourism Package!</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box wont-buy">'
                        '<h2>❌ WON\'T PURCHASE</h2>'
                        '<p>This customer is unlikely to purchase the Wellness Tourism Package.</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                # Probability visualization
                fig = create_prediction_visualization(prediction_proba)
                st.plotly_chart(fig, use_container_width=True)

                # Probability breakdown
                col_prob1, col_prob2 = st.columns(2)

                with col_prob1:
                    st.metric(
                        "Won't Purchase Probability",
                        f"{prediction_proba[0]:.2%}",
                        delta=None
                    )

                with col_prob2:
                    st.metric(
                        "Will Purchase Probability",
                        f"{prediction_proba[1]:.2%}",
                        delta=None
                    )

                # Additional insights
                st.markdown("### 💡 Insights")
                if prediction_proba[1] > 0.7:
                    st.success("🎯 High conversion probability - Priority customer for targeted marketing!")
                elif prediction_proba[1] > 0.4:
                    st.warning("⚠️ Moderate conversion probability - May need additional incentives.")
                else:
                    st.info("📊 Low conversion probability - Focus on other prospects or reconsider approach.")

if __name__ == "__main__":
    main()
