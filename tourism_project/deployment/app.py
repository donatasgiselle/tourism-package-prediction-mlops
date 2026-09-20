import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "")
MODEL_FILE = "best_tourism_package_model_v1.joblib"
CLASSIFICATION_THRESHOLD = 0.45


@st.cache_resource
def load_model():
    """Download the registered model from the Hugging Face model hub once per container."""
    path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILE, repo_type="model")
    return joblib.load(path)


st.set_page_config(page_title="Wellness Tourism Package Prediction")
st.title("Wellness Tourism Package Prediction")
st.write("Enter the customer profile and the details of the sales interaction to estimate "
         "whether the customer is likely to purchase the package.")

model = load_model()

st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=18, max_value=80, value=35)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly income", min_value=1000, max_value=100000, value=23000, step=500)
    CityTier = st.selectbox("City tier", [1, 2, 3])
with col2:
    NumberOfPersonVisiting = st.number_input("Persons visiting", min_value=1, max_value=5, value=3)
    NumberOfChildrenVisiting = st.number_input("Children visiting (below 5)", min_value=0, max_value=3, value=1)
    PreferredPropertyStar = st.selectbox("Preferred hotel rating", [3, 4, 5])
    NumberOfTrips = st.number_input("Trips per year", min_value=1, max_value=25, value=3)
    Passport = st.radio("Valid passport", ["Yes", "No"], horizontal=True)
    OwnCar = st.radio("Owns a car", ["Yes", "No"], horizontal=True)

st.subheader("Sales interaction")
col3, col4 = st.columns(2)
with col3:
    TypeofContact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    ProductPitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=130, value=15)
with col4:
    NumberOfFollowups = st.number_input("Number of follow-ups", min_value=1, max_value=6, value=4)
    PitchSatisfactionScore = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect the inputs in a single-row DataFrame with the training column names
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_data)[0, 1])
    st.metric("Purchase probability", f"{probability:.0%}")
    if probability >= CLASSIFICATION_THRESHOLD:
        st.success("The customer is likely to purchase the package. Recommended for contact.")
    else:
        st.info("The customer is unlikely to purchase the package.")
    with st.expander("Input sent to the model"):
        st.dataframe(input_data)
