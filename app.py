import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🏡 Housing Loan Prediction App")
st.write("This app predicts if a customer will subscribe to a term deposit (based on the marketing campaign data).")

# Input form
st.header("Enter Customer Details:")

# --- Sample inputs based on features typically found in the dataset ---
age = st.slider('Age', 18, 95, 30)
duration = st.slider('Call Duration (seconds)', 0, 5000, 300)
campaign = st.slider('Number of Contacts in this Campaign', 1, 50, 3)
pdays = st.slider('Days Since Last Contact (-1 means never contacted)', -1, 999, -1)
previous = st.slider('Number of Previous Contacts', 0, 50, 0)

# Example encoded categorical variables (you can expand based on your encoding map)
job = st.selectbox('Job Type', [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician',
    'unemployed', 'unknown'
])
education = st.selectbox('Education Level', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                              'illiterate', 'professional.course', 'university.degree', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
loan = st.selectbox('Has Personal Loan?', ['yes', 'no'])
housing = st.selectbox('Has Housing Loan?', ['yes', 'no'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])

# --- Encode categorical inputs (you must match label encoding used during training) ---
label_maps = {
    'job': {
        'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
        'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
        'unemployed': 10, 'unknown': 11
    },
    'education': {
        'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4,
        'professional.course': 5, 'university.degree': 6, 'unknown': 7
    },
    'marital': {
        'divorced': 0, 'married': 1, 'single': 2, 'unknown': 3
    },
    'loan': {'yes': 1, 'no': 0},
    'housing': {'yes': 1, 'no': 0},
    'contact': {'cellular': 0, 'telephone': 1}
}

# Collect input data in model's expected format
input_data = np.array([
    age,
    duration,
    campaign,
    pdays,
    previous,
    label_maps['job'][job],
    label_maps['education'][education],
    label_maps['marital'][marital],
    label_maps['loan'][loan],
    label_maps['housing'][housing],
    label_maps['contact'][contact]
]).reshape(1, -1)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The customer is likely to subscribe to the loan.")
    else:
        st.error("❌ The customer is not likely to subscribe to the loan.")
