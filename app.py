import streamlit as st
import numpy as np
import pickle

# Load model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏠 Housing Loan Subscription Predictor")
st.write("Enter customer details to predict if they will subscribe to a term deposit (loan).")

# --- Categorical input mappings (same as in LabelEncoder used during training) ---
label_maps = {
    'job': {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
            'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
            'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11},

    'marital': {'divorced': 0, 'married': 1, 'single': 2, 'unknown': 3},

    'education': {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
                  'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7},

    'default': {'yes': 1, 'no': 0, 'unknown': 2},

    'housing': {'yes': 1, 'no': 0, 'unknown': 2},

    'loan': {'yes': 1, 'no': 0, 'unknown': 2},

    'contact': {'cellular': 0, 'telephone': 1},

    'month': {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
              'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},

    'day_of_week': {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
}

# --- Streamlit Form Inputs ---

job = st.selectbox("Job", list(label_maps['job'].keys()))
marital = st.selectbox("Marital Status", list(label_maps['marital'].keys()))
education = st.selectbox("Education", list(label_maps['education'].keys()))
default = st.selectbox("Credit in Default?", list(label_maps['default'].keys()))
housing = st.selectbox("Housing Loan?", list(label_maps['housing'].keys()))
loan = st.selectbox("Personal Loan?", list(label_maps['loan'].keys()))
contact = st.selectbox("Contact Type", list(label_maps['contact'].keys()))
month = st.selectbox("Last Contact Month", list(label_maps['month'].keys()))
day_of_week = st.selectbox("Day of Week Contacted", list(label_maps['day_of_week'].keys()))

duration = st.slider("Call Duration (seconds)", 0, 5000, 300)
campaign = st.slider("Number of Contacts During Campaign", 1, 50, 2)
previous = st.slider("Number of Previous Contacts", 0, 30, 0)
emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 0.0, step=0.1)

# Encode all inputs
input_data = np.array([[
    label_maps['job'][job],
    label_maps['marital'][marital],
    label_maps['education'][education],
    label_maps['default'][default],
    label_maps['housing'][housing],
    label_maps['loan'][loan],
    label_maps['contact'][contact],
    label_maps['month'][month],
    label_maps['day_of_week'][day_of_week],
    duration,
    campaign,
    previous,
    emp_var_rate
]])

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The customer is likely to subscribe to the loan.")
    else:
        st.error("❌ The customer is not likely to subscribe to the loan.")
