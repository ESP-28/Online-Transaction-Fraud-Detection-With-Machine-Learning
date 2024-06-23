#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Creating Prediction application using Streamlit


# In[13]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# In[14]:


# Define the file path to the saved model
model_file_path = 'random_forest_model.joblib'

# Load the trained model
rf_model = joblib.load(model_file_path)


# In[15]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_file_path = 'random_forest_model.joblib'
rf_model = joblib.load(model_file_path)

# Define the title and introductory text for the app
st.title('Online Fraud Detection App')
st.write('Enter transaction details to predict if it is fraudulent.')

# Add input fields for user input
st.sidebar.header('User Input')
amount = st.sidebar.number_input('Transaction Amount', min_value=0.0)
old_balance = st.sidebar.number_input('Old Balance', min_value=0.0)
new_balance = st.sidebar.number_input('New Balance', min_value=0.0)
type_of_transaction = st.sidebar.selectbox('Transaction Type', ['Type 1', 'Type 2', 'Type 3'])

# Process the input to match the model's requirements
input_data = pd.DataFrame({
    'amount': [amount],
    'oldbalanceOrg': [old_balance],
    'newbalanceOrig': [new_balance],
    'type': [type_of_transaction]
})

# Perform any necessary data preprocessing
# Example: Encoding categorical variables if needed
# Example: Scaling numerical variables if needed

# Make predictions using the model
try:
    prediction = rf_model.predict(input_data)
    st.write('Prediction:')
    if prediction[0] == 1:
        st.write('Fraudulent Transaction')
    else:
        st.write('Non-Fraudulent Transaction')

except ValueError as ve:
    st.error(f"Error encountered: {ve}")


# In[ ]:




