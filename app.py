import streamlit as st
import numpy as np
import joblib
model = joblib.load("hypertuned_tfidf_logreg_model.joblib")  # Load the pre-trained model
print(model)

st.title("Sentiment Review App")

# Create a text input box for the user to enter text
user_input = st.text_area("Enter your review:")

# Predict the review when the user submits text
if st.button('Check your Review'):
    if user_input:
        user_input = [user_input]
        prediction = model.predict(user_input)  # Get model prediction
        st.write(f"Review: {prediction}")
    else:
        st.write("Please enter a review text to analyze.")