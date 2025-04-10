import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('iris_model.pkl')

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict the type of Iris flower using ML model")

# User input via sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    classes = ['Setosa 🌼', 'Versicolor 🌺', 'Virginica 🌻']
    st.success(f"Prediction: {classes[prediction]}")
