import streamlit as st
from predict import predict_image

st.title("Car Damage Classification 🚗💥")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    damage_type, confidence = predict_image("temp.jpg")
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    st.write(f"**Damage Type:** {damage_type}")
    st.write(f"**Confidence:** {confidence:.2f}")
