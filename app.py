import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

model_path = "67102010518_mnist_model.keras"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Cannot load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)

        st.image(img, caption="Uploaded Image")

        img = img.convert("L")
        img = img.resize((28, 28))

        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)

        st.success(
            f"The model predicts the digit is: **{predicted_digit}**"
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
