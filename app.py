import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import streamlit.web.cli as stcli
import sys
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Load the trained model
model = load_model("alexnet_seed_classification.keras")

# Load class labels dynamically
with open("class_indices.json", "r") as f:
    class_labels = json.load(f)  # Load saved class indices
class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping

# Streamlit GUI
def predict_seed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = class_labels.get(class_index, "Unknown")  # Ensure valid mapping
    return class_name

st.title("Seed Classification Using AlexNet")
uploaded_file = st.file_uploader("Upload an image of a seed", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_path = os.path.join("temp.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(img_path, caption="Uploaded Image", use_container_width=True)
    class_name = predict_seed(img_path)
    st.write(f"Predicted Class: {class_name}")


