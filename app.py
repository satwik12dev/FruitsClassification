import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.header('Image Classification Model - Fruits & Vegetables')

model = load_model('Image_classify.h5', compile=False)


data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika',
    'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach',
    'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

st.subheader("Upload Image or Paste Image URL")
option = st.radio("Select image input method:", ('Upload from device', 'Enter image URL'))

image = None

if option == 'Upload from device':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == 'Enter image URL':
    url = st.text_input("Paste image URL here:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Failed to load image from URL. Please check the URL and try again.")


if image:
    st.image(image, caption="Uploaded Image", width=200)

    image = image.convert("RGB")
    image = image.resize((img_width, img_height))

    img_array = tf.keras.utils.img_to_array(image) # type: ignore
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])

    st.success(f'Prediction: **{data_cat[np.argmax(score)]}**')
    st.info(f'Confidence: **{np.max(score) * 100:.2f}%**')
