import certifi
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Завантаження моделей
cnn_model = tf.keras.models.load_model('cnn_model.h5')
vgg16_model = VGG16(weights='imagenet')

# Функція для передбачення класу зображення за допомогою вашої моделі
def predict_with_cnn_model(img):
    img = img.convert('RGB')  # Конвертуємо зображення в RGB
    img = img.resize((32, 32))  # Змінено розмір для відповідності моделі CIFAR-10
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = cnn_model.predict(img_array)
    return predictions

# Функція для передбачення класу зображення за допомогою VGG16
def predict_with_vgg16_model(img):
    img = img.convert('RGB')  # Конвертуємо зображення в RGB
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = vgg16_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Функція для відображення графіків функції втрат і точності
def plot_metrics(history):
    if 'loss' in history and 'accuracy' in history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Графік функції втрат
        ax1.plot(history['loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Графік точності
        ax2.plot(history['accuracy'], label='Train Accuracy')
        ax2.plot(history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        st.pyplot(fig)
    else:
        st.write("No training history available to plot metrics.")

# Інтерфейс Streamlit
st.title("Image Classification Web App")
st.write("Upload an image to classify it using the trained CNN model or VGG16 model.")

# Вибір моделі
model_option = st.selectbox("Choose a model", ("CNN Model", "VGG16 Model"))

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    if model_option == "CNN Model":
        # Передбачення за допомогою вашої моделі
        cnn_predictions = predict_with_cnn_model(img)
        st.write("Predictions from your CNN model:")
        st.write(cnn_predictions)
        
        # Виведення графіків функції втрат і точності
        # Завантаження історії навчання моделі
        try:
            history = np.load('cnn_model_history.npy', allow_pickle=True).item()
            plot_metrics(history)
        except FileNotFoundError:
            st.write("No training history available for the CNN model.")
        
    elif model_option == "VGG16 Model":
        # Передбачення за допомогою VGG16
        vgg16_predictions = predict_with_vgg16_model(img)
        st.write("Predictions from VGG16 model:")
        for pred in vgg16_predictions:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
