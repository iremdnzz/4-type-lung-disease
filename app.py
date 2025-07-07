import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Modeli yükle
model = tf.keras.models.load_model("akciğer_modeli.h5")

# Sınıf isimleri (Türkçe)
class_labels = [
    'Bakteriyel Zatürre',
    'Koronavirüs Hastalığı',
    'Normal',
    'Verem (Tüberküloz)',
    'Viral Zatürre'
]

# Başlık
st.title("🩺 Akciğer Hastalığı Tespiti")
st.write("Bu uygulama, yüklediğiniz akciğer röntgenine göre bir hastalık tahmini yapar.")

# Görsel yükleme
uploaded_file = st.file_uploader("Lütfen bir röntgen görseli yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli yükle
    img = Image.open(uploaded_file).convert("L")  # Grayscale
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 128, 128, 1)

    # Görseli göster
    st.image(img, caption="Yüklenen Görsel", use_column_width=True, channels="GRAY")

    # Tahmin
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = predictions[predicted_index]

    # Sonuçları göster
    st.markdown(f"""
<div style="text-align: center; padding: 20px; background-color: #f4f6f7; border-radius: 12px;">
    <h2 style="color: #2C3E50;">🔍 Tahmin Edilen Hastalık</h2>
    <p style="font-size: 28px; font-weight: bold; color: #16A085;">{predicted_class}</p>
    <h3 style="color: #2C3E50;">Tahmin Oranı</h3>
    <p style="font-size: 22px; font-weight: bold; color: #D35400;">{confidence:.2%}</p>
</div>
""", unsafe_allow_html=True)
