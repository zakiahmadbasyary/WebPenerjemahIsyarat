import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model (path sesuaikan)
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

st.title("Deteksi Bahasa Isyarat dari Gambar")

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("Gambar Asli")
    st.image(image, use_column_width=True)

    # Inference
    results = model(img_array)

    # Visualisasi hasil
    annotated_img = results[0].plot()

    st.subheader("Hasil Deteksi")
    st.image(annotated_img, use_column_width=True)

    # Ambil label
    if len(results[0].boxes) > 0:
        labels = results[0].names
        detected_classes = results[0].boxes.cls.cpu().numpy()

        st.subheader("Kelas Terdeteksi")
        for cls in detected_classes:
            st.write(f"- {labels[int(cls)]}")
    else:
        st.warning("Tidak ada objek terdeteksi.")
