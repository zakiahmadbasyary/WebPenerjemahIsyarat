import streamlit as st
import cv2
from ultralytics import YOLO

st.title("Deteksi Bahasa Isyarat (Webcam)")

MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# ===============================
# PILIH KAMERA
# ===============================
camera_index = st.selectbox(
    "Pilih Kamera",
    options=[0, 1, 2, 3],
    index=0
)

run = st.checkbox("Aktifkan Kamera")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("Kamera tidak dapat dibuka")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame tidak terbaca")
                break

            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()

            FRAME_WINDOW.image(annotated_frame, channels="BGR")

if cap:
    cap.release()
