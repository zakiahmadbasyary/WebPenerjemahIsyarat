import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Deteksi Bahasa Isyarat", layout="wide")
st.title("Deteksi Bahasa Isyarat (Realtime Kamera)")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# ===============================
# VIDEO PROCESSOR
# ===============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # OPTIONAL: resize biar FPS naik
        img = cv2.resize(img, (480, 360))

        # Skip frame (optimasi FPS)
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return img

        # YOLO inference
        results = model(img, conf=0.5)
        annotated_frame = results[0].plot()

        return annotated_frame

# ===============================
# WEBRTC STREAM
# ===============================
webrtc_streamer(
    key="sibi-realtime",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
