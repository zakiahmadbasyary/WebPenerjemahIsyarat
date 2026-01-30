import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Tes Kamera WebRTC", layout="wide")
st.title("Tes Kamera WebRTC (Tanpa YOLO)")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Ambil frame dari browser
        img = frame.to_ndarray(format="bgr24")

        # Resize biar ringan
        img = cv2.resize(img, (480, 360))

        # Tambah teks biar kelihatan real-time
        cv2.putText(
            img,
            "WEBRTC TEST - LIVE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return img

webrtc_streamer(
    key="test-webrtc",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
