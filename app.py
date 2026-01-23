import streamlit as st

st.set_page_config(
    page_title="Sistem Penerjemah Bahasa Isyarat",
    layout="wide"
)

st.title("Sistem Penerjemah Bahasa Isyarat")
st.markdown("""
Aplikasi ini merupakan implementasi model **YOLOv8**  
untuk mendeteksi dan menerjemahkan bahasa isyarat secara visual.
""")

st.info("Gunakan menu di sebelah kiri untuk memilih fitur.")
