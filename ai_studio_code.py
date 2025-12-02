import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Trash Detection", page_icon="🗑️")

st.title("🗑️ Smart Trash Bin Monitor")
st.write("Sistem deteksi volume sampah berbasis AI menggunakan YOLOv8.")

# --- CLASS DETEKSI ---
class SmartTrashBin:
    def __init__(self, model_path='yolov8n.pt', threshold=70.0):
        # Load model (akan download otomatis jika belum ada)
        self.model = YOLO(model_path)
        self.full_threshold = threshold
        # ID Class COCO: 39=Bottle, 41=Cup, 67=Cell phone (contoh sampah)
        # Jika pakai model custom, hapus filter ini
        self.classes_to_detect = [39, 41, 67] 

    def calculate_volume(self, boxes, img_area):
        total_trash_area = 0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            total_trash_area += area
        
        fill_percentage = (total_trash_area / img_area) * 100
        return min(fill_percentage, 100.0)

    def process_frame(self, image_input):
        # Konversi PIL Image ke OpenCV format
        frame = np.array(image_input)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        height, width, _ = frame.shape
        total_area = height * width

        # Deteksi
        results = self.model.predict(frame, conf=0.25, classes=self.classes_to_detect)
        result = results[0]
        
        # Gambar anotasi
        annotated_frame = result.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) # Balik ke RGB untuk Streamlit

        # Hitung Volume
        current_fill = 0.0
        if result.boxes:
            current_fill = self.calculate_volume(result.boxes, total_area)

        return annotated_frame, current_fill

# --- ANTARMUKA WEBSITE ---

# 1. Sidebar untuk pengaturan
st.sidebar.header("Pengaturan")
threshold_input = st.sidebar.slider("Batas Penuh (%)", 0, 100, 70)

# Inisialisasi Sistem
detector = SmartTrashBin(threshold=threshold_input)

# 2. Upload Gambar
uploaded_file = st.file_uploader("Upload Foto Tempat Sampah", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file)
    
    # Proses tombol
    if st.button("Hitung Volume Sampah"):
        with st.spinner('Sedang memproses...'):
            result_img, fill_level = detector.process_frame(image)
            
            # Tampilkan Hasil Gambar
            st.image(result_img, caption="Hasil Deteksi", use_column_width=True)
            
            # Tampilkan Metrik
            st.metric(label="Estimasi Volume Sampah", value=f"{fill_level:.2f}%")
            
            # Progress Bar Visual
            st.progress(int(fill_level))
            
            # Logika Notifikasi
            if fill_level > threshold_input:
                st.error(f"🚨 PERINGATAN: Sampah Penuh! (Level > {threshold_input}%)")
                st.toast("Notifikasi dikirim ke petugas!", icon="📧")
            else:
                st.success("✅ Status: Masih Aman")

# Footer
st.markdown("---")
st.caption("Dibuat dengan Streamlit & YOLOv8")