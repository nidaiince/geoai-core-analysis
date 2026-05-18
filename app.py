import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Başlık
st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# Model yükle
model = YOLO("best (3).pt")

# Upload
uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Göster
    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Tahmin
    results = model(temp_path)

    names = model.names

    boxes = []
    classes = []

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    # Sayaçlar
    core_count = 0

    total_core = 0
    rqd_core = 0
    scr_core = 0

    # Ortalama tray sayısı
    tray_count = 4

    # Ortalama tray genişliği
    tray_width = image.width * 0.22

    # Segmentler
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        if label == "core_segment":

            core_count += 1

            x1, y1, x2, y2 = box

            # Normalize uzunluk
            length_ratio = (x2 - x1) / tray_width

            total_core += length_ratio

            # RQD
            if length_ratio >= 0.10:
                rqd_core += length_ratio

            # SCR
            if length_ratio >= 0.30:
                scr_core += length_ratio

    # Tray sayısına böl
    total_core = total_core / tray_count
    rqd_core = rqd_core / tray_count
    scr_core = scr_core / tray_count

    # Yüzde
    tcr = total_core * 100
    rqd = rqd_core * 100
    scr = scr_core * 100

    # Limit
    tcr = min(tcr, 100)
    rqd = min(rqd, 100)
    scr = min(scr, 100)

    # Sonuçlar
    st.success(f"Detected Core Segments: {core_count}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
