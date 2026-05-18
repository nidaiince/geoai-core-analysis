import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Başlık
st.title("GeoNA Core Analysis")
st.write("AI-based geological core analysis system")

# Model yükle
model = YOLO("best (3).pt")

# Dosya yükleme
uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

# Görüntü varsa çalış
if uploaded_file is not None:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Görsel göster
    st.image(image, caption="Yüklenen Görüntü")

    # Geçici kaydet
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Tahmin
    results = model(temp_path)

    # Sınıf isimleri
    names = model.names

    # Box ve class listeleri
    boxes = []
    classes = []

    # Detection varsa al
    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    # Sayaçlar
    core_count = 0

    total_core_ratio = 0
    rqd_ratio = 0
    scr_ratio = 0

    # Görüntü genişliği
    image_width = image.width

    # Detection loop
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        # Sadece core segmentleri kullan
        if label == "core_segment":

            core_count += 1

            # Box koordinatları
            x1, y1, x2, y2 = box

            # Segment genişliği
            segment_width = x2 - x1

            # Görsel oranı
            ratio = segment_width / image_width

            # TCR
            total_core_ratio += ratio

            # RQD
            if ratio >= 0.03:
                rqd_ratio += ratio

            # SCR
            if ratio >= 0.08:
                scr_ratio += ratio

    # Yüzdelere çevir
    tcr = total_core_ratio * 100
    rqd = rqd_ratio * 100
    scr = scr_ratio * 100

    # Limit
    tcr = min(tcr, 100)
    rqd = min(rqd, 100)
    scr = min(scr, 100)

    # Sonuçlar
    st.success(f"Detected Core Segments: {core_count}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
