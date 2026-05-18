import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ---------------------------------------------------
# Başlık
# ---------------------------------------------------

st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# ---------------------------------------------------
# Model yükle
# ---------------------------------------------------

model = YOLO("best (3).pt")

# ---------------------------------------------------
# Upload
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Eğer görsel yüklendiyse
# ---------------------------------------------------

if uploaded_file is not None:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Göster
    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_path = "temp.jpg"
    image.save(temp_path)

    # YOLO tahmini
    results = model(temp_path)

    # Class isimleri
    names = model.names

    # Detection listeleri
    boxes = []
    classes = []

    # Detection varsa al
    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    # ---------------------------------------------------
    # Segment uzunlukları
    # ---------------------------------------------------

    segment_lengths = []

    # Ortalama tray genişliği
    tray_width = image.width * 0.30

    # Detection loop
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        # Sadece core segment
        if label == "core_segment":

            x1, y1, x2, y2 = box

            # Segment uzunluğu
            segment_width = x2 - x1

            # Normalize oran
            length_ratio = segment_width / tray_width

            # Clamp
            length_ratio = min(length_ratio, 1.0)

            # Çok küçük detectionları at
            if length_ratio > 0.02:

                segment_lengths.append(length_ratio)

    # ---------------------------------------------------
    # Duplicate azalt
    # ---------------------------------------------------

    segment_lengths.sort(reverse=True)

    # En büyük 12 segment
    segment_lengths = segment_lengths[:12]

    # ---------------------------------------------------
    # Hesaplar
    # ---------------------------------------------------

    total_core = sum(segment_lengths)

    rqd_core = 0
    scr_core = 0

    for length in segment_lengths:

        # RQD -> 10 cm üstü
        if length >= 0.10:
            rqd_core += length

        # SCR -> 30 cm üstü
        if 0.18 <= length <= 0.40:
            scr_core += length * 0.5

    # Ortalama 4 tray normalize
    tray_count = 4

    tcr = (total_core / tray_count) * 100
    rqd = (rqd_core / tray_count) * 100
    scr = (scr_core / tray_count) * 100

    # Clamp
    tcr = min(tcr, 100)
    rqd = min(rqd, 100)
    scr = min(scr, 100)

    # ---------------------------------------------------
    # Sonuçlar
    # ---------------------------------------------------

    st.success(f"Detected Core Segments: {len(segment_lengths)}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
