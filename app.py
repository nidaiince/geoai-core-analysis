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
# Görsel yükleme
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Görsel geldiyse
# ---------------------------------------------------

if uploaded_file is not None:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Göster
    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_path = "temp.jpg"
    image.save(temp_path)

    # ---------------------------------------------------
    # YOLO Tahmin
    # ---------------------------------------------------

    results = model(temp_path)

    names = model.names

    boxes = []
    classes = []

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    # ---------------------------------------------------
    # Segment uzunlukları
    # ---------------------------------------------------

    segment_lengths = []

    # Ortalama tray genişliği
    tray_width = image.width * 0.30

    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        if label == "core_segment":

            x1, y1, x2, y2 = box

            # Segment genişliği
            segment_width = x2 - x1

            # Normalize uzunluk
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

    total_core = 0
    scr_core = 0
    rqd_core = 0

    for length in segment_lengths:

        # TCR → tüm segmentler
        total_core += length

        # SCR → orta + uzun segmentler
        if length >= 0.15:
            scr_core += length

        # RQD → sadece uzun segmentler
        if length >= 0.20:
            rqd_core += length

    # Ortalama tray sayısı
    tray_count = 4

    # Hesap
    tcr = (total_core / tray_count) * 100
    scr = (scr_core / tray_count) * 100
    rqd = (rqd_core / tray_count) * 100

    # Clamp
    tcr = min(tcr, 100)
    scr = min(scr, 100)
    rqd = min(rqd, 100)

    # ---------------------------------------------------
    # Kırık Sayısı
    # ---------------------------------------------------

    fracture_count = len(segment_lengths)

    # ---------------------------------------------------
    # Kaya Dayanımı
    # ---------------------------------------------------

    if fracture_count <= 5:
        rock_strength = "Çok Sağlam"

    elif fracture_count <= 10:
        rock_strength = "Sağlam"

    elif fracture_count <= 18:
        rock_strength = "Orta Dayanımlı"

    elif fracture_count <= 30:
        rock_strength = "Zayıf"

    else:
        rock_strength = "Çok Zayıf"

    # ---------------------------------------------------
    # Kaya Kalitesi
    # ---------------------------------------------------

    if rqd >= 90:
        rock_quality = "Mükemmel"

    elif rqd >= 75:
        rock_quality = "İyi"

    elif rqd >= 50:
        rock_quality = "Orta"

    elif rqd >= 25:
        rock_quality = "Zayıf"

    else:
        rock_quality = "Çok Zayıf"

    # ---------------------------------------------------
    # Sonuçlar
    # ---------------------------------------------------

    st.success(f"Detected Core Segments: {len(segment_lengths)}")

    st.metric("TCR", f"{tcr:.2f}%")

    st.metric("SCR", f"{scr:.2f}%")

    st.metric("RQD", f"{rqd:.2f}%")

    st.metric("Kırık Sayısı", fracture_count)

    st.metric("Kaya Dayanımı", rock_strength)

    st.metric("Kaya Kalitesi", rock_quality)
