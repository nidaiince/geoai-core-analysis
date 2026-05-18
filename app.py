import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Başlık
st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# Model
model = YOLO("best (3).pt")

# Upload
uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Görsel
    image = Image.open(uploaded_file).convert("RGB")

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

    # Segment uzunlukları
    segment_lengths = []

    # Tray genişliği
    tray_width = image.width * 0.22

    # Detection loop
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        if label == "core_segment":

            x1, y1, x2, y2 = box

            length_ratio = (x2 - x1) / tray_width

            segment_lengths.append(length_ratio)

    # Büyükten küçüğe sırala
    segment_lengths.sort(reverse=True)

    # İlk 12 segmenti kullan
    segment_lengths = segment_lengths[:12]

    # Hesaplar
    total_core = sum(segment_lengths)

    rqd_core = sum([x for x in segment_lengths if x >= 0.10])

    scr_core = sum([x for x in segment_lengths if x >= 0.30])

    # Normalize
    tray_count = 4

    tcr = (total_core / tray_count) * 100
    rqd = (rqd_core / tray_count) * 100
    scr = (scr_core / tray_count) * 100

    # Clamp
    tcr = min(tcr, 100)
    rqd = min(rqd, 100)
    scr = min(scr, 100)

    # Sonuç
    st.success(f"Detected Core Segments: {len(segment_lengths)}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
