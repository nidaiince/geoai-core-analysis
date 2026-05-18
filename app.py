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

# Eğer görüntü yüklendiyse
if uploaded_file is not None:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Görsel göster
    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Model tahmini
    results = model(temp_path)

    # Class isimleri
    names = model.names

    # Detection listeleri
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

    # Görsel genişliği
    image_width = image.width

    # Ortalama tray genişliği
    tray_width = image_width * 0.22

    # Detection loop
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        # Sadece core segment kullan
        if label == "core_segment":

            core_count += 1

            x1, y1, x2, y2 = box

            # Segment uzunluğu oranı
            ratio = (x2 - x1) / tray_width

            # TCR
            total_core += ratio

            # RQD (10 cm üstü)
            if ratio >= 0.10:
                rqd_core += ratio

            # SCR (30 cm üstü)
            if ratio >= 0.30:
                scr_core += ratio

    # Yüzdelere çevir
    tcr = total_core * 100
    rqd = rqd_core * 100
    scr = scr_core * 100

    # Maksimum 100
    tcr = min(tcr, 100)
    rqd = min(rqd, 100)
    scr = min(scr, 100)

    # Sonuçlar
    st.success(f"Detected Core Segments: {core_count}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
