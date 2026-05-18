import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# Model yükle
model = YOLO("best (3).pt")

uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Görsel aç
    image = Image.open(uploaded_file).convert("RGB")

    # Görsel göster
    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Model tahmini
    results = model(temp_path)

    names = model.names

    boxes = []
    classes = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    core_count = 0

    total_core = 0
    rqd_core = 0
    scr_core = 0

    # Tray genişliği yaklaşık hesap
    tray_width = image.width * 0.22

    # Tespit edilen karotları işle
    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        if label == "core_segment":

            core_count += 1

            x1, y1, x2, y2 = box

            # Karot uzunluğu
            core_length = (x2 - x1) / tray_width

            total_core += core_length

            # RQD (>10 cm)
            if core_length >= 0.10:
                rqd_core += core_length

            # SCR (>30 cm)
            if core_length >= 0.30:
                scr_core += core_length

    # Toplam tray uzunluğu
    tray_total = 4.0

    # Hesaplamalar
    rqd = (rqd_core / tray_total) * 100
    tcr = (total_core / tray_total) * 100
    scr = (scr_core / tray_total) * 100

    # 100 üstüne çıkmasın
    rqd = min(rqd, 100)
    tcr = min(tcr, 100)
    scr = min(scr, 100)

    # Sonuçlar
    st.success(f"Detected Core Segments: {core_count}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
