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

    core_count = 0

    total_core = 0
    rqd_core = 0
    scr_core = 0

    for box, cls in zip(boxes, classes):

        label = names[int(cls)]

        if label == "core_segment":

            core_count += 1

            x1, y1, x2, y2 = box

            core_length = (x2 - x1) / image.width

            total_core += core_length

            if core_length >= 0.03:
                rqd_core += core_length

            if core_length >= 0.08:
                scr_core += core_length

    rqd = rqd_core * 100
    tcr = total_core * 100
    scr = scr_core * 100

    if rqd > 100:
        rqd = 100

    if tcr > 100:
        tcr = 100

    if scr > 100:
        scr = 100

    st.success(f"Detected Core Segments: {core_count}")

    st.metric("RQD", f"{rqd:.2f}%")
    st.metric("TCR", f"{tcr:.2f}%")
    st.metric("SCR", f"{scr:.2f}%")
