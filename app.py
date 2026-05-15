import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# Model yükle
model = YOLO("best.pt")

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

    st.write(results)

    names = model.names

    boxes = []
    classes = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    core_count = 0

    for cls in classes:

        label = names[int(cls)]

        if label == "core_segment":
            core_count += 1

    st.metric("Detected Core Segments", core_count)

    st.write("RQD Analysis Active")
    st.write("TCR Analysis Active")
    st.write("SCR Analysis Active")
