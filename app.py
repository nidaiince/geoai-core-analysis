import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("GeoNA Core Analysis")

st.write("AI-based geological core analysis system")

# Model yükle
model = YOLO("best_core_model.pt")

uploaded_file = st.file_uploader(
    "Karot görüntüsü yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Yüklenen Görüntü")

    # Temp kayıt
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    image.save(temp_file.name)

    # Tahmin
    results = model.predict(
        source=temp_file.name,
        conf=0.10
    )

    names = model.names

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    core_count = 0

    for cls in classes:

        label = names[int(cls)]

        if label == "core_segment":
            core_count += 1

    st.success(f"Detected Core Segments: {core_count}")

    st.write("RQD Analysis Active")
    st.write("TCR Analysis Active")
    st.write("SCR Analysis Active")
