from ultralytics import YOLO

# Model yükle

model = YOLO("best_core_model.pt")

# Tahmin yap

results = model.predict(
source="test.jpg",
conf=0.05,
save=True
)

print("Analiz tamamlandı.")
