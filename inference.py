from ultralytics import YOLO

# Model yükle
model = YOLO("best_core_model.pt")

# Tahmin yap
results = model.predict(
    source="newtest.jpeg",
    conf=0.10,
    save=True
)

# Class isimleri
names = model.names

# Boxları al
boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()

core_boxes = []

# Core segmentleri filtrele
for box, cls in zip(boxes, classes):

    label = names[int(cls)]

    if label == "core_segment":
        core_boxes.append(box)

# Satırlara ayır
rows = {}

for box in core_boxes:

    x1, y1, x2, y2 = box

    cy = (y1 + y2) / 2

    found = False

    for key in rows:

        if abs(cy - key) < 80:
            rows[key].append(box)
            found = True
            break

    if not found:
        rows[cy] = [box]

# Satırları sırala
sorted_rows = []

for key in sorted(rows.keys()):

    row = sorted(rows[key], key=lambda b: b[0])

    sorted_rows.append(row)

# Gerçek tray uzunluğu
tray_real_m = 1.0

run_no = 1

# Her satırı analiz et
for row in sorted_rows:

    print(f"\n===== RUN {run_no} =====")

    total_core = 0
    rqd_core = 0
    scr_core = 0
    # Tray genişliği
    min_x = min([b[0] for b in row])
    max_x = max([b[2] for b in row])

    tray_width_px = max_x - min_x

    for box in row:

        x1, y1, x2, y2 = box

        core_len = ((x2 - x1) / tray_width_px) * tray_real_m

        print(f"Core Length: {core_len:.2f} m")

        total_core += core_len

        if core_len >= 0.10:
            rqd_core += core_len
            if core_len >= 0.30:
    scr_core += core_len

    rqd = (rqd_core / tray_real_m) * 100
    tcr = (total_core / tray_real_m) * 100
    scr = (scr_core / tray_real_m) * 100

    if rqd > 100:
        rqd = 100

    if tcr > 100:
        tcr = 100
    
    if scr > 100:
    scr = 100
        

    print(f"\nRQD = %{rqd:.2f}")
   print(f"TCR = %{tcr:.2f}")
   print(f"SCR = %{scr:.2f}")
    print(f"Toplam Core = {total_core:.2f} m")

    run_no += 1
