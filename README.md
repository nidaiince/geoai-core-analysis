# GeoNA Core Analysis

AI-based geological core analysis system for automatic RQD/TCR/SCR estimation using computer vision and deep learning.

# Developer & Founder
GeoAI Core Analysis was developed and founded by Nida İNCE.

---

# Features

* Automatic core segmentation
* Divider detection
* Depth marker detection
* SPT marker detection
* Automatic RQD calculation
* Automatic TCR calculation
* Geological core image analysis
* YOLO-based instance segmentation
* Geological tray/run analysis

---

# Project Goal

This project aims to automate geological core logging operations using artificial intelligence.

The system detects:

* Core segments
* Tray dividers
* Depth markers
* SPT markers

and automatically calculates:

* RQD (Rock Quality Designation)
* TCR (Total Core Recovery)
* SCR (Solid Core Recovery)

Future versions will include:

* Lithology interpretation
* Fracture analysis
* Weathering estimation
* PDF report generation
* Mobile application support
* Cloud-based analysis

---

# Model Information

Model Type:

* YOLO Segmentation Model

Detected Classes:

* core_segment
* divider
* depth_marker
* spt_marker

---

# Installation

```bash
pip install -r requirements.txt
```

# Usage

```bash
python inference.py
```

---

# Example Output

```text
===== RUN 1 =====

Core Length: 0.84 m
Core Length: 0.97 m

RQD = %97.01
TCR = %97.01
```

---

# Repository Structure

```text
geoai-core-analysis/
│
├── inference.py
├── requirements.txt
├── best_core_model.pt
├── newtest.jpeg
├── train_results.zip
└── README.md
```

---

# Technologies Used

* Python
* YOLO
* Ultralytics
* OpenCV
* NumPy
* Deep Learning
* Computer Vision

---

# Future Roadmap

* [ ] Automatic SCR calculation
* [ ] Lithology AI interpretation
* [ ] FastAPI backend
* [ ] Flutter mobile application
* [ ] Web dashboard
* [ ] PDF reporting
* [ ] Multi-tray analysis
* [ ] Cloud deployment

---

# Developer & Founder

GeoAI Core Analysis was developed and founded by Nida İNCE.

