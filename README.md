# 🏆 AI Multi-Sport Action Analysis System (YOLO11 Powered)

## Overview
This project is an end-to-end AI-based sports analytics system designed to perform real-time player detection, tracking, and action analysis.  
It supports multiple sports such as **Football** and **Kabaddi**, providing intelligent insights that can assist referees and coaches in understanding player movements and game intensity.

The system leverages the latest **YOLO11 architecture** for fast and efficient object detection, making it suitable for real-time applications even on CPU-based systems.

---

## 🚀 Key Features

- ✅ **YOLO11 Player Detection** – High-speed, low-latency object detection  
- ✅ **Multi-Object Tracking** – Persistent player IDs using ByteTrack  
- ✅ **Action Intelligence** – Speed-based movement analysis to identify attacking or defensive behavior  
- ✅ **Multi-Sport Support** – Adaptable logic for Football and Kabaddi  
- ✅ **Live Camera Integration** – Real-time analytics from webcam  
- ✅ **Video Upload System** – Analyze recorded matches  
- ✅ **Streamlit Frontend** – Interactive and user-friendly interface  
- ✅ **Modular Architecture** – Scalable and production-style code structure  

---

## 🧠 System Architecture

**Frontend:** Streamlit  
**Backend:** Python  
**AI Engine:** YOLO11 + ByteTrack Tracking  


---

## ⚙️ Technologies Used

- Python  
- Ultralytics YOLO11  
- OpenCV  
- PyTorch  
- Streamlit  

---

## 📌 Applications

- AI-assisted sports officiating  
- Player performance analysis  
- Tactical movement insights  
- Coaching support  
- Sports analytics research  

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
cd src
streamlit run app.py
