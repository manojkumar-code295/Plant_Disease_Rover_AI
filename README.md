# AgriVision AI: Live Plant Leaf Detection & Automatic Smart Spraying System

An upgraded, production-grade AI precision agriculture and smart rover system featuring:
1. **Continuous Live Webcam Detection**: Real-time video frame stream processing without requiring manual snapshot clicks.
2. **Dual-Model Computer Vision Pipeline**:
   - **YOLOv8 Object Detection**: High-speed localization of plant leaves and dynamic obstacle detection in front of the rover.
   - **Strict Leaf-Only Verification CNN**: Binary classifier rejecting non-leaf objects (faces, hands, vehicles, furniture, room walls).
   - **Deep Plant Disease CNN**: Accurate 4-class disease diagnostic (**Healthy Leaf**, **Cercospora**, **Anthracnose**, **Bacterial Spot**).
3. **Live VS Code Terminal Logging**: Formatted real-time telemetry streaming cleanly in the terminal with duplicate suppression.
4. **Automatic Water Spraying System (WATER ONLY)**: Precision spraying activated upon disease verification with configurable spray duration, cooldown intervals, and safety cutoffs.
5. **Differential Drive Motor Speed Control**: Low, Medium, and High PWM throttle speeds with automatic obstacle braking and clear-path resumption.
6. **Automatic Image Archiving & CSV History**: Automatic saving into `detected_images/<Disease>/` and appending audit logs to `detection_logs/detection_history.csv`.
7. **Interactive Web Dashboard**: Video HUD with dynamic canvas bounding boxes, live telemetry indicators, rover D-pad, manual water spray toggle, and settings panel.
8. **Render Cloud Deployment**: Live 24/7 web service running at `https://precision-agriculture-smart-rover.onrender.com`.

---

## 📁 Project Directory Structure

```text
Plant_Leaf_Disease_Detection/
│
├── detected_images/                 # Stored leaf detection images
│   ├── Healthy/
│   ├── Cercospora/
│   ├── Anthracnose/
│   └── Bacterial_Spot/
│
├── detection_logs/                  # Detection history & audit records
│   ├── detection_history.csv        # Detailed CSV history of every detection
│   └── system_logs.txt              # Timestamped system events
│
├── models/
│   ├── plant_disease_cnn.pth        # Trained 4-Class Disease CNN (PyTorch)
│   ├── leaf_verifier_cnn.pth        # Trained Leaf Verifier CNN (PyTorch)
│   ├── yolov8n.pt                   # Local YOLOv8 Nano model weights
│   ├── disease_classes.json         # Disease category labels
│   ├── verifier_classes.json        # Binary verifier labels
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── config.py                    # Pinouts, thresholds, and limits
│   ├── hardware_controller.py       # Motor PWM & water pump relay controller
│   ├── object_detector.py           # YOLOv8 leaf & obstacle detection
│   ├── leaf_verifier.py             # Strict Leaf-Only verification filter
│   ├── disease_classifier.py        # 4-class disease classifier & treatments
│   ├── terminal_logger.py           # Real-time formatted terminal logger
│   ├── image_logger.py              # Automatic image archiving & CSV history
│   ├── pi_agent.py                  # Edge agent for Raspberry Pi camera & GPIO
│   └── pipeline.py                  # Integrated multi-stage inference pipeline
│
├── templates/
│   └── index.html                   # Live Web Dashboard HUD & Control Panel
│
├── static/
│   ├── css/style.css                # Modern responsive UI & HUD styling
│   └── js/app.js                    # Continuous stream controller & canvas renderer
│
├── website/                         # Render Cloud Deployment Service
│   ├── app.py                       # Cloud Flask server & telemetry aggregator
│   ├── requirements.txt             # Lightweight cloud dependencies
│   ├── templates/                   # Cloud web pages
│   └── static/                      # Cloud static assets
│
├── app.py                           # Full local Flask web application
├── test_project_upgrade.py          # Complete system integration test suite
├── run_tests.py                     # Offline AI verification test suite
├── requirements.txt                 # Python dependencies
├── Procfile                         # Render process specification
├── render.yaml                      # Render Blueprint configuration
└── README.md                        # Documentation
```

---

## 🌿 Supported Disease Classes & Treatments

| Classification | Category | Recommended Treatment (Water + Organic Soluble) |
| :--- | :--- | :--- |
| **Healthy Leaf** | Healthy | Plant foliage is vigorous; continue routine irrigation. |
| **Cercospora** | Disease | Water spray with organic copper-based fungicide; improve plant spacing. |
| **Anthracnose** | Disease | Water spray with neem extract or bio-fungicide; remove fallen debris. |
| **Bacterial Spot** | Disease | Water spray with copper bactericide; avoid overhead high-pressure watering. |
| **NO LEAF DETECTED** | Rejection | No identifiable leaf in frame; non-plant objects ignored. |

---

## ⚙️ Hardware GPIO Pin Mapping (Raspberry Pi 4 Model B)

| Subsystem | Signal Name | BCM GPIO | Physical Board Pin | Function |
| :--- | :--- | :--- | :--- | :--- |
| **Left Motor (Channel A)** | `LEFT_IN1` | `GPIO 17` | Pin 11 | Forward Direction |
| | `LEFT_IN2` | `GPIO 27` | Pin 13 | Reverse Direction |
| | `LEFT_EN` | `GPIO 12` | Pin 32 | Speed PWM0 |
| **Right Motor (Channel B)** | `RIGHT_IN1` | `GPIO 22` | Pin 15 | Forward Direction |
| | `RIGHT_IN2` | `GPIO 23` | Pin 16 | Reverse Direction |
| | `RIGHT_EN` | `GPIO 13` | Pin 33 | Speed PWM1 |
| **Water Spray Pump** | `PUMP_PIN` | `GPIO 26` | Pin 37 | 5V Relay Trigger (Water Only) |
| **Ultrasonic Distance Sensor** | `TRIG_PIN` | `GPIO 24` | Pin 18 | Trigger (10 µs pulse) |
| | `ECHO_PIN` | `GPIO 25` | Pin 22 | Echo (1kΩ/2kΩ divider) |
| **Camera** | `CAM_PORT` | `CSI-2` / USB | Ribbon / USB 3.0 | 30 FPS Live Stream |

---

## ⚡ How to Run in Visual Studio Code (VS Code)

### Step 1: Open the Project in VS Code
Open your VS Code terminal and navigate to the project directory:
```powershell
cd "C:\Users\B Siddarth\Desktop\Plant_Leaf_Disease_Detection"
```

### Step 2: Run the Application
```powershell
python app.py
```
Visit the local dashboard at:
👉 **`http://127.0.0.1:5000`**

### Step 3: Run the Edge Client on Raspberry Pi (or Laptop Edge Mode)
```bash
python src/pi_agent.py
```

---

## 🌐 Render Cloud Deployment

* **Live Cloud URL**: `https://precision-agriculture-smart-rover.onrender.com`
* **Health Check**: `GET https://precision-agriculture-smart-rover.onrender.com/api/health`
* **Rover Command Queue**: `GET https://precision-agriculture-smart-rover.onrender.com/api/rover/command`
* **Telemetry Sync**: `POST https://precision-agriculture-smart-rover.onrender.com/api/rover/heartbeat`

---

## 🧪 Automated Test Verification

Run the comprehensive integration test suite anytime:
```powershell
python test_project_upgrade.py
```
All 7 subsystem integration tests pass with 100% success!
