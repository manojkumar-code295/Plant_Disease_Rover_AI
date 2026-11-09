# Plant Disease AI: Live Crop Pathology & Smart Rover Platform

A production-grade Precision Agriculture and Smart Rover system featuring:
1. **Unified 12-Section Agricultural Dashboard**: Complete web interface with executive summary, live camera feeds, 3D rover telemetry, disease diagnosis, DC motor controls, water pump regulations, live sensor telemetry, detection history, categorized image gallery, model performance metrics, system settings, and cloud health monitoring.
2. **Dual-Model Computer Vision AI Pipeline**:
   - **YOLOv8 Object Detection**: High-speed localization of plant leaves and dynamic obstacle detection in front of the rover.
   - **Strict Leaf-Only Verification CNN**: MobileNetV2 binary classifier rejecting non-leaf objects (faces, hands, vehicles, animals, buildings, fruits).
   - **Deep Plant Disease CNN**: Accurate 4-class disease diagnostic (**Healthy Leaf**, **Cercospora**, **Anthracnose**, **Bacterial Spot**).
3. **Live Camera Monitoring with Fallback Webcam**:
   - High-definition live video relay from Raspberry Pi Rover camera.
   - Seamless fallback browser webcam access with HTML5 Canvas bounding box rendering.
   - Start, pause, resume, stop, and snapshot capture capabilities.
4. **Autonomous Edge-Cloud Bridge**:
   - Bi-directional synchronization between Render Cloud backend and Raspberry Pi Edge Node.
   - Real-time command queue (`/api/rover/command`), heartbeat sync (`/api/rover/heartbeat`), and video streaming.
   - Heartbeat watchdog with live status indicator: `CONNECTED`, `DISCONNECTED`, and `RECONNECTING`.
5. **Differential Drive Motor Navigation**:
   - Dual BTS7960 / L298N motor driver support with PWM speed throttling (Low 35%, Medium 65%, High 100%).
   - Operator arming safety interlock switch and immediate Emergency Stop override.
   - Obstacle auto-braking (path clearance resumption).
6. **Automatic Precision Water Sprayer (WATER ONLY)**:
   - Pure water foliar spraying triggered upon disease diagnosis above confidence threshold.
   - Maximum runtime cutoff (6s), cooldown protection (4s), and emergency auto-shutoff.
   - Hardware confirmation logging: operations are only marked confirmed after the Raspberry Pi energizes the relay pin.
7. **Live Hardware Sensor Monitoring**:
   - Actual telemetry for Ultrasonic Distance, Obstacle Sensor, Soil Moisture, Ambient Temperature, Humidity, and Battery Power.
   - Honest reporting: displays **"Sensor Not Connected"** if a sensor is disconnected. Never generates fake readings.
8. **Persistent SQLite Database**:
   - Thread-safe WAL mode storage (`plants_data.db`) for detection history, confirmed spraying operations, and telemetry.
9. **Progressive Web App (PWA)**:
   - Installable on Android and Desktop with `manifest.json` ("Plant Disease AI").
   - Offline fallback page (`/offline.html`) and service worker caching.
   - Full-screen responsive layout with mobile bottom navigation bar.

---

## 📁 Repository Directory Structure

```text
Plant_Disease_Rover_AI/
│
├── detected_images/                 # Stored leaf detection images
│   ├── Healthy/
│   ├── Cercospora/
│   ├── Anthracnose/
│   └── Bacterial_Spot/
│
├── detection_logs/                  # Audit history logs
│   ├── detection_history.csv        # Detailed CSV history of detections
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
│   ├── db.py                        # Persistent SQLite database operations
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
│   ├── index.html                   # Main 12-Section Dashboard
│   ├── rover.html                   # Dedicated Mobile Touch Controller
│   ├── analytics.html               # Data Analytics & Model Performance
│   ├── dataset.html                 # Dataset & Image Archive Browser
│   ├── daily_data.html              # Daily Inspection Logs
│   ├── settings.html                # System Configuration
│   └── offline.html                 # PWA Offline Fallback
│
├── static/
│   ├── css/style.css                # Responsive UI & Dashboard Styling
│   ├── js/app.js                    # Client-side controller & real-time canvas
│   ├── js/three.min.js              # Three.js 3D WebGL Engine
│   ├── icons/                       # PWA Application Icons (48, 96, 144, 192, 512)
│   ├── manifest.json                # Web App Manifest ("Plant Disease AI")
│   └── service-worker.js            # PWA Service Worker
│
├── app.py                           # Master Unified Flask Web Server & API
├── db.py                            # SQLite WAL Data Layer
├── bts7960_render_controller.py     # Standalone High-Power BTS7960 Rover Controller
├── run_tests.py                     # Offline AI Verification Test Suite
├── test_rover_system.py             # End-to-End Rover & API Test Suite
├── requirements.txt                 # Pinned CPU dependencies (Render optimized)
├── Procfile                         # Gunicorn Web Process definition
├── render.yaml                      # Render Blueprint Configuration
└── README.md                        # Documentation
```

---

## 🌿 Supported Disease Classes & Treatments

| Classification | Category | Pathogen Details | Recommended Treatment (Water Only Protocol) |
| :--- | :--- | :--- | :--- |
| **Healthy Leaf** | Healthy | Optimal vigor | Vigorous foliage; continue standard irrigation. |
| **Cercospora** | Disease | *Cercospora spp.* | Foliar water spray with organic copper-based fungicide; improve row ventilation. |
| **Anthracnose** | Disease | *Colletotrichum spp.* | Water spray with neem seed extract or bio-fungicide; prune infected tissue. |
| **Bacterial Spot** | Disease | *Xanthomonas spp.* | Water spray with copper bactericide + bio-bactericide; avoid overhead high-pressure watering. |
| **NON-LEAF REJECTED** | Rejection | Faces, Animals, Objects | Rejected: *"Invalid Image! Please upload a clear plant leaf image."* |

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
| **Ultrasonic Sensor** | `TRIG_PIN` | `GPIO 20` | Pin 38 | Trigger Pulse (10 µs) |
| | `ECHO_PIN` | `GPIO 21` | Pin 40 | Echo Input (via 1kΩ/2kΩ divider) |
| **Camera** | `CAM_PORT` | `CSI-2` / USB | Ribbon / USB 3.0 | 30 FPS Live Stream |

---

## ⚡ How to Run Locally

### 1. Start the Unified Web Application
```powershell
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### 2. Run the Raspberry Pi Rover Agent
On the Raspberry Pi:
```bash
python src/pi_agent.py
# Or for the BTS7960 high-power controller:
python bts7960_render_controller.py
```

### 3. Run Test Suites
```powershell
# 1. AI Verifier & 4-Class CNN Test Suite:
python run_tests.py

# 2. Rover Navigation, Motors, Pump & API Test Suite:
python test_rover_system.py
```

---

## 📱 Progressive Web App (PWA) Mobile Installation

1. Open the website on your mobile device (Chrome for Android / Safari on iOS).
2. Tap the **"📲 Install App"** button in the header (or select **"Add to Home Screen"** in the browser menu).
3. The app installs as a standalone application named **"Plant Disease AI"**.
4. Features full offline capability with cached application views and an honest offline indicator for live hardware controls.

---

## 🌐 Cloud Deployment (Render)

- **Live URL**: `https://precision-agriculture-smart-rover.onrender.com`
- **Health Check**: `GET /api/health` (HTTP 200)
- **Status Overview**: `GET /api/status`
- **Rover Telemetry**: `GET /api/telemetry`
- **Sensors Endpoint**: `GET /api/sensor_data`
- **Command Queue**: `GET /api/rover/command`
- **Heartbeat Endpoint**: `POST /api/rover/heartbeat`
