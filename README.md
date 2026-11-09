# AgriVision AI: Live Plant Leaf Detection & Automatic Water Spraying System

An upgraded, production-ready AI agricultural system that integrates:
1. **Continuous Live Webcam Detection** (Video frames continuously analyzed without manual photo capture).
2. **Dual-Model Computer Vision**:
   - **YOLOv8 Object Detection**: Localizes plant leaves and identifies obstacles in front of the rover.
   - **Strict Leaf Verification CNN**: Filters out non-leaf objects (faces, animals, vehicles, buildings, fruits).
   - **Deep Plant Disease CNN**: Accurately classifies into **Healthy Leaf**, **Cercospora**, **Anthracnose**, or **Bacterial Spot**.
3. **Live VS Code Terminal Logging**: Formatted real-time telemetry printed directly in your terminal.
4. **Automatic Water Spraying System (WATER ONLY)**: Precision spraying activated upon disease verification with configurable duration, cooldown intervals, and safety cutoffs.
5. **Differential Drive Motor Navigation**: Multi-speed PWM throttle with automatic obstacle braking and clearance resumption.
6. **Automatic Image Archiving & CSV History**: Saves detected leaf photos into organized folders and appends diagnostic records into a CSV log.
7. **Interactive Web Dashboard**: Features real-time bounding boxes drawn on video, live telemetry, and the new **LIVE MONITORING AND HARDWARE CONTROL** settings panel.

---

## 📁 Project Directory Structure

```text
Plant_Leaf_Disease_Detection/
│
├── detected_images/                 # Automatically stored leaf detection images
│   ├── Healthy/
│   ├── Cercospora/
│   ├── Anthracnose/
│   └── Bacterial_Spot/
│
├── detection_logs/                  # Detection history & system audit logs
│   ├── detection_history.csv        # Detailed CSV record of every detection
│   └── system_logs.txt              # Real-time event log
│
├── models/
│   ├── plant_disease_cnn.pth        # Trained Disease CNN model weights
│   ├── leaf_verifier_cnn.pth        # Trained Leaf Verifier CNN weights
│   ├── yolov8n.pt                   # Local YOLOv8 Object Detection model
│   ├── disease_classes.json         # Disease class labels
│   ├── verifier_classes.json        # Verifier labels
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── config.py                    # Hardware pinout, thresholds, and limits
│   ├── hardware_controller.py       # Motor driver & water pump safety controller
│   ├── object_detector.py           # YOLOv8 object & obstacle detection engine
│   ├── leaf_verifier.py             # Strict Leaf-Only verification filter
│   ├── disease_classifier.py        # 4-class disease classification CNN
│   ├── terminal_logger.py           # Real-time formatted terminal logger
│   ├── image_logger.py              # Automatic image archiving & CSV history
│   └── pipeline.py                  # Integrated multi-stage inference pipeline
│
├── templates/
│   └── index.html                   # Interactive web dashboard & settings panel
│
├── static/
│   ├── css/style.css                # Modern responsive UI & HUD styling
│   └── js/app.js                    # Continuous stream controller & canvas renderer
│
├── app.py                           # Flask backend web application
├── test_project_upgrade.py          # Complete system integration test suite
├── run_tests.py                     # 10-point offline AI test suite
├── requirements.txt                 # Dependencies
└── README.md                        # Complete project documentation
```

---

## ⚡ How to Run in Visual Studio Code (VS Code)

### Step 1: Open the Project in VS Code
Open your VS Code terminal and navigate to the project directory:
```powershell
cd "C:\Users\B Siddarth\Desktop\Plant_Leaf_Disease_Detection"
code .
```

### Step 2: Install Dependencies
Ensure all required libraries are installed:
```powershell
pip install -r requirements.txt
```

### Step 3: Start the Complete System
Run `app.py` directly inside your VS Code integrated terminal:
```powershell
python app.py
```
You will immediately see the initialization banner:
```text
==================================================================
🌿 AgriVision AI: Plant Leaf Detection & Smart Rover System
   Hardware Mode: SIMULATION
   VS Code Terminal Active - Real-Time Logs Will Appear Below
   Open Web Dashboard at: http://127.0.0.1:5000
==================================================================
```

### Step 4: Open the Web Dashboard
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🖥️ Live VS Code Terminal Output

As the live webcam stream runs, the system prints structured diagnostic logs directly to your VS Code terminal (with duplicate suppression):

```text
[CAMERA] Webcam Connected
[OBJECT DETECTION] Leaf Detected
[LEAF VERIFICATION] Valid Plant Leaf
[DISEASE DETECTION] Cercospora Detected
[CONFIDENCE] 94.2%
[MOTOR CONTROL] Speed Updated (65% PWM)
[WATER PUMP] Activated (Water Only)
[SPRAYING] Started
[OBJECT DETECTION] No Leaf Detected
[WATER PUMP] OFF
[MOTOR CONTROL] Standby
```

When an obstacle (person, chair, pet, obstruction) is in the rover's path:
```text
[OBJECT DETECTION] Obstacle Detected: person (Conf: 89.0%)
[MOTOR CONTROL] Obstacle in path -> Stopped!
```

---

## ⚙️ Web Dashboard Settings: LIVE MONITORING AND HARDWARE CONTROL

The dashboard includes a dedicated configuration section:

### A. Camera Settings
* **Select Webcam**: Switch between built-in webcam (`0`) and external USB camera (`1`).
* **Start / Pause / Stop**: Full continuous stream controls.
* **Live Video Preview**: Real-time video with transparent bounding box canvas.

### B. Detection Settings
* **Enable/Disable YOLO Object Detection**: Toggle object and obstacle localization.
* **Enable/Disable Disease Detection**: Toggle deep disease classification CNN.
* **Confidence Threshold Slider**: Adjust sensitivity (50% – 95%, default 70%).

### C. Motor Settings
* **Control Mode**: Toggle between **Manual Mode** and **Autonomous Mode**.
* **Speed Presets**: **Low (35%)**, **Medium (65%)**, **High (100%)**, or custom slider.
* **Obstacle Braking**: Rover automatically halts if an obstacle enters its path and resumes when cleared.
* **Emergency Stop**: Immediately disables all motor PWM and pump relay outputs.

### D. Water Pump Settings (WATER ONLY)
* **Enable/Disable Pump**: Master safety switch.
* **Automatic Spraying Mode**: Auto-triggers water spray when an infected leaf is verified.
* **Spray Duration**: Configurable spray burst time (0.5s – 5.0s, default 2.0s).
* **Spray Interval**: Minimum cooldown interval between bursts (default 4.0s).
* **Max Continuous Runtime Guard**: Safety cutoff prevents pump from running longer than 6.0s continuously.

### E. Image Storage Settings
* **Auto-Save Detected Images**: Automatically archives verified leaf frames to `detected_images/<Disease>/`.
* **Save Diseased / Healthy Toggles**: Selectively archive only infected or all leaves.
* **Duplicate Suppression**: Minimum 3.5s interval between saves of the same category to prevent disk flooding.
* **CSV History**: Appends date, time, disease, confidence, spray status, and motor speed to `detection_logs/detection_history.csv`.

---

## 🍓 Transferring to Raspberry Pi for Field Testing

The exact same project folder runs on a Raspberry Pi without changing code:

1. **Copy Folder to Raspberry Pi**:
   ```bash
   scp -r "C:\Users\B Siddarth\Desktop\Plant_Leaf_Disease_Detection" pi@raspberrypi.local:/home/pi/
   ```
2. **Install Raspberry Pi Dependencies**:
   ```bash
   cd /home/pi/Plant_Leaf_Disease_Detection
   pip3 install -r requirements.txt RPi.GPIO
   python3 app.py
   ```
3. **Automatic Hardware Detection**:
   The controller will automatically detect the Raspberry Pi architecture and activate hardware GPIO pin control with PWM:
   * Left Motor: `ENA = GPIO 12 (PWM)`, `IN1 = GPIO 17`, `IN2 = GPIO 27`
   * Right Motor: `ENB = GPIO 13 (PWM)`, `IN3 = GPIO 22`, `IN4 = GPIO 23`
   * Water Pump Relay: `GPIO 24`
   * Status LED / Buzzer: `GPIO 25`

---

## 🧪 Automated Testing

To run the complete system verification test suite:
```powershell
python test_project_upgrade.py
```
Expected output:
```text
======================================================================
  VERIFYING UPGRADED LIVE PLANT DETECTION & SMART SPRAYING SYSTEM
======================================================================
[1/7] Testing System Settings API (/api/settings)...           -> PASS
[2/7] Testing Live Frame Continuous Inference...                -> PASS
[3/7] Testing Strict Non-Leaf Rejection (Human Face)...         -> PASS
[4/7] Testing Obstacle Detection & Auto-Braking...              -> PASS
[5/7] Testing Automatic Image Storage & CSV History Log...      -> PASS
[6/7] Testing Rover Motor Controls...                           -> PASS
[7/7] Testing Emergency Stop Cutoff...                          -> PASS
======================================================================
  ALL UPGRADE TESTS PASSED WITH 100% SUCCESS! 🎉
======================================================================
```
