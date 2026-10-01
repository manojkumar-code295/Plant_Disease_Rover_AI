# Precision Agriculture Smart Rover - AI Chili Disease Detection

A high-performance, real-time edge AI system designed for Raspberry Pi 4 Model B rovers to detect and diagnose plant leaf diseases in chili crops, reject non-leaf objects (faces, hands, room backgrounds), and stream live telemetry to a cloud dashboard on Render.

---

## 1. Supported Disease Classes & Healthy Detection

The system is trained and calibrated specifically for chili plant health:

| Classification | Category | Treatment / Action |
| :--- | :--- | :--- |
| **Bacterial Spot** | Disease | Copper oxychloride bactericide; avoid overhead watering. |
| **Cercospora Leaf Spot** | Disease | Mancozeb / copper-based fungicide; space plants for airflow. |
| **Curl Virus** | Disease | Imidacloprid / Neem oil vector control; isolate stunted plants. |
| **Healthy Leaf** | Healthy | Foliage is vigorous; maintain routine monitoring. |
| **NO LEAF DETECTED** | State | No plant tissue in view; faces, hands, walls, objects ignored. |
| **Uncertain** | State | Valid leaf detected, but confidence is below threshold (< 50%). |

---

## 2. Multi-Stage Leaf-Only Filtering Pipeline

To prevent false disease predictions on human faces, hands, room walls, and furniture, the system applies a multi-stage botanical filter before model inference:

1. **Color Normalization & Orientation**:
   - Camera frames from Picamera2 are converted from native `RGB888` to OpenCV `BGR`.
   - Rotated 180° (`ROTATE_CAMERA_180 = True`) to match chassis mount orientation.
2. **Plant Foliage Mask**:
   - HSV foliage mask: `Hue: [22, 95]`, `Saturation: [45, 255]`, `Value: [25, 250]`.
   - Rejects low-saturation walls, white ceilings, and grey surfaces ($S < 45$).
3. **Human Skin Tone Suppression**:
   - Skin mask: `Hue: [0, 18]` and `[165, 180]`, `Saturation: [25, 180]`.
   - If skin tone exceeds 14% of the bounding box or exceeds 40% of foliage pixels, the region is **rejected** as a face or hand.
4. **Botanical Vegetation Index (Excess Green)**:
   - Computes $ExG = 2 \times G - R - B$.
   - Real leaves have $ExG \ge 14$ and Saturation $\ge 42$.
   - Walls ($ExG \approx 6$), human skin ($ExG < -20$), and clothing are **rejected**.
5. **Texture & Surface Variance**:
   - Computes grayscale Laplacian variance: $\text{Var}(\nabla^2 I) \ge 20$.
   - Rejects smooth painted walls, flat ceilings, and glossy furniture.
6. **Geometry & Scale Bounds**:
   - $3,500 \le \text{Area} \le 70\%$ of total frame.
   - Bounding boxes cannot simultaneously span $\ge 85\%$ of width and height (rejects full-frame background walls).
7. **Inference Execution**:
   - **Only** when a candidate passes all checks does the model run.
   - If no candidate qualifies, display outputs **`NO LEAF DETECTED`** and model inference is bypassed.

---

## 3. Automatic Startup on Raspberry Pi Boot (Systemd)

The AI application runs automatically on boot via a dedicated systemd service.

### Service Details
* **Service Name**: `agrirover-ai.service`
* **User Service Unit**: `~/.config/systemd/user/agrirover-ai.service`
* **System-wide Unit Template**: `/home/agrirover/Desktop/chili_disease_ai/agrirover-ai.service`
* **Python Executable**: `/home/agrirover/Desktop/chili_disease_ai/ai_env/bin/python`
* **Script**: `/home/agrirover/Desktop/chili_disease_ai/complete_ai.py`
* **Restart Policy**: Automatically restarts on crashes (`Restart=always`, `RestartSec=5`).
* **Safe Initialization**: Motors and spray pump are explicitly locked in the **OFF** state at startup.

### Useful Management Commands

| Action | User Service Command (Default) | System-Wide Command |
| :--- | :--- | :--- |
| **Check Status** | `systemctl --user status agrirover-ai.service` | `sudo systemctl status agrirover-ai.service` |
| **View Live Logs** | `journalctl --user -u agrirover-ai.service -f` | `journalctl -u agrirover-ai.service -f` |
| **Restart Service** | `systemctl --user restart agrirover-ai.service` | `sudo systemctl restart agrirover-ai.service` |
| **Stop Service** | `systemctl --user stop agrirover-ai.service` | `sudo systemctl stop agrirover-ai.service` |
| **Start Service** | `systemctl --user start agrirover-ai.service` | `sudo systemctl start agrirover-ai.service` |
| **Enable at Boot** | `systemctl --user enable agrirover-ai.service` | `sudo systemctl enable agrirover-ai.service` |

---

## 4. Configuration & Thresholds

All thresholds are centralized at the top of [`complete_ai.py`](file:///home/agrirover/Desktop/chili_disease_ai/complete_ai.py) and can also be overridden via environment variables or a local `.env` file:

```python
# Confidence threshold for disease / healthy declaration
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.50))

# Leaf contour area limits (pixels)
MIN_LEAF_AREA = int(os.environ.get("MIN_LEAF_AREA", 3500))
MAX_LEAF_AREA_RATIO = float(os.environ.get("MAX_LEAF_AREA_RATIO", 0.70))

# Foliage density inside bounding box (minimum 22%)
MIN_LEAF_DENSITY = float(os.environ.get("MIN_LEAF_DENSITY", 0.22))

# Human skin tone suppression threshold (maximum 14%)
MAX_SKIN_RATIO = float(os.environ.get("MAX_SKIN_RATIO", 0.14))

# Surface texture variance (Laplacian)
MIN_TEXTURE_VAR = float(os.environ.get("MIN_TEXTURE_VAR", 20.0))

# Chassis camera mounting rotation (180 degrees)
ROTATE_CAMERA_180 = os.environ.get("ROTATE_CAMERA_180", "1") == "1"

# Upload rate limit (seconds between cloud syncs)
UPLOAD_INTERVAL = float(os.environ.get("UPLOAD_INTERVAL", 2.0))
```

---

## 5. Hardware GPIO Pin Mapping (Raspberry Pi 4 Model B)

| Subsystem | Signal Name | BCM GPIO | Physical Board Pin | Function |
| :--- | :--- | :--- | :--- | :--- |
| **Left Motor (L298N Channel A)** | `LEFT_IN1` | `GPIO 17` | Pin 11 | Forward Direction |
| | `LEFT_IN2` | `GPIO 27` | Pin 13 | Reverse Direction |
| | `LEFT_EN` | `GPIO 12` | Pin 32 | Speed PWM0 |
| **Right Motor (L298N Channel B)** | `RIGHT_IN1` | `GPIO 22` | Pin 15 | Forward Direction |
| | `RIGHT_IN2` | `GPIO 23` | Pin 16 | Reverse Direction |
| | `RIGHT_EN` | `GPIO 13` | Pin 33 | Speed PWM1 |
| **Pesticide Spray Pump** | `PUMP_PIN` | `GPIO 26` | Pin 37 | 5V Relay Trigger (IN) |
| **Ultrasonic Distance Sensor** | `TRIG_PIN` | `GPIO 24` | Pin 18 | Trigger (10 µs pulse) |
| | `ECHO_PIN` | `GPIO 25` | Pin 22 | Echo (via 1kΩ/2kΩ divider) |
| **Camera** | `CAM_PORT` | `CSI-2` | 15-Pin Ribbon | OV5647 / IMX708 |

---

## 6. Render Cloud Deployment & Security

* **Dashboard URL**: `https://precision-agriculture-smart-rover.onrender.com`
* **Telemetry Endpoints**:
  - `POST /api/update-result`: Uploads latest detected leaf image, disease, confidence, and treatment.
  - `POST /api/rover/heartbeat`: Sends rover keepalive signal every 5 seconds.
  - `GET /api/rover/command`: Polls steering and spray pump commands every 0.4 seconds.
  - `GET /api/rover/gpio-config`: Serves active BCM/board hardware pinout.
* **Environment Variables & Secrets**:
  - `RENDER_URL`: Base URL of the deployed web application.
  - `RENDER_API_TOKEN`: Optional authentication token passed via `Authorization: Bearer <TOKEN>`.
  - Secrets must **never** be committed to Git. `.env` and `*.env` are excluded via `.gitignore`.
