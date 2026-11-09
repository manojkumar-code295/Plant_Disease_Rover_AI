/**
 * AgriVision AI - Client-Side Controller
 * Live Webcam Continuous Detection, YOLO Bounding Box Canvas Renderer,
 * Smart Rover Navigation, Precision Water Sprayer, and Settings Manager.
 */

document.addEventListener('DOMContentLoaded', () => {

  // Video & Viewport Elements
  const video = document.getElementById('webcam-video');
  const boxCanvas = document.getElementById('bounding-box-canvas');
  const captureCanvas = document.getElementById('capture-canvas');
  const boxCtx = boxCanvas ? boxCanvas.getContext('2d') : null;

  // Toolbar Controls
  const btnStartDetection = document.getElementById('btn-start-detection');
  const btnPauseDetection = document.getElementById('btn-pause-detection');
  const btnStopDetection = document.getElementById('btn-stop-detection');
  const btnToggleUpload = document.getElementById('btn-toggle-upload');
  const dropZoneDrawer = document.getElementById('drop-zone-drawer');
  const fileInput = document.getElementById('file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const dropZone = document.getElementById('drop-zone');

  // Real-time HUD Elements
  const hudFps = document.getElementById('hud-fps');
  const hudLatency = document.getElementById('hud-latency');
  const hudLeafStatus = document.getElementById('hud-leaf-status');
  const hudObsBadge = document.getElementById('hud-obs-badge');
  const hudDetectionPill = document.getElementById('hud-detection-pill');
  const hudStatusText = document.getElementById('hud-status-text');

  // Stats Counters
  const statLeaves = document.getElementById('stat-leaves-detected');
  const statHealthy = document.getElementById('stat-healthy-leaves');
  const statDiseased = document.getElementById('stat-diseased-leaves');
  const statImages = document.getElementById('stat-images-stored');
  const statMotorSpeed = document.getElementById('stat-motor-speed');
  const statPump = document.getElementById('stat-pump-status');

  // Rover & Pump Controls
  const dpadBtns = document.querySelectorAll('.dpad-btn');
  const speedSlider = document.getElementById('speed-slider');
  const speedDisplay = document.getElementById('speed-display');
  const speedPresets = document.querySelectorAll('.btn-preset');
  const btnPumpToggle = document.getElementById('btn-pump-toggle');
  const pumpToggleText = document.getElementById('pump-toggle-text');
  const pumpStatusPill = document.getElementById('pump-status-pill');
  const btnPumpBurst = document.getElementById('btn-pump-burst');
  const autoSprayToggle = document.getElementById('auto-spray-toggle');
  const btnEmergencyStop = document.getElementById('btn-emergency-stop');

  // Telemetry Elements
  const teleDir = document.getElementById('tele-dir');
  const teleSpeed = document.getElementById('tele-speed');
  const telePump = document.getElementById('tele-pump');
  const teleAuto = document.getElementById('tele-auto');
  const teleHardware = document.getElementById('tele-hardware');
  const platformBadge = document.getElementById('platform-badge');
  const navModeBadge = document.getElementById('nav-mode-badge');

  // Diagnosis Elements
  const emptyState = document.getElementById('empty-state');
  const rejectionBox = document.getElementById('rejection-box');
  const rejectionTitle = document.getElementById('rejection-title');
  const rejectionReason = document.getElementById('rejection-reason');
  const uncertainBox = document.getElementById('uncertain-box');
  const resultDetails = document.getElementById('result-details');
  const verifierText = document.getElementById('verifier-text');
  const healthPill = document.getElementById('health-pill');
  const predictionTitle = document.getElementById('prediction-title');
  const predictionPathogen = document.getElementById('prediction-pathogen');
  const confidenceVal = document.getElementById('confidence-val');
  const sprayTriggerBanner = document.getElementById('spray-trigger-banner');
  const sprayDurDisp = document.getElementById('spray-dur-disp');
  const probList = document.getElementById('prob-list');
  const diseaseDesc = document.getElementById('disease-desc');
  const symptomList = document.getElementById('symptom-list');
  const treatmentChemical = document.getElementById('treatment-chemical');
  const treatmentOrganic = document.getElementById('treatment-organic');
  const treatmentPrevention = document.getElementById('treatment-prevention');
  const presetsList = document.getElementById('presets-list');

  // Settings Panel Inputs
  const settingCameraSelect = document.getElementById('setting-camera-select');
  const btnSettingCamStart = document.getElementById('btn-setting-cam-start');
  const btnSettingCamStop = document.getElementById('btn-setting-cam-stop');
  const settingObjDetect = document.getElementById('setting-obj-detect');
  const settingDiseaseDetect = document.getElementById('setting-disease-detect');
  const settingConfSlider = document.getElementById('setting-conf-slider');
  const settingConfVal = document.getElementById('setting-conf-val');
  const btnMotorManual = document.getElementById('btn-motor-manual');
  const btnMotorAuto = document.getElementById('btn-motor-auto');
  const settingPumpMaster = document.getElementById('setting-pump-master');
  const settingAutoSprayToggle = document.getElementById('setting-auto-spray-toggle');
  const settingSprayDur = document.getElementById('setting-spray-dur');
  const settingSprayInt = document.getElementById('setting-spray-int');
  const settingStorageToggle = document.getElementById('setting-storage-toggle');
  const settingSaveDiseased = document.getElementById('setting-save-diseased');
  const settingSaveHealthy = document.getElementById('setting-save-healthy');

  // Metrics Modal
  const btnOpenMetrics = document.getElementById('btn-open-metrics');
  const btnCloseMetrics = document.getElementById('btn-close-metrics');
  const metricsModal = document.getElementById('metrics-modal');
  const modalBackdrop = document.getElementById('modal-backdrop');

  // Application State
  let mediaStream = null;
  let isDetectionActive = false;
  let isDetectionPaused = false;
  let isInferencing = false;
  let inferenceLoopId = null;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 0;
  let pumpActive = false;

  // ==========================================
  // 1. CONTINUOUS LIVE CAMERA & INFERENCE LOOP
  // ==========================================

  btnStartDetection.addEventListener('click', startDetection);
  if (btnSettingCamStart) btnSettingCamStart.addEventListener('click', startDetection);

  async function startDetection() {
    try {
      if (!mediaStream) {
        const deviceId = settingCameraSelect ? settingCameraSelect.value : undefined;
        const constraints = {
          video: deviceId && deviceId !== "0" ? { deviceId: { exact: deviceId } } : { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
        };
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = mediaStream;
      }

      isDetectionActive = true;
      isDetectionPaused = false;
      btnStartDetection.disabled = true;
      btnPauseDetection.disabled = false;
      btnStopDetection.disabled = false;
      btnPauseDetection.textContent = "⏸ Pause";

      updateSettingsBackend({ camera: { is_active: true, is_paused: false } });
      startContinuousInference();
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  }

  btnPauseDetection.addEventListener('click', () => {
    isDetectionPaused = !isDetectionPaused;
    btnPauseDetection.textContent = isDetectionPaused ? "▶ Resume" : "⏸ Pause";
    hudStatusText.textContent = isDetectionPaused ? "Detection Paused." : "Resuming live detection...";
    updateSettingsBackend({ camera: { is_paused: isDetectionPaused } });
  });

  btnStopDetection.addEventListener('click', stopDetection);
  if (btnSettingCamStop) btnSettingCamStop.addEventListener('click', stopDetection);

  function stopDetection() {
    isDetectionActive = false;
    isDetectionPaused = false;
    if (inferenceLoopId) {
      clearTimeout(inferenceLoopId);
      inferenceLoopId = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    video.srcObject = null;
    btnStartDetection.disabled = false;
    btnPauseDetection.disabled = true;
    btnStopDetection.disabled = true;
    btnPauseDetection.textContent = "⏸ Pause";

    // Clear bounding box canvas
    if (boxCtx) {
      boxCtx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);
    }

    hudFps.textContent = "FPS: --";
    hudLatency.textContent = "Latency: -- ms";
    hudStatusText.textContent = "Camera stopped.";
    hudLeafStatus.textContent = "Camera Stopped";
    hudObsBadge.style.display = 'none';

    updateSettingsBackend({ camera: { is_active: false, is_paused: false } });
  }

  function startContinuousInference() {
    if (!isDetectionActive) return;

    if (!isDetectionPaused && video.videoWidth > 0 && !isInferencing) {
      sendLiveFrameForInference();
    }

    // Schedule next frame check (every 280-320ms for smooth edge inference)
    inferenceLoopId = setTimeout(startContinuousInference, 300);
  }

  async function sendLiveFrameForInference() {
    if (!video.videoWidth || isInferencing) return;
    isInferencing = true;
    const startTime = performance.now();

    // Scale down for ultra-fast edge transfer
    captureCanvas.width = 320;
    captureCanvas.height = 240;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 320, 240);
    const b64 = captureCanvas.toDataURL('image/jpeg', 0.85);

    try {
      const response = await fetch('/api/predict_live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 })
      });
      const data = await response.json();
      const latency = Math.round(performance.now() - startTime);

      // Compute FPS
      frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
      }

      hudFps.textContent = `FPS: ${currentFps || 3}`;
      hudLatency.textContent = `Latency: ${latency}ms`;

      // Render Bounding Boxes & Results
      drawBoundingBoxes(data.bounding_boxes, data);
      renderDiagnosisReport(data);

      if (data.hardware_telemetry) {
        updateHardwareTelemetry(data.hardware_telemetry);
      }
      if (data.statistics) {
        updateStatsCounters(data.statistics);
      }

    } catch (e) {
      console.warn("Live frame inference error:", e);
    } finally {
      isInferencing = false;
    }
  }

  // ==========================================
  // 2. LIVE BOUNDING BOX CANVAS RENDERER
  // ==========================================

  function drawBoundingBoxes(boxes, data) {
    if (!boxCanvas || !boxCtx) return;

    const vWidth = video.videoWidth || 640;
    const vHeight = video.videoHeight || 480;

    boxCanvas.width = vWidth;
    boxCanvas.height = vHeight;
    boxCtx.clearRect(0, 0, vWidth, vHeight);

    // Scale factors from capture canvas (320x240) to video native
    const scaleX = vWidth / 320;
    const scaleY = vHeight / 240;

    // Obstacle HUD Badge
    if (data.has_obstacle) {
      hudObsBadge.style.display = 'block';
    } else {
      hudObsBadge.style.display = 'none';
    }

    if (!boxes || boxes.length === 0) return;

    boxes.forEach(obj => {
      const [bx1, by1, bx2, by2] = obj.box;
      const x = bx1 * scaleX;
      const y = by1 * scaleY;
      const w = (bx2 - bx1) * scaleX;
      const h = (by2 - by1) * scaleY;

      const isLeaf = obj.type === 'leaf';
      const isObstacle = obj.type === 'obstacle';

      // Colors
      let strokeColor = '#3b82f6';
      let fillColor = 'rgba(59, 130, 246, 0.15)';
      let labelText = `${obj.label} (${obj.confidence}%)`;

      if (isLeaf) {
        if (data.is_healthy) {
          strokeColor = '#10b981';
          fillColor = 'rgba(16, 185, 129, 0.18)';
          labelText = `🌿 Healthy Leaf (${data.confidence}%)`;
        } else {
          strokeColor = '#ef4444';
          fillColor = 'rgba(239, 68, 68, 0.20)';
          labelText = `🍂 ${data.display_name || data.prediction} (${data.confidence}%)`;
        }
      } else if (isObstacle) {
        strokeColor = '#f59e0b';
        fillColor = 'rgba(245, 158, 11, 0.20)';
        labelText = `🛑 Obstacle: ${obj.label}`;
      }

      // Draw Box
      boxCtx.strokeStyle = strokeColor;
      boxCtx.lineWidth = 3;
      boxCtx.strokeRect(x, y, w, h);

      // Draw Semi-transparent Fill
      boxCtx.fillStyle = fillColor;
      boxCtx.fillRect(x, y, w, h);

      // Draw Label Background
      boxCtx.font = 'bold 12px "JetBrains Mono", monospace';
      const textWidth = boxCtx.measureText(labelText).width;
      const textHeight = 18;

      boxCtx.fillStyle = strokeColor;
      boxCtx.fillRect(x, Math.max(0, y - textHeight - 4), textWidth + 12, textHeight + 4);

      // Draw Text
      boxCtx.fillStyle = '#ffffff';
      boxCtx.fillText(labelText, x + 6, Math.max(14, y - 6));
    });
  }

  // ==========================================
  // 3. RENDER DIAGNOSIS & REJECTION REPORT
  // ==========================================

  function renderDiagnosisReport(data) {
    emptyState.style.display = 'none';

    // 1. Non-Leaf Rejection
    if (!data.success || data.is_leaf === false) {
      rejectionBox.style.display = 'block';
      uncertainBox.style.display = 'none';
      resultDetails.style.display = 'none';

      rejectionTitle.textContent = data.message || "Invalid Image! Please upload a clear plant leaf image.";
      rejectionReason.textContent = data.verifier_details || "Strict leaf-only filter rejected non-leaf image.";

      hudLeafStatus.textContent = "Non-Leaf Rejected";
      hudDetectionPill.className = 'hud-detection-pill rejected';
      hudStatusText.textContent = "🚫 Invalid Image! Please upload a clear plant leaf image.";
      return;
    }

    // 2. Uncertain Prediction
    if (data.status === 'uncertain') {
      rejectionBox.style.display = 'none';
      uncertainBox.style.display = 'block';
      resultDetails.style.display = 'none';

      hudLeafStatus.textContent = "Uncertain Prediction";
      hudDetectionPill.className = 'hud-detection-pill';
      hudStatusText.textContent = "⚠️ Uncertain Prediction (< 70%)";
      return;
    }

    // 3. Valid Leaf Success
    rejectionBox.style.display = 'none';
    uncertainBox.style.display = 'none';
    resultDetails.style.display = 'block';

    if (data.is_healthy) {
      hudLeafStatus.textContent = "Healthy Leaf Verified";
      hudDetectionPill.className = 'hud-detection-pill verified';
      hudStatusText.textContent = `🌱 Verified: Healthy Leaf (${data.confidence}%)`;
    } else {
      hudLeafStatus.textContent = `${data.display_name} Detected`;
      hudDetectionPill.className = 'hud-detection-pill diseased';
      hudStatusText.textContent = `🍂 ${data.display_name} (${data.confidence}%)`;
    }

    verifierText.textContent = `Leaf Verified Successfully (Confidence: ${data.leaf_confidence || 99.8}%)`;

    if (data.is_healthy) {
      healthPill.className = 'health-pill healthy';
      healthPill.textContent = 'Healthy Plant';
    } else {
      healthPill.className = 'health-pill diseased';
      healthPill.textContent = 'Diseased Foliage';
    }

    predictionTitle.textContent = data.display_name || data.prediction;
    confidenceVal.textContent = `${data.confidence}%`;

    // Auto-Spray Banner
    if (data.auto_spray_trigger && data.auto_spray_trigger.triggered) {
      if (sprayDurDisp) sprayDurDisp.textContent = data.auto_spray_trigger.duration || 2.0;
      sprayTriggerBanner.style.display = 'flex';
      setTimeout(() => { sprayTriggerBanner.style.display = 'none'; }, 3000);
    } else {
      sprayTriggerBanner.style.display = 'none';
    }

    const info = data.disease_info || {};
    predictionPathogen.textContent = info.pathogen || '';
    diseaseDesc.textContent = info.description || '';

    // Symptoms
    symptomList.innerHTML = '';
    if (info.symptoms) {
      info.symptoms.forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        symptomList.appendChild(li);
      });
    }

    // Treatments
    const trt = info.treatments || {};
    treatmentChemical.textContent = trt.chemical || trt.preventive || "N/A";
    treatmentOrganic.textContent = trt.organic || trt.soil || "N/A";
    treatmentPrevention.textContent = trt.prevention || trt.monitoring || "N/A";

    // Probability Bars
    probList.innerHTML = '';
    const probs = data.all_probabilities || {};
    const colors = {
      'Healthy': '#10b981',
      'Cercospora': '#f59e0b',
      'Anthracnose': '#ef4444',
      'Bacterial_Spot': '#ec4899'
    };

    for (const [cls, pct] of Object.entries(probs)) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      const color = colors[cls] || '#3b82f6';
      row.innerHTML = `
        <div class="prob-header">
          <span>${cls.replace('_', ' ')}</span>
          <span>${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-bar" style="width: ${pct}%; background-color: ${color};"></div>
        </div>
      `;
      probList.appendChild(row);
    }
  }

  // ==========================================
  // 4. STATS COUNTERS & TELEMETRY UPDATES
  // ==========================================

  function updateStatsCounters(stats) {
    if (!stats) return;
    if (statLeaves) statLeaves.textContent = stats.total_leaves_detected || 0;
    if (statHealthy) statHealthy.textContent = stats.total_healthy_leaves || 0;
    if (statDiseased) statDiseased.textContent = stats.total_diseased_leaves || 0;
    if (statImages) statImages.textContent = stats.total_images_stored || 0;
  }

  function updateHardwareTelemetry(t) {
    if (!t) return;
    if (teleDir) teleDir.textContent = t.direction;
    if (teleSpeed) teleSpeed.textContent = `${t.speed}%`;
    if (statMotorSpeed) statMotorSpeed.textContent = `${t.speed}% PWM`;
    if (telePump) telePump.textContent = t.pump_active ? "SPRAYING" : "IDLE";
    if (statPump) {
      statPump.textContent = t.pump_active ? "SPRAYING WATER" : (t.pump_enabled ? "READY" : "DISABLED");
      statPump.className = t.pump_active ? "stat-value text-primary" : "stat-value";
    }
    if (teleAuto) teleAuto.textContent = t.auto_spray_enabled ? "ENABLED" : "OFF";
    if (teleHardware) teleHardware.textContent = t.hardware_mode.includes("HARDWARE") ? "Raspberry Pi (GPIO)" : "Simulation (Laptop)";

    if (platformBadge) {
      if (t.hardware_mode.includes("HARDWARE")) {
        platformBadge.innerHTML = '<span class="pulse-dot"></span> Raspberry Pi Hardware';
      } else {
        platformBadge.innerHTML = '<span class="pulse-dot"></span> Simulation Mode (Laptop)';
      }
    }

    if (navModeBadge) {
      navModeBadge.textContent = t.control_mode === 'automatic' ? 'Autonomous Mode' : 'Manual Driving';
    }

    pumpActive = t.pump_active;
    if (pumpStatusPill) {
      if (pumpActive) {
        pumpStatusPill.className = "pump-status-pill active";
        pumpStatusPill.textContent = "SPRAYING WATER";
        if (pumpToggleText) pumpToggleText.textContent = "Turn Pump OFF";
      } else {
        pumpStatusPill.className = "pump-status-pill";
        pumpStatusPill.textContent = "PUMP OFF";
        if (pumpToggleText) pumpToggleText.textContent = "Turn Pump ON";
      }
    }

    // Highlight D-Pad direction
    dpadBtns.forEach(btn => {
      const act = btn.getAttribute('data-action');
      if (act && act.toLowerCase() === t.direction.toLowerCase()) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  // ==========================================
  // 5. MOTOR & PUMP CONTROLS
  // ==========================================

  async function sendRoverCommand(action) {
    try {
      const res = await fetch('/api/rover/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action })
      });
      const data = await res.json();
      if (data.telemetry) updateHardwareTelemetry(data.telemetry);
    } catch (e) {
      console.warn("Rover control error:", e);
    }
  }

  dpadBtns.forEach(btn => {
    const action = btn.getAttribute('data-action');
    btn.addEventListener('click', () => sendRoverCommand(action));
  });

  // Speed Presets
  speedPresets.forEach(btn => {
    btn.addEventListener('click', () => {
      speedPresets.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = parseInt(btn.getAttribute('data-speed'));
      if (speedSlider) speedSlider.value = val;
      updateSpeed(val);
    });
  });

  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      updateSpeed(e.target.value);
    });
  }

  async function updateSpeed(val) {
    if (speedDisplay) speedDisplay.textContent = `${val}% PWM`;
    try {
      const res = await fetch('/api/rover/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: val })
      });
      const data = await res.json();
      if (data.telemetry) updateHardwareTelemetry(data.telemetry);
    } catch (e) {
      console.warn("Speed update error:", e);
    }
  }

  // Sprayer Pump Controls (Water Only)
  if (btnPumpToggle) {
    btnPumpToggle.addEventListener('click', async () => {
      const action = pumpActive ? 'off' : 'on';
      try {
        const res = await fetch('/api/rover/pump', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: action })
        });
        const data = await res.json();
        if (data.telemetry) updateHardwareTelemetry(data.telemetry);
      } catch (e) {
        console.warn("Pump toggle error:", e);
      }
    });
  }

  if (btnPumpBurst) {
    btnPumpBurst.addEventListener('click', async () => {
      try {
        btnPumpBurst.disabled = true;
        const dur = settingSprayDur ? parseFloat(settingSprayDur.value) : 2.0;
        const res = await fetch('/api/rover/pump', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'burst', duration: dur })
        });
        const data = await res.json();
        if (data.telemetry) updateHardwareTelemetry(data.telemetry);
        setTimeout(() => { btnPumpBurst.disabled = false; }, (dur + 0.5) * 1000);
      } catch (e) {
        btnPumpBurst.disabled = false;
      }
    });
  }

  if (autoSprayToggle) {
    autoSprayToggle.addEventListener('change', (e) => {
      if (settingAutoSprayToggle) settingAutoSprayToggle.checked = e.target.checked;
      updateSettingsBackend({ pump: { auto_spray_enabled: e.target.checked } });
    });
  }

  // Emergency Stop Button
  if (btnEmergencyStop) {
    btnEmergencyStop.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/rover/emergency_stop', { method: 'POST' });
        const data = await res.json();
        if (data.telemetry) updateHardwareTelemetry(data.telemetry);
        alert("EMERGENCY STOP EXECUTED: All motors and water pump outputs disabled.");
      } catch (e) {
        console.warn("Emergency stop error:", e);
      }
    });
  }

  // Keyboard Navigation
  const keyMap = {
    'KeyW': 'forward', 'ArrowUp': 'forward',
    'KeyS': 'backward', 'ArrowDown': 'backward',
    'KeyA': 'left', 'ArrowLeft': 'left',
    'KeyD': 'right', 'ArrowRight': 'right',
    'Space': 'stop'
  };

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    if (keyMap[e.code]) {
      e.preventDefault();
      sendRoverCommand(keyMap[e.code]);
    } else if (e.code === 'KeyP') {
      e.preventDefault();
      if (btnPumpToggle) btnPumpToggle.click();
    } else if (e.code === 'KeyB') {
      e.preventDefault();
      if (btnPumpBurst) btnPumpBurst.click();
    } else if (e.code === 'Digit1' && speedPresets[0]) {
      speedPresets[0].click();
    } else if (e.code === 'Digit2' && speedPresets[1]) {
      speedPresets[1].click();
    } else if (e.code === 'Digit3' && speedPresets[2]) {
      speedPresets[2].click();
    }
  });

  // ==========================================
  // 6. SYSTEM SETTINGS PANEL SYNC
  // ==========================================

  async function updateSettingsBackend(settingsPatch) {
    try {
      const res = await fetch('/api/settings/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsPatch)
      });
      const data = await res.json();
      if (data.telemetry) updateHardwareTelemetry(data.telemetry);
    } catch (e) {
      console.warn("Settings update error:", e);
    }
  }

  // Detection Settings
  if (settingObjDetect) {
    settingObjDetect.addEventListener('change', (e) => {
      updateSettingsBackend({ detection: { object_detection_enabled: e.target.checked } });
    });
  }

  if (settingDiseaseDetect) {
    settingDiseaseDetect.addEventListener('change', (e) => {
      updateSettingsBackend({ detection: { disease_detection_enabled: e.target.checked } });
    });
  }

  if (settingConfSlider) {
    settingConfSlider.addEventListener('input', (e) => {
      if (settingConfVal) settingConfVal.textContent = `${e.target.value}%`;
      updateSettingsBackend({ detection: { confidence_threshold: parseFloat(e.target.value) } });
    });
  }

  // Motor Mode Radio Pills
  if (btnMotorManual && btnMotorAuto) {
    btnMotorManual.addEventListener('click', () => {
      btnMotorManual.classList.add('active');
      btnMotorAuto.classList.remove('active');
      updateSettingsBackend({ motor: { control_mode: 'manual' } });
    });
    btnMotorAuto.addEventListener('click', () => {
      btnMotorAuto.classList.add('active');
      btnMotorManual.classList.remove('active');
      updateSettingsBackend({ motor: { control_mode: 'automatic' } });
    });
  }

  // Pump Settings
  if (settingPumpMaster) {
    settingPumpMaster.addEventListener('change', (e) => {
      updateSettingsBackend({ pump: { pump_enabled: e.target.checked } });
    });
  }

  if (settingAutoSprayToggle) {
    settingAutoSprayToggle.addEventListener('change', (e) => {
      if (autoSprayToggle) autoSprayToggle.checked = e.target.checked;
      updateSettingsBackend({ pump: { auto_spray_enabled: e.target.checked } });
    });
  }

  if (settingSprayDur) {
    settingSprayDur.addEventListener('change', (e) => {
      updateSettingsBackend({ pump: { spray_duration: parseFloat(e.target.value) } });
    });
  }

  if (settingSprayInt) {
    settingSprayInt.addEventListener('change', (e) => {
      updateSettingsBackend({ pump: { spray_interval: parseFloat(e.target.value) } });
    });
  }

  // Storage Settings
  if (settingStorageToggle) {
    settingStorageToggle.addEventListener('change', (e) => {
      updateSettingsBackend({ storage: { auto_storage_enabled: e.target.checked } });
    });
  }

  if (settingSaveDiseased) {
    settingSaveDiseased.addEventListener('change', (e) => {
      updateSettingsBackend({ storage: { save_diseased: e.target.checked } });
    });
  }

  if (settingSaveHealthy) {
    settingSaveHealthy.addEventListener('change', (e) => {
      updateSettingsBackend({ storage: { save_healthy: e.target.checked } });
    });
  }

  // Toggle Upload File Drawer
  if (btnToggleUpload) {
    btnToggleUpload.addEventListener('click', () => {
      const isHidden = dropZoneDrawer.style.display === 'none';
      dropZoneDrawer.style.display = isHidden ? 'block' : 'none';
      btnToggleUpload.textContent = isHidden ? '✕ Close Upload' : '📁 Upload Image';
    });
  }

  if (btnBrowse && fileInput) {
    btnBrowse.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    hudStatusText.textContent = `Analyzing ${file.name}...`;

    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();
    renderDiagnosisReport(data);
    drawBoundingBoxes(data.bounding_boxes, data);
  }

  // ==========================================
  // 7. PRESET TEST SAMPLES & TELEMETRY SYNC
  // ==========================================

  async function loadPresets() {
    try {
      const res = await fetch('/api/samples');
      const samples = await res.json();
      presetsList.innerHTML = '';
      samples.forEach(sample => {
        const item = document.createElement('div');
        item.className = 'preset-item';
        const tagClass = sample.category === 'Leaf' ? 'leaf' : 'non-leaf';
        item.innerHTML = `
          <img src="${sample.url}" alt="${sample.title}" class="preset-thumb">
          <span class="preset-title">${sample.title}</span>
          <span class="preset-tag ${tagClass}">${sample.category}</span>
        `;
        item.addEventListener('click', async () => {
          hudStatusText.textContent = `Testing ${sample.title}...`;
          const blobRes = await fetch(sample.url);
          const blob = await blobRes.blob();
          const file = new File([blob], sample.file, { type: blob.type || 'image/jpeg' });
          handleFileUpload(file);
        });
        presetsList.appendChild(item);
      });
    } catch (e) {
      console.warn("Could not load presets:", e);
    }
  }
  loadPresets();

  // Initial Sync
  fetch('/api/telemetry')
    .then(r => r.json())
    .then(data => {
      if (data.hardware) updateHardwareTelemetry(data.hardware);
      if (data.statistics) updateStatsCounters(data.statistics);
    })
    .catch(() => {});

  // Metrics Modal
  if (btnOpenMetrics && metricsModal) {
    btnOpenMetrics.addEventListener('click', () => metricsModal.classList.add('active'));
    const closeModal = () => metricsModal.classList.remove('active');
    if (btnCloseMetrics) btnCloseMetrics.addEventListener('click', closeModal);
    if (modalBackdrop) modalBackdrop.addEventListener('click', closeModal);
  }

});
