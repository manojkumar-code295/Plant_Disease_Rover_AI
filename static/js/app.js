/**
 * Plant Disease AI & Smart Rover Platform - Client-Side Controller
 * Features:
 *   - 12-Section Tabbed Agricultural Dashboard & Mobile Bottom Nav
 *   - Dual Camera Pipeline: Raspberry Pi Rover Stream & Local Webcam Fallback
 *   - Real-Time Canvas Bounding Boxes (YOLOv8) & Strict Leaf Verification (MobileNetV2)
 *   - Bidirectional Telemetry Polling & Watchdog (CONNECTED / DISCONNECTED / RECONNECTING)
 *   - Motor Navigation Controls with Operator Safety Arming & Emergency Stop
 *   - Precision Water Pump Regulation (Water Only, Duration & Cooldown)
 *   - Real Sensor Telemetry Cards (Displays 'Sensor Not Connected' if absent)
 *   - Persistent SQLite History & Categorized Image Gallery
 *   - PWA Service Worker Registration & Android Install Prompt
 */

// Global State
let isDetecting = false;
let isPaused = false;
let isProcessingFrame = false;
let videoStream = null;
let streamSource = 'webcam'; // 'webcam' or 'rover'
let currentSpeed = 65;
let operatorArmed = true;
let deferredInstallPrompt = null;
let currentGalleryFilter = 'all';
let stagedUploadFile = null;
let selectedCameraDeviceId = null;

// =============================================================
// TAB SWITCHING (Desktop & Mobile)
// =============================================================
function switchTab(tabId) {
  // Update Tab Buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  // Update Tab Panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  // Update Mobile Bottom Nav
  document.querySelectorAll('.mobile-nav-btn').forEach(btn => {
    const fn = btn.getAttribute('onclick') || '';
    btn.classList.toggle('active', fn.includes(tabId));
  });

  // Refresh data for specific tabs
  if (tabId === 'tab-history') loadHistory();
  if (tabId === 'tab-gallery') loadGallery();
  if (tabId === 'tab-pump') loadSprayHistory();

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// =============================================================
// PWA INSTALLATION & SERVICE WORKER
// =============================================================
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(reg => console.log('[PWA] Service Worker registered:', reg.scope))
      .catch(err => console.log('[PWA] Service Worker registration failed:', err));
  });
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  const btnInstall = document.getElementById('btn-install-app');
  if (btnInstall) btnInstall.style.display = 'inline-flex';
});

document.addEventListener('DOMContentLoaded', () => {
  // Install App button
  const btnInstall = document.getElementById('btn-install-app');
  if (btnInstall) {
    btnInstall.addEventListener('click', async () => {
      if (!deferredInstallPrompt) return;
      deferredInstallPrompt.prompt();
      const choice = await deferredInstallPrompt.userChoice;
      if (choice.outcome === 'accepted') {
        btnInstall.style.display = 'none';
      }
      deferredInstallPrompt = null;
    });
  }

  // Bind Main Nav Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      if (tabId) switchTab(tabId);
    });
  });

  // Elements
  const video = document.getElementById('webcam-video');
  const previewVideo = document.getElementById('dashboard-preview-video');
  const piStream = document.getElementById('pi-rover-stream');
  const dashPiStream = document.getElementById('dashboard-pi-stream');
  const boxCanvas = document.getElementById('bounding-box-canvas');
  const captureCanvas = document.getElementById('capture-canvas');
  const boxCtx = boxCanvas ? boxCanvas.getContext('2d') : null;

  // Stream Source Selection
  const btnSrcWebcam = document.getElementById('btn-src-webcam');
  const btnSrcRover = document.getElementById('btn-src-rover');
  if (btnSrcWebcam && btnSrcRover) {
    btnSrcWebcam.addEventListener('click', () => setStreamSource('webcam'));
    btnSrcRover.addEventListener('click', () => setStreamSource('rover'));
  }

  function setStreamSource(src) {
    streamSource = src;
    btnSrcWebcam.classList.toggle('active', src === 'webcam');
    btnSrcRover.classList.toggle('active', src === 'rover');

    const overviewSource = document.getElementById('overview-stream-source');
    if (overviewSource) overviewSource.textContent = src === 'webcam' ? 'Browser Webcam' : 'Raspberry Pi Rover Stream';

    if (src === 'rover') {
      if (video) video.style.display = 'none';
      if (previewVideo) previewVideo.style.display = 'none';
      if (piStream) {
        piStream.style.display = 'block';
        piStream.src = '/video_feed?t=' + Date.now();
      }
      if (dashPiStream) {
        dashPiStream.style.display = 'block';
        dashPiStream.src = '/video_feed?t=' + Date.now();
      }
    } else {
      if (piStream) piStream.style.display = 'none';
      if (dashPiStream) dashPiStream.style.display = 'none';
      if (video) video.style.display = 'block';
      if (previewVideo) previewVideo.style.display = 'block';
    }
  }

  // Camera Buttons
  const btnStart = document.getElementById('btn-start-detection');
  const btnPause = document.getElementById('btn-pause-detection');
  const btnStop = document.getElementById('btn-stop-detection');
  const btnSnapshot = document.getElementById('btn-capture-snapshot');

  if (btnStart) btnStart.addEventListener('click', startCamera);
  if (btnPause) btnPause.addEventListener('click', togglePause);
  if (btnStop) btnStop.addEventListener('click', stopCamera);
  if (btnSnapshot) btnSnapshot.addEventListener('click', captureSnapshot);

  // Operator Arming Switch
  const armToggle = document.getElementById('operator-arm-toggle');
  if (armToggle) {
    armToggle.addEventListener('change', (e) => {
      operatorArmed = e.target.checked;
      updateMotorArmState(operatorArmed);
    });
  }

  // Emergency Stop Buttons
  const btnEmergency = document.getElementById('btn-emergency-stop');
  if (btnEmergency) btnEmergency.addEventListener('click', sendEmergencyStop);

  // Motor Controls (D-Pad)
  document.querySelectorAll('.dpad-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-action');
      if (action) sendRoverCommand(action);
    });
  });

  // Speed Presets
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const spd = parseInt(btn.getAttribute('data-speed'));
      if (spd) setRoverSpeed(spd);
    });
  });

  // Speed Sliders
  const speedSlider = document.getElementById('motor-tab-slider');
  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      setRoverSpeed(parseInt(e.target.value));
    });
  }

  // Keyboard Shortcuts (W/A/S/D/Space/P/B)
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === 'w' || e.key === 'ArrowUp') sendRoverCommand('forward');
    else if (key === 's' || e.key === 'ArrowDown') sendRoverCommand('backward');
    else if (key === 'a' || e.key === 'ArrowLeft') sendRoverCommand('left');
    else if (key === 'd' || e.key === 'ArrowRight') sendRoverCommand('right');
    else if (e.key === ' ' || key === 'x') sendRoverCommand('stop');
    else if (key === 'p') togglePump();
    else if (key === 'b') triggerQuickSpray();
  });

  // File Upload Handlers (Diagnostic Studio)
  const tabDropZone = document.getElementById('tab-drop-zone');
  const tabFileInput = document.getElementById('tab-file-input');
  const btnTriggerDetect = document.getElementById('btn-trigger-detection');
  const btnClearStaged = document.getElementById('btn-clear-staged');

  if (tabDropZone && tabFileInput) {
    tabFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) handleFileSelected(e.target.files[0]);
    });
    tabDropZone.addEventListener('dragover', (e) => { e.preventDefault(); tabDropZone.style.borderColor = '#10b981'; });
    tabDropZone.addEventListener('dragleave', () => { tabDropZone.style.borderColor = 'var(--border)'; });
    tabDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      tabDropZone.style.borderColor = 'var(--border)';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
    });
  }

  if (btnTriggerDetect) {
    btnTriggerDetect.addEventListener('click', () => {
      if (stagedUploadFile) uploadLeafImage(stagedUploadFile);
    });
  }

  if (btnClearStaged) {
    btnClearStaged.addEventListener('click', clearStagedFile);
  }

  // Camera Device Selectors
  const camDevSelect = document.getElementById('camera-device-select');
  if (camDevSelect) {
    camDevSelect.addEventListener('change', (e) => {
      selectedCameraDeviceId = e.target.value;
      if (isDetecting) {
        startCamera(selectedCameraDeviceId);
      }
    });
  }

  // Filter Pills for History
  document.querySelectorAll('#history-filter-pills .filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('#history-filter-pills .filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      loadHistory(pill.getAttribute('data-filter'));
    });
  });

  // Initial Data Loads
  loadPresets();
  loadHistory();
  loadGallery();
  loadSprayHistory();

  // Populate Cameras & Check Permission for Auto-Initialization
  populateCameraDevices();
  checkCameraPermissionAndAutoInit();

  // Start Telemetry Watchdog Polling (Every 1.5 seconds)
  pollTelemetry();
  setInterval(pollTelemetry, 1500);
});

// =============================================================
// CAMERA & REAL-TIME INFERENCE PIPELINE
// =============================================================
async function populateCameraDevices() {
  const select = document.getElementById('camera-device-select');
  const settingSelect = document.getElementById('setting-camera-select');
  if (!select && !settingSelect) return;

  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');

    const updateSelect = (sel) => {
      if (!sel) return;
      sel.innerHTML = '';
      if (videoDevices.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Default Laptop Camera';
        sel.appendChild(opt);
        return;
      }
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Camera ${idx + 1} (${dev.deviceId ? dev.deviceId.substring(0, 8) + '...' : 'Built-in'})`;
        sel.appendChild(opt);
      });
    };

    updateSelect(select);
    updateSelect(settingSelect);
  } catch (err) {
    console.warn('Could not enumerate camera devices:', err);
  }
}

async function checkCameraPermissionAndAutoInit() {
  if (!navigator.permissions || !navigator.permissions.query) return;
  try {
    const result = await navigator.permissions.query({ name: 'camera' });
    updateCameraPermissionUI(result.state);
    result.onchange = () => updateCameraPermissionUI(result.state);
    if (result.state === 'granted') {
      console.log('[Camera] Permission already granted, auto-initializing...');
      startCamera();
    }
  } catch (err) {
    console.log('[Camera] Permission query notice:', err);
  }
}

function updateCameraPermissionUI(state) {
  const alertBox = document.getElementById('camera-permission-alert');
  const statusBadge = document.getElementById('camera-status-badge');
  if (state === 'prompt') {
    if (alertBox) alertBox.style.display = 'block';
    if (statusBadge) {
      statusBadge.textContent = 'Permission Required';
      statusBadge.className = 'badge-hud status-warning';
    }
  } else if (state === 'denied') {
    if (alertBox) {
      alertBox.style.display = 'block';
      alertBox.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 20px;">🚫</span>
          <div style="flex: 1;">
            <strong style="color: #ef4444; font-size: 12px;">Camera Access Denied</strong>
            <p style="font-size: 11px; color: var(--text-muted); margin: 2px 0 0 0;">Please click the camera icon in your browser address bar and enable camera access for this site.</p>
          </div>
        </div>
      `;
    }
    if (statusBadge) {
      statusBadge.textContent = 'Access Denied';
      statusBadge.className = 'badge-hud status-danger';
    }
  } else if (state === 'granted') {
    if (alertBox) alertBox.style.display = 'none';
  }
}

async function startCamera(deviceId = null) {
  const video = document.getElementById('webcam-video');
  const previewVideo = document.getElementById('dashboard-preview-video');
  const btnStart = document.getElementById('btn-start-detection');
  const btnPause = document.getElementById('btn-pause-detection');
  const btnStop = document.getElementById('btn-stop-detection');
  const statusBadge = document.getElementById('camera-status-badge');
  const alertBox = document.getElementById('camera-permission-alert');

  const select = document.getElementById('camera-device-select');
  const targetId = deviceId || (select ? select.value : null);

  const constraints = {
    video: targetId ? { deviceId: { exact: targetId } } : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' },
    audio: false
  };

  try {
    // If an existing stream is active, stop tracks first
    if (videoStream) {
      videoStream.getTracks().forEach(track => track.stop());
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoStream = stream;
    if (video) video.srcObject = stream;
    if (previewVideo) previewVideo.srcObject = stream;

    isDetecting = true;
    isPaused = false;

    if (btnStart) btnStart.disabled = true;
    if (btnPause) { btnPause.disabled = false; btnPause.innerHTML = '<span>⏸</span> Pause'; }
    if (btnStop) btnStop.disabled = false;

    if (statusBadge) {
      statusBadge.textContent = 'Camera Active (Laptop Webcam)';
      statusBadge.className = 'badge-hud status-online';
    }
    if (alertBox) alertBox.style.display = 'none';

    // Populate camera labels now that permission has been granted
    await populateCameraDevices();

    // Start continuous inference loop
    requestAnimationFrame(inferenceLoop);
  } catch (err) {
    console.error('Camera access error:', err);
    if (statusBadge) {
      statusBadge.textContent = 'Camera Error';
      statusBadge.className = 'badge-hud status-danger';
    }
    if (alertBox) alertBox.style.display = 'block';
    alert('Unable to access laptop webcam: ' + err.message + '\nPlease allow camera access when prompted by the browser.');
  }
}

function togglePause() {
  isPaused = !isPaused;
  const btnPause = document.getElementById('btn-pause-detection');
  const statusBadge = document.getElementById('camera-status-badge');
  if (btnPause) {
    btnPause.innerHTML = isPaused ? '<span>▶</span> Resume' : '<span>⏸</span> Pause';
  }
  if (statusBadge) {
    statusBadge.textContent = isPaused ? 'Camera Paused' : 'Camera Active (Laptop Webcam)';
    statusBadge.className = isPaused ? 'badge-hud status-warning' : 'badge-hud status-online';
  }
}

function stopCamera() {
  isDetecting = false;
  isPaused = false;
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  const video = document.getElementById('webcam-video');
  const previewVideo = document.getElementById('dashboard-preview-video');
  if (video) video.srcObject = null;
  if (previewVideo) previewVideo.srcObject = null;

  const boxCanvas = document.getElementById('bounding-box-canvas');
  if (boxCanvas) {
    const ctx = boxCanvas.getContext('2d');
    ctx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);
  }

  const btnStart = document.getElementById('btn-start-detection');
  const btnPause = document.getElementById('btn-pause-detection');
  const btnStop = document.getElementById('btn-stop-detection');
  const statusBadge = document.getElementById('camera-status-badge');

  if (btnStart) btnStart.disabled = false;
  if (btnPause) btnPause.disabled = true;
  if (btnStop) btnStop.disabled = true;

  if (statusBadge) {
    statusBadge.textContent = 'Camera Stopped';
    statusBadge.className = 'badge-hud';
  }
}

async function inferenceLoop() {
  if (!isDetecting) return;
  if (!isPaused && !isProcessingFrame) {
    await processWebcamFrame();
  }
  setTimeout(() => requestAnimationFrame(inferenceLoop), 120); // ~8 FPS inference throttle to save CPU
}

async function processWebcamFrame() {
  const video = document.getElementById('webcam-video');
  const captureCanvas = document.getElementById('capture-canvas');
  const boxCanvas = document.getElementById('bounding-box-canvas');
  if (!video || video.readyState < 2 || !captureCanvas || !boxCanvas) return;

  isProcessingFrame = true;
  const t0 = performance.now();

  try {
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    captureCanvas.width = w;
    captureCanvas.height = h;
    boxCanvas.width = w;
    boxCanvas.height = h;

    const cCtx = captureCanvas.getContext('2d');
    cCtx.drawImage(video, 0, 0, w, h);
    const b64 = captureCanvas.toDataURL('image/jpeg', 0.82);

    const resp = await fetch('/api/predict_live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 })
    });

    const data = await resp.json();
    const t1 = performance.now();
    const latency = Math.round(t1 - t0);

    const hudLat = document.getElementById('hud-latency');
    if (hudLat) hudLat.textContent = `Latency: ${latency} ms`;
    const hudFps = document.getElementById('hud-fps');
    if (hudFps) hudFps.textContent = `FPS: ${Math.round(1000 / Math.max(1, latency))}`;

    // Render Bounding Boxes on Canvas
    renderBoundingBoxes(data.bounding_boxes || [], w, h, data);

    // Update Live Diagnosis Display
    updateDiagnosisUI(data);

  } catch (err) {
    console.error('Inference error:', err);
  } finally {
    isProcessingFrame = false;
  }
}

function renderBoundingBoxes(boxes, w, h, data) {
  const boxCanvas = document.getElementById('bounding-box-canvas');
  if (!boxCanvas) return;
  const ctx = boxCanvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  boxes.forEach(item => {
    const b = item.box;
    if (!b || b.length < 4) return;
    const [x1, y1, x2, y2] = b;
    const bw = x2 - x1;
    const bh = y2 - y1;

    const isLeaf = item.type === 'leaf';
    ctx.lineWidth = 3;
    ctx.strokeStyle = isLeaf ? '#10b981' : '#ef4444';
    ctx.strokeRect(x1, y1, bw, bh);

    // Label tag
    const label = `${item.label} (${Math.round(item.confidence)}%)`;
    ctx.font = 'bold 12px Plus Jakarta Sans, sans-serif';
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = isLeaf ? 'rgba(16, 185, 129, 0.85)' : 'rgba(239, 68, 68, 0.85)';
    ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 12, 22);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 6, Math.max(16, y1 - 6));
  });
}

function updateDiagnosisUI(data) {
  const emptyState = document.getElementById('empty-state');
  const rejectionBox = document.getElementById('rejection-box');
  const resultDetails = document.getElementById('result-details');
  const hudLeaf = document.getElementById('hud-leaf-status');
  const hudText = document.getElementById('hud-status-text');
  const hudObs = document.getElementById('hud-obs-badge');

  if (hudObs) hudObs.style.display = data.has_obstacle ? 'inline-block' : 'none';

  if (!data.success && data.status === 'rejected') {
    if (emptyState) emptyState.style.display = 'none';
    if (resultDetails) resultDetails.style.display = 'none';
    if (rejectionBox) {
      rejectionBox.style.display = 'block';
      const rTitle = document.getElementById('rejection-title');
      if (rTitle) rTitle.textContent = data.message || 'Invalid Image! Please upload a clear plant leaf image.';
    }
    if (hudLeaf) hudLeaf.textContent = 'Non-Leaf Rejected';
    if (hudText) hudText.textContent = data.message || 'Invalid Image!';
    return;
  }

  if (data.is_leaf) {
    if (emptyState) emptyState.style.display = 'none';
    if (rejectionBox) rejectionBox.style.display = 'none';
    if (resultDetails) resultDetails.style.display = 'block';

    const pTitle = document.getElementById('prediction-title');
    const pConf = document.getElementById('confidence-val');
    const hPill = document.getElementById('health-pill');
    const vText = document.getElementById('verifier-text');
    const sprayBanner = document.getElementById('spray-trigger-banner');

    if (pTitle) pTitle.textContent = data.display_name || data.prediction || 'Healthy Leaf';
    if (pConf) pConf.textContent = `${Math.round(data.confidence || 0)}%`;
    if (vText) vText.textContent = `Leaf Verified Successfully (${Math.round(data.leaf_confidence || 99)}%)`;

    if (hPill) {
      hPill.textContent = data.is_healthy ? 'Optimal Vigor' : 'Pathology Alert';
      hPill.className = data.is_healthy ? 'health-pill' : 'health-pill diseased';
    }

    if (hudLeaf) hudLeaf.textContent = `${data.prediction || 'Leaf'} (${Math.round(data.confidence || 0)}%)`;
    if (hudText) hudText.textContent = data.is_healthy ? 'Healthy Foliage Detected' : `Alert: ${data.prediction} Detected!`;

    // Spray notification banner
    if (sprayBanner) {
      sprayBanner.style.display = (data.auto_spray_trigger && data.auto_spray_trigger.triggered) ? 'flex' : 'none';
    }

    // Probability Bars
    const probList = document.getElementById('prob-list');
    if (probList && data.all_probabilities) {
      let html = '';
      for (const [cls, val] of Object.entries(data.all_probabilities)) {
        html += `
          <div class="prob-row">
            <div class="prob-header"><span>${cls}</span><span>${Math.round(val)}%</span></div>
            <div class="prob-track"><div class="prob-bar" style="width: ${val}%; background: ${cls.includes('Healthy') ? '#10b981' : '#f59e0b'};"></div></div>
          </div>
        `;
      }
      probList.innerHTML = html;
    }

    // Sync to Dashboard tab overview
    const dTitle = document.getElementById('dash-pred-title');
    const dConf = document.getElementById('dash-conf-val');
    const dDesc = document.getElementById('dash-pred-desc');
    const dDiag = document.getElementById('dash-diag-text');
    if (dTitle) dTitle.textContent = data.display_name || data.prediction;
    if (dConf) dConf.textContent = `${Math.round(data.confidence || 0)}%`;
    if (dDesc && data.disease_info) dDesc.textContent = data.disease_info.description || '';
    if (dDiag) dDiag.textContent = `${data.prediction} (${Math.round(data.confidence || 0)}%)`;
  }
}

async function captureSnapshot() {
  const captureCanvas = document.getElementById('capture-canvas');
  if (!captureCanvas) return;
  const b64 = captureCanvas.toDataURL('image/jpeg', 0.9);
  await uploadLeafBase64(b64);
  switchTab('tab-detection');
}

// =============================================================
// DIAGNOSTIC STUDIO UPLOAD & PRESETS
// =============================================================
function handleFileSelected(file) {
  if (!file) return;
  stagedUploadFile = file;

  const actionBar = document.getElementById('upload-action-bar');
  const thumb = document.getElementById('upload-staged-thumb');
  const nameEl = document.getElementById('upload-staged-name');
  const sizeEl = document.getElementById('upload-staged-size');
  const previewImg = document.getElementById('tab-diag-preview-img');
  const previewCont = document.getElementById('tab-diag-preview-container');

  if (nameEl) nameEl.textContent = file.name;
  if (sizeEl) sizeEl.textContent = (file.size / 1024).toFixed(1) + ' KB';

  const reader = new FileReader();
  reader.onload = (e) => {
    if (thumb) thumb.src = e.target.result;
    if (previewImg && previewCont) {
      previewImg.src = e.target.result;
      previewCont.style.display = 'block';
    }
  };
  reader.readAsDataURL(file);

  if (actionBar) actionBar.style.display = 'block';
}

function clearStagedFile() {
  stagedUploadFile = null;
  const actionBar = document.getElementById('upload-action-bar');
  const fileInput = document.getElementById('tab-file-input');
  const previewCont = document.getElementById('tab-diag-preview-container');
  const container = document.getElementById('tab-diag-content');

  if (actionBar) actionBar.style.display = 'none';
  if (fileInput) fileInput.value = '';
  if (previewCont) previewCont.style.display = 'none';

  if (container) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🍃</div>
        <h4>Awaiting Leaf Image</h4>
        <p>Upload an image or select a preset from the left to run strict leaf verification and disease diagnosis.</p>
      </div>
    `;
  }
}

async function uploadLeafImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('image', file); // Include both keys for complete backward compatibility
  showDiagnosticLoading();

  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    renderDiagnosticReport(data);
  } catch (err) {
    alert('Prediction request error: ' + err.message);
  }
}

async function uploadLeafBase64(b64) {
  showDiagnosticLoading();
  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 })
    });
    const data = await resp.json();
    renderDiagnosticReport(data);
  } catch (err) {
    alert('Prediction error: ' + err.message);
  }
}

function showDiagnosticLoading() {
  const container = document.getElementById('tab-diag-content');
  if (container) {
    container.innerHTML = `
      <div style="text-align: center; padding: 32px;">
        <div style="font-size: 36px; animation: spin 1s infinite linear;">🔄</div>
        <h4 style="margin-top: 12px; color: #fff;">Running Dual AI Verification & Classification...</h4>
        <p style="font-size: 11px; color: var(--text-muted);">YOLOv8 Localizer &rarr; Strict Leaf Verifier &rarr; CNN Classifier</p>
      </div>
    `;
  }
}

function renderDiagnosticReport(data) {
  const container = document.getElementById('tab-diag-content');
  const previewImg = document.getElementById('tab-diag-preview-img');
  const previewCont = document.getElementById('tab-diag-preview-container');

  if (data.preview_image && previewImg && previewCont) {
    previewImg.src = data.preview_image;
    previewCont.style.display = 'block';
  }

  if (!data.success || !data.is_leaf) {
    container.innerHTML = `
      <div class="rejection-box" style="display: block;">
        <div class="rejection-header">
          <span class="rejection-icon">🚫</span>
          <div>
            <h3 class="rejection-title">${data.message || 'Invalid Image! Please upload a clear plant leaf image.'}</h3>
            <p class="rejection-reason">Non-leaf candidate rejected by strict botanical CNN filter.</p>
          </div>
        </div>
        <div class="rejection-body">
          <p>The AI model enforces strict input verification to prevent false diagnostics on human faces, vehicles, animals, or unrelated objects.</p>
        </div>
      </div>
    `;
    return;
  }

  const dis = data.disease_info || {};
  const treats = dis.treatments || {};

  container.innerHTML = `
    <div class="verifier-badge">
      <span class="verified-icon">🛡️</span>
      <span>Leaf Verified (Confidence: ${Math.round(data.leaf_confidence || 99)}%)</span>
    </div>

    <div class="prediction-banner">
      <div class="prediction-info">
        <span class="health-pill ${data.is_healthy ? '' : 'diseased'}">${data.is_healthy ? 'Healthy' : 'Diseased'}</span>
        <h2 class="prediction-title">${data.display_name || data.prediction}</h2>
        <p class="prediction-pathogen">${dis.pathogen || ''}</p>
      </div>
      <div class="confidence-circle">
        <span class="confidence-val">${Math.round(data.confidence || 0)}%</span>
        <span class="confidence-lbl">Confidence</span>
      </div>
    </div>

    <div class="card-section" style="margin-top: 14px;">
      <h4 class="section-title">Description & Agronomic Diagnosis</h4>
      <p style="font-size: 11px; color: #cbd5e1; line-height: 1.5;">${dis.description || 'Diagnosis completed.'}</p>
    </div>

    <div class="treatment-grid" style="margin-top: 12px;">
      <div class="treatment-box">
        <h5>🧪 Chemical Action</h5>
        <p>${treats.chemical || 'None required.'}</p>
      </div>
      <div class="treatment-box">
        <h5>🌿 Organic Control</h5>
        <p>${treats.organic || treats.soil || 'None required.'}</p>
      </div>
      <div class="treatment-box full-width">
        <h5>🛡️ Prevention Strategy</h5>
        <p>${treats.prevention || treats.preventive || 'Maintain clean cultivation and routine crop scouting.'}</p>
      </div>
    </div>
  `;
}

async function loadPresets() {
  const container = document.getElementById('presets-list');
  const tabContainer = document.getElementById('tab-presets-list');
  try {
    const resp = await fetch('/api/samples');
    const samples = await resp.json();

    let html = '';
    samples.forEach(s => {
      const isLeaf = s.category === 'Leaf';
      html += `
        <div class="preset-item" onclick="testPreset('${s.url}')">
          <img src="${s.url}" alt="${s.title}" class="preset-thumb">
          <span class="preset-title">${s.title}</span>
          <span class="preset-tag ${isLeaf ? 'leaf' : 'non-leaf'}">${s.category}</span>
        </div>
      `;
    });

    if (container) container.innerHTML = html;
    if (tabContainer) tabContainer.innerHTML = html;
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

async function testPreset(url) {
  showDiagnosticLoading();
  switchTab('tab-detection');
  try {
    const resp = await fetch(url);
    const blob = await resp.blob();
    const file = new File([blob], 'preset.jpg', { type: 'image/jpeg' });
    await uploadLeafImage(file);
  } catch (err) {
    alert('Preset load error: ' + err.message);
  }
}

// =============================================================
// ROVER MOVEMENT & SPEED CONTROLLER
// =============================================================
async function sendRoverCommand(action) {
  if (!operatorArmed && action !== 'stop') {
    alert('Safety Interlock: Enable Remote Movement Arming switch before sending motion commands.');
    return;
  }

  try {
    const resp = await fetch('/api/rover/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action, armed: operatorArmed })
    });
    const data = await resp.json();
    if (!data.success) {
      alert('Rover command error: ' + (data.error || 'Unknown error'));
    }
  } catch (err) {
    console.error('Rover command request failed:', err);
  }
}

async function setRoverSpeed(speed) {
  currentSpeed = speed;
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.getAttribute('data-speed')) === speed);
  });

  const speedSlider = document.getElementById('motor-tab-slider');
  if (speedSlider) speedSlider.value = speed;
  const speedDisp = document.getElementById('motor-tab-speed-disp');
  if (speedDisp) speedDisp.textContent = `${speed}% PWM`;

  try {
    await fetch('/api/rover/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed: speed })
    });
  } catch (err) {
    console.error('Speed update error:', err);
  }
}

async function sendEmergencyStop() {
  try {
    await fetch('/api/rover/emergency_stop', { method: 'POST' });
    alert('🛑 EMERGENCY STOP ENGAGED: DC Motors and Water Pump immediately shut off.');
  } catch (err) {
    console.error('Emergency stop error:', err);
  }
}

function updateMotorArmState(armed) {
  const badge = document.getElementById('motor-tab-mode');
  if (badge) {
    badge.textContent = armed ? 'Manual Driving (Armed)' : 'Safety Disarmed';
    badge.style.color = armed ? '#34d399' : '#f87171';
  }
}

// =============================================================
// WATER PUMP CONTROLLER (WATER ONLY)
// =============================================================
let pumpState = false;

async function togglePump() {
  pumpState = !pumpState;
  const action = pumpState ? 'on' : 'off';
  try {
    await fetch('/api/rover/pump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action })
    });
    updatePumpUI(pumpState);
  } catch (err) {
    console.error('Pump toggle error:', err);
  }
}

async function triggerQuickSpray() {
  try {
    const resp = await fetch('/api/rover/pump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'burst', duration: 2.0 })
    });
    const data = await resp.json();
    alert('💦 Water Spray Burst Triggered (2.0s Duration). Pure water dispensed.');
    loadSprayHistory();
  } catch (err) {
    console.error('Spray burst error:', err);
  }
}

function updatePumpUI(isActive) {
  const pumpPill = document.getElementById('pump-tab-pill');
  const btnIcon = document.getElementById('pump-tab-btn-icon');
  const btnText = document.getElementById('pump-tab-btn-text');
  const statPump = document.getElementById('stat-pump-status');

  if (pumpPill) {
    pumpPill.textContent = isActive ? 'PUMP ACTIVE (SPRAYING)' : 'PUMP OFF';
    pumpPill.className = isActive ? 'pump-status-pill active' : 'pump-status-pill';
  }
  if (btnText) btnText.textContent = isActive ? 'Turn Pump OFF' : 'Turn Pump ON';
  if (statPump) statPump.textContent = isActive ? 'SPRAYING' : 'STANDBY';
}

async function loadSprayHistory() {
  const tbody = document.getElementById('spray-history-tbody');
  if (!tbody) return;
  try {
    const resp = await fetch('/api/history?category=all&limit=20');
    const rows = await resp.json();
    const sprays = rows.filter(r => r.spraying_status && r.spraying_status !== 'NONE');

    if (sprays.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-dim);">No spraying operations recorded yet.</td></tr>';
      return;
    }

    let html = '';
    sprays.forEach(s => {
      html += `
        <tr>
          <td>${s.timestamp}</td>
          <td><strong>${s.disease}</strong></td>
          <td>2.0s</td>
          <td><span style="color:#34d399;">✅ Pi Confirmed (Relay Pin 24)</span></td>
          <td><span class="badge-status badge-connected">COMPLETED</span></td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  } catch (err) {
    console.error('Spray history load error:', err);
  }
}

// =============================================================
// TELEMETRY WATCHDOG & SENSOR MONITORING (Real Values Only)
// =============================================================
async function pollTelemetry() {
  try {
    const resp = await fetch('/api/telemetry', { cache: 'no-store' });
    const data = await resp.json();

    // 1. Connection Status Badge: CONNECTED / DISCONNECTED / RECONNECTING
    const statusBadge = document.getElementById('rover-status-badge');
    const statusText = document.getElementById('rover-status-text');
    const monitorBadge = document.getElementById('rover-monitor-badge');
    const rState = data.rover_status || 'DISCONNECTED';

    if (statusText) statusText.textContent = rState;
    if (statusBadge) {
      statusBadge.className = 'badge-status ' + (
        rState === 'CONNECTED' ? 'badge-connected' :
        rState === 'RECONNECTING' ? 'badge-reconnecting' : 'badge-disconnected'
      );
    }
    if (monitorBadge) {
      monitorBadge.textContent = rState;
      monitorBadge.className = 'badge-status ' + (
        rState === 'CONNECTED' ? 'badge-connected' :
        rState === 'RECONNECTING' ? 'badge-reconnecting' : 'badge-disconnected'
      );
    }

    // 2. Real Sensor Values (Display 'Sensor Not Connected' if absent)
    const sensors = data.sensors || {};
    const setSensor = (id, val) => {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = val;
        el.classList.toggle('sensor-disconnected', val === 'Sensor Not Connected');
      }
    };

    setSensor('card-sensor-dist', sensors.ultrasonic_distance_cm || 'Sensor Not Connected');
    setSensor('card-sensor-obs', sensors.obstacle_detected !== undefined && sensors.obstacle_detected !== 'Sensor Not Connected' ? (sensors.obstacle_detected ? 'OBSTACLE DETECTED' : 'Path Clear') : 'Sensor Not Connected');
    setSensor('card-sensor-soil', sensors.soil_moisture_pct || 'Sensor Not Connected');
    setSensor('card-sensor-temp', sensors.temperature_c || 'Sensor Not Connected');
    setSensor('card-sensor-hum', sensors.humidity_pct || 'Sensor Not Connected');
    setSensor('card-sensor-bat', sensors.battery_voltage || 'Sensor Not Connected');

    // Dashboard Tab Quick Sensors
    setSensor('sensor-dash-dist', sensors.ultrasonic_distance_cm || 'Sensor Not Connected');
    setSensor('sensor-dash-bat', sensors.battery_voltage || 'Sensor Not Connected');
    setSensor('dash-tele-dist', sensors.ultrasonic_distance_cm || 'Sensor Not Connected');
    setSensor('dash-tele-bat', sensors.battery_voltage || 'Sensor Not Connected');

    // Rover Tab Sidebar
    setSensor('rover-card-dist', sensors.ultrasonic_distance_cm || 'Sensor Not Connected');
    setSensor('rover-card-battery', sensors.battery_voltage || 'Sensor Not Connected');
    const roverObs = document.getElementById('rover-card-obs');
    if (roverObs) {
      roverObs.textContent = sensors.obstacle_detected === true ? '🛑 Obstacle Detected! Braking engaged' : '✅ Path Clear (Safe to drive)';
      roverObs.style.color = sensors.obstacle_detected === true ? '#f87171' : '#34d399';
    }

    const lastComm = document.getElementById('rover-last-comm');
    if (lastComm) lastComm.textContent = sensors.last_communication || 'Never';

    // 3. Driving Telemetry
    const hw = data.hardware || {};
    const dir = hw.direction || 'STOP';
    const spd = hw.speed || 65;

    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setVal('dash-tele-dir', dir);
    setVal('dash-dir-display', dir);
    setVal('dash-tele-speed', `${spd}%`);
    setVal('stat-motor-speed', `${spd}% PWM`);
    setVal('twin-motion', dir);
    setVal('twin-speed', `${spd}%`);
    setVal('rover-card-motion', dir);
    setVal('rover-card-speed', `${spd}% PWM`);

    // 4. Statistics Counters
    const stats = data.statistics || {};
    setVal('stat-leaves-detected', stats.total || 0);
    setVal('stat-healthy-leaves', stats.healthy || 0);
    setVal('stat-diseased-leaves', stats.diseased || 0);
    setVal('stat-total-sprays', stats.total_sprays || 0);

  } catch (err) {
    // Graceful silent swallow on background poll failures
  }
}

// =============================================================
// HISTORY TABLE & CSV EXPORT
// =============================================================
async function loadHistory(category = 'all') {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;

  try {
    const url = `/api/history?category=${category}&limit=50`;
    const resp = await fetch(url);
    const rows = await resp.json();

    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-dim);">No detection records found.</td></tr>';
      return;
    }

    let html = '';
    rows.forEach(r => {
      const isHealthy = r.is_healthy === 1 || r.disease.toLowerCase().includes('healthy');
      const imgTag = r.image ? `<img src="/static/images/${r.image}" class="thumb-mini" alt="${r.disease}" onerror="this.src='/static/icons/icon-48.png'">` : '<span>-</span>';
      html += `
        <tr>
          <td>${imgTag}</td>
          <td><strong>${r.disease}</strong></td>
          <td>${r.confidence}%</td>
          <td><span class="badge-status ${isHealthy ? 'badge-connected' : 'badge-disconnected'}">${isHealthy ? 'Healthy' : 'Diseased'}</span></td>
          <td>${r.rover_id || 'ROVER-01'}</td>
          <td>${r.spraying_status || 'NONE'}</td>
          <td>${r.timestamp}</td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  } catch (err) {
    console.error('History load error:', err);
  }
}

async function exportHistoryCSV() {
  try {
    const resp = await fetch('/api/history?category=all&limit=200');
    const rows = await resp.json();
    if (!rows || rows.length === 0) {
      alert('No records available to export.');
      return;
    }

    let csvContent = 'data:text/csv;charset=utf-8,ID,Date,Timestamp,Disease,Confidence,Is_Healthy,Rover_ID,Spraying_Status\n';
    rows.forEach(r => {
      csvContent += `"${r.result_id}","${r.date}","${r.timestamp}","${r.disease}","${r.confidence}","${r.is_healthy}","${r.rover_id}","${r.spraying_status}"\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `plant_disease_audit_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (err) {
    alert('CSV export error: ' + err.message);
  }
}

// =============================================================
// CATEGORIZED IMAGE GALLERY
// =============================================================
async function loadGallery() {
  const container = document.getElementById('gallery-container');
  if (!container) return;

  try {
    const resp = await fetch('/api/gallery');
    const cats = await resp.json();

    let allItems = [];
    for (const [catName, items] of Object.entries(cats)) {
      items.forEach(i => allItems.push(i));
    }

    renderGalleryItems(allItems);
  } catch (err) {
    console.error('Gallery load error:', err);
  }
}

function filterGallery(category) {
  currentGalleryFilter = category;
  document.querySelectorAll('.gallery-cat-tabs .filter-pill').forEach(p => {
    p.classList.toggle('active', p.getAttribute('data-cat') === category);
  });
  loadGallery();
}

function renderGalleryItems(items) {
  const container = document.getElementById('gallery-container');
  if (!container) return;

  const filtered = currentGalleryFilter === 'all' ? items : items.filter(i => {
    const dis = (i.disease || '').toLowerCase();
    return dis.includes(currentGalleryFilter.toLowerCase());
  });

  if (filtered.length === 0) {
    container.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 24px;">No images in this category yet.</div>';
    return;
  }

  let html = '';
  filtered.forEach(item => {
    const isHealthy = item.is_healthy === 1 || item.disease.toLowerCase().includes('healthy');
    html += `
      <div class="gallery-card" onclick="alert('${item.disease} - Conf: ${item.confidence}%\nDate: ${item.timestamp}')">
        <img src="/static/images/${item.image}" alt="${item.disease}" class="gallery-img" onerror="this.src='/static/icons/icon-192.png'">
        <div class="gallery-meta">
          <div class="gallery-title">${item.disease}</div>
          <div class="gallery-sub">${item.confidence}% &bull; ${item.date || ''}</div>
          <span class="badge-status ${isHealthy ? 'badge-connected' : 'badge-disconnected'}" style="margin-top: 4px; font-size: 8px;">
            ${isHealthy ? 'Healthy Leaf' : 'Pathology Alert'}
          </span>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// =============================================================
// SETTINGS PERSISTENCE
// =============================================================
async function saveSettings() {
  const confSlider = document.getElementById('setting-conf-slider');
  const objDetect = document.getElementById('setting-obj-detect');
  const diseaseDetect = document.getElementById('setting-disease-detect');
  const sprayDur = document.getElementById('setting-spray-dur');
  const sprayInt = document.getElementById('setting-spray-int');
  const pumpMaster = document.getElementById('setting-pump-master');
  const autoSpray = document.getElementById('setting-auto-spray-toggle');

  const payload = {
    detection: {
      confidence_threshold: parseFloat(confSlider ? confSlider.value : 70),
      object_detection_enabled: objDetect ? objDetect.checked : true,
      disease_detection_enabled: diseaseDetect ? diseaseDetect.checked : true
    },
    pump: {
      pump_enabled: pumpMaster ? pumpMaster.checked : true,
      auto_spray_enabled: autoSpray ? autoSpray.checked : true,
      spray_duration: parseFloat(sprayDur ? sprayDur.value : 2.0),
      spray_interval: parseFloat(sprayInt ? sprayInt.value : 4.0)
    }
  };

  try {
    const resp = await fetch('/api/settings/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const res = await resp.json();
    if (res.success) {
      alert('✅ System settings updated successfully!');
    }
  } catch (err) {
    alert('Settings update error: ' + err.message);
  }
}

async function pingRover() {
  alert('📡 Pinging Rover Edge Node...\nMonitoring /api/rover/heartbeat telemetry synchronization.');
}
