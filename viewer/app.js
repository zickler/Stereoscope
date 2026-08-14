import { DEFAULT_PROFILE, parseCardboardProfileText } from "./cardboard-profile.js";
import { GlViewer } from "./gl-viewer.js";
import { QrScanner, qrScanningSupported } from "./qr-scan.js";
import { loadPxPerMeter, initCalibration } from "./calibration.js";

const PROFILE_STORAGE_KEY = "cardboardViewer.profile";

const el = (id) => document.getElementById(id);
const galleryEl = el("gallery");
const vrView = el("vr-view");
const canvas = el("gl-canvas");
const settingsPanel = el("settings-panel");
const qrScanView = el("qr-scan-view");
const calibrationView = el("calibration-view");
const profileReadout = el("profile-readout");
const calibReadoutSummary = el("calib-readout-summary");

let profile = loadProfile();
let pxPerMeter = loadPxPerMeter();
let scenes = [];
let viewer = null;
let currentScene = null;
let rafPending = false;

function loadProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
    if (raw) return { ...DEFAULT_PROFILE, ...JSON.parse(raw) };
  } catch {
    // fall through to default
  }
  return { ...DEFAULT_PROFILE };
}

function saveProfile(p) {
  profile = p;
  localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(p));
  renderProfileReadout();
}

function renderProfileReadout() {
  profileReadout.textContent =
    `${profile.vendor} — ${profile.model}\n` +
    `inter-lens ${(profile.interLensDistance * 1000).toFixed(1)}mm, ` +
    `screen-to-lens ${(profile.screenToLensDistance * 1000).toFixed(1)}mm, ` +
    `distortion [${profile.distortionCoefficients.map((v) => v.toFixed(3)).join(", ")}]`;
}

function renderCalibSummary() {
  calibReadoutSummary.textContent = `${Math.round(pxPerMeter)} device px/meter`;
}

async function loadScenes() {
  const res = await fetch("scenes.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to load scenes.json: ${res.status}`);
  scenes = await res.json();
}

function renderGallery() {
  galleryEl.innerHTML = "";
  for (const scene of scenes) {
    const item = document.createElement("div");
    item.className = "gallery-item";
    item.innerHTML = `
      <img src="${scene.thumbnail}" alt="${scene.title}" loading="lazy">
      <div class="label">${scene.title}</div>
    `;
    item.addEventListener("click", () => enterVr(scene));
    galleryEl.appendChild(item);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load image ${url}`));
    img.src = url;
  });
}

async function enterVr(scene) {
  currentScene = scene;
  vrView.hidden = false;
  document.getElementById("gallery-view").hidden = true;

  if (!viewer) viewer = new GlViewer(canvas);
  const [leftImg, rightImg] = await Promise.all([
    loadImage(scene.leftImage),
    loadImage(scene.rightImage),
  ]);
  const meta = { focalPx: scene.focalPx, cx: scene.cx, cy: scene.cy, width: scene.width, height: scene.height };
  viewer.setEyeImage("left", leftImg, meta);
  viewer.setEyeImage("right", rightImg, meta);
  requestRedraw();

  if (canvas.requestFullscreen) {
    canvas.requestFullscreen().catch(() => {});
  } else if (canvas.webkitRequestFullscreen) {
    canvas.webkitRequestFullscreen();
  }
  if (screen.orientation && screen.orientation.lock) {
    screen.orientation.lock("landscape").catch(() => {});
  }
}

function exitVr() {
  vrView.hidden = true;
  document.getElementById("gallery-view").hidden = false;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else if (document.webkitFullscreenElement && document.webkitExitFullscreen) document.webkitExitFullscreen();
}

function requestRedraw() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (!vrView.hidden && viewer) viewer.render(profile, pxPerMeter);
  });
}

window.addEventListener("resize", requestRedraw);
window.addEventListener("orientationchange", requestRedraw);

el("exit-vr-btn").addEventListener("click", exitVr);
el("settings-btn").addEventListener("click", () => {
  renderProfileReadout();
  renderCalibSummary();
  settingsPanel.hidden = false;
});
el("close-settings-btn").addEventListener("click", () => {
  settingsPanel.hidden = true;
});
el("reset-profile-btn").addEventListener("click", () => {
  saveProfile({ ...DEFAULT_PROFILE });
  requestRedraw();
});
el("manual-qr-apply").addEventListener("click", () => {
  const text = el("manual-qr-input").value;
  try {
    saveProfile(parseCardboardProfileText(text));
    requestRedraw();
  } catch (err) {
    alert(`Could not parse that QR/config text: ${err.message}`);
  }
});

let scanner = null;

function setQrStatus(text, variant) {
  const status = el("qr-status");
  const video = el("qr-video");
  status.textContent = text;
  status.classList.remove("qr-status--success", "qr-status--error");
  video.classList.remove("qr-video--success", "qr-video--error");
  if (variant) {
    status.classList.add(`qr-status--${variant}`);
    video.classList.add(`qr-video--${variant}`);
  }
}

el("scan-qr-btn").addEventListener("click", async () => {
  if (!qrScanningSupported()) {
    alert("Camera access isn't available in this browser; use 'Paste QR URL manually' instead.");
    return;
  }
  settingsPanel.hidden = true;
  qrScanView.hidden = false;
  setQrStatus("Starting camera…", null);
  scanner = new QrScanner(el("qr-video"));
  try {
    await scanner.start();
    setQrStatus("Scanning…", null);
    const text = await scanner.scanOnce();
    const scanned = parseCardboardProfileText(text);
    saveProfile(scanned);
    requestRedraw();
    setQrStatus(`✓ Scanned "${scanned.vendor} ${scanned.model}"`, "success");
    await new Promise((r) => setTimeout(r, 1800));
  } catch (err) {
    setQrStatus(`✕ Scan failed: ${err.message}`, "error");
    await new Promise((r) => setTimeout(r, 1500));
  } finally {
    scanner.stop();
    qrScanView.hidden = true;
    settingsPanel.hidden = false;
  }
});
el("qr-cancel-btn").addEventListener("click", () => {
  if (scanner) scanner.stop();
  qrScanView.hidden = true;
  settingsPanel.hidden = false;
});

const calibController = initCalibration(calibrationView, (newPxPerMeter) => {
  pxPerMeter = newPxPerMeter;
  calibController.hide();
  settingsPanel.hidden = false;
  renderCalibSummary();
  requestRedraw();
});
el("recalibrate-btn").addEventListener("click", () => {
  settingsPanel.hidden = true;
  calibController.show();
});

(async function init() {
  renderProfileReadout();
  renderCalibSummary();
  try {
    await loadScenes();
    renderGallery();
  } catch (err) {
    galleryEl.textContent = `Could not load scenes.json: ${err.message}`;
  }
})();
