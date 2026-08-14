import { DEFAULT_PROFILE, resolveCardboardProfileText, UnresolvedLinkError } from "./cardboard-profile.js";
import { GlViewer } from "./gl-viewer.js";
import { QrScanner, qrScanningSupported } from "./qr-scan.js";
import { loadPxPerMeter, isCalibrated, pxPerMeterSource, initCalibration } from "./calibration.js";

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
let pendingScene = null;

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

const CALIB_SOURCE_LABELS = {
  calibrated: "your calibration",
  "known-device-exact": "known-device auto-detect",
  "known-device-approximate": "known-device auto-detect, approximate — recalibrating recommended",
  default: "generic default",
};

function renderCalibSummary() {
  const sourceLabel = CALIB_SOURCE_LABELS[pxPerMeterSource()];
  calibReadoutSummary.textContent = `${Math.round(pxPerMeter)} device px/meter (${sourceLabel})`;
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
  // An uncalibrated device falls back to a guessed px/meter that can be wildly off for the
  // actual phone -- through real magnifying lenses even a modest mismatch there is enough to
  // make the two eye views fail to fuse ("overlapping", clipped off-screen). Require
  // calibration before the first VR view rather than silently rendering with that guess.
  if (!isCalibrated()) {
    pendingScene = scene;
    calibController.show();
    return;
  }
  await enterVrNow(scene);
}

async function enterVrNow(scene) {
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
  requestRedrawSoon(); // in case the phone is still mid-rotation into the headset right now

  if (canvas.requestFullscreen) {
    canvas.requestFullscreen().then(requestRedrawSoon).catch(() => {});
  } else if (canvas.webkitRequestFullscreen) {
    canvas.webkitRequestFullscreen();
  }
  if (screen.orientation && screen.orientation.lock) {
    screen.orientation.lock("landscape").then(requestRedrawSoon).catch(() => {});
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

// iOS Safari (and some other mobile browsers) fire "resize"/"orientationchange" with STALE
// window.innerWidth/innerHeight -- the values only settle a little later, after the browser
// chrome/layout transition actually finishes. Re-measuring immediately can capture a canvas
// size from the orientation the phone was just IN, not the one it's rotating INTO, which is
// enough on its own to make the eye split look badly wrong. Redraw once immediately for
// responsiveness, then again after a short settle delay to correct for that.
function requestRedrawSoon() {
  requestRedraw();
  setTimeout(requestRedraw, 250);
}

window.addEventListener("resize", requestRedrawSoon);
window.addEventListener("orientationchange", requestRedrawSoon);
document.addEventListener("fullscreenchange", requestRedrawSoon);
document.addEventListener("webkitfullscreenchange", requestRedrawSoon);
// visualViewport fires when mobile browser chrome (address bar, etc.) shows/hides, which
// changes the visible viewport size without a matching "resize" event in some browsers --
// without this, the canvas can end up sized for a viewport that no longer matches reality.
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", requestRedrawSoon);
}

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
// Renders a parse error into `statusEl` (a plain-text status element). For an
// UnresolvedLinkError (e.g. a QR code encoding a goo.gl short link this page can't follow
// itself due to CORS), adds a real, clickable link the user can open themselves -- a normal
// browser navigation isn't subject to CORS -- so they can copy back whatever URL it lands on.
function renderParseError(statusEl, err) {
  statusEl.textContent = "";
  if (err instanceof UnresolvedLinkError) {
    statusEl.append("✕ That's a link, but it has no Cardboard config in it. ");
    const link = document.createElement("a");
    link.href = err.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "Open it ↗";
    statusEl.append(link);
    statusEl.append(", then paste the URL it lands on above.");
  } else {
    statusEl.textContent = `✕ Could not parse that: ${err.message}`;
  }
}

el("manual-qr-apply").addEventListener("click", async () => {
  const text = el("manual-qr-input").value;
  const status = el("manual-qr-status");
  status.classList.remove("qr-status--error");
  status.textContent = "Resolving…";
  try {
    const p = await resolveCardboardProfileText(text);
    saveProfile(p);
    requestRedraw();
    status.textContent = "";
  } catch (err) {
    status.classList.add("qr-status--error");
    renderParseError(status, err);
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
  let keepOpenForLink = false;
  try {
    await scanner.start();
    setQrStatus("Scanning…", null);
    const text = await scanner.scanOnce();
    setQrStatus("Resolving…", null);
    const scanned = await resolveCardboardProfileText(text);
    saveProfile(scanned);
    requestRedraw();
    setQrStatus(`✓ Scanned "${scanned.vendor} ${scanned.model}"`, "success");
    await new Promise((r) => setTimeout(r, 1800));
  } catch (err) {
    if (err instanceof UnresolvedLinkError) {
      // Keep the panel open (with Cancel available) so the user has time to tap the link.
      keepOpenForLink = true;
      setQrStatus("", "error");
      renderParseError(el("qr-status"), err);
    } else {
      setQrStatus(`✕ Scan failed: ${err.message}`, "error");
      await new Promise((r) => setTimeout(r, 1500));
    }
  } finally {
    scanner.stop();
    if (!keepOpenForLink) {
      qrScanView.hidden = true;
      settingsPanel.hidden = false;
    }
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
  renderCalibSummary();
  if (pendingScene) {
    const scene = pendingScene;
    pendingScene = null;
    enterVrNow(scene);
  } else {
    settingsPanel.hidden = false;
    requestRedraw();
  }
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
