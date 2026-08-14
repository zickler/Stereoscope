// Browsers expose no reliable API for a screen's physical size, so "auto-adjusting to the
// phone's display" is done via a one-time calibration: the user drags two handles to match
// the width of a real ISO/IEC 7810 ID-1 card (a credit card / driver's license, 85.60mm wide
// — something almost everyone has on hand) held against the screen. That yields real device
// pixels per meter, which is then reused for every scene on this device until recalibrated.

const CARD_WIDTH_M = 0.0856;
const STORAGE_KEY = "cardboardViewer.pxPerMeter";
const DEFAULT_PX_PER_METER = 401 / 0.0254; // ~401ppi, a reasonable modern-phone fallback

// Best-effort auto-detection for iPhones: Apple publishes exact ppi per model, and (unlike
// most Android OEMs) reliably pairs a given (CSS points, devicePixelRatio) combination with a
// single ppi across each model's lifetime, so this can stand in for the credit-card
// calibration for known devices. Keyed by [portrait CSS width, portrait CSS height, dpr] ->
// ppi. Explicit calibration (localStorage) always takes precedence over this when present.
const KNOWN_IPHONE_PPI = [
  [320, 480, 2, 326], // iPhone 4/4s
  [320, 568, 2, 326], // iPhone 5/5s/5c/SE (1st gen)
  [375, 667, 2, 326], // iPhone 6/6s/7/8/SE (2nd/3rd gen)
  [414, 736, 3, 401], // iPhone 6+/6s+/7+/8+
  [375, 812, 3, 458], // iPhone X/XS/11 Pro
  [414, 896, 2, 326], // iPhone XR/11
  [414, 896, 3, 458], // iPhone XS Max/11 Pro Max
  [390, 844, 3, 460], // iPhone 12/12 Pro/13/13 Pro/14
  [428, 926, 3, 458], // iPhone 12 Pro Max/13 Pro Max/14 Plus
  [393, 852, 3, 460], // iPhone 14 Pro/15/15 Pro
  [430, 932, 3, 460], // iPhone 14 Pro Max/15 Pro Max/15 Plus
];

// Best-effort auto-detection for Android: UNLIKE iPhones, Android's devicePixelRatio is
// rounded to a coarse density "bucket" rather than reflecting true density, and CSS
// dimensions/dpr can also change if the user adjusts system display-scaling settings (common
// on Android, rare on iOS). Concretely verified collisions: Pixel 6 (411ppi), 7 (416ppi), and
// 8 (428ppi) all report the IDENTICAL [412, 915, 2.625] signature; Galaxy S23 (422ppi) and S24
// (416ppi) both report [360, 780, 3]. So a match here is a *cluster* of models/generations
// spanning a real ppi range, not a precise identification -- each entry's ppi is the middle of
// that observed range. This is deliberately treated as lower-confidence than the iPhone table:
// it improves the pre-calibration default guess, but does NOT count as "calibrated" (see
// isCalibrated below) -- Android users still get the accurate credit-card calibration flow.
const KNOWN_ANDROID_PPI = [
  [412, 915, 2.625, 418], // Google Pixel 6/7/8 (verified range 411-428ppi)
  [360, 808, 3, 422], // Google Pixel 9
  [360, 780, 3, 419], // Samsung Galaxy S21/S22/S23/S24 compact tier (verified range 415-423ppi)
  [384, 854, 3.75, 515], // Samsung Galaxy S21 Ultra (and likely S22/S23 Ultra -- same tier, unverified)
];

function findMatch(table) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.min(window.screen.width, window.screen.height);
  const h = Math.max(window.screen.width, window.screen.height);
  const match = table.find(([mw, mh, mdpr]) => mw === w && mh === h && mdpr === dpr);
  return match ? match[3] / 0.0254 : null;
}

function guessExactPxPerMeter() {
  return findMatch(KNOWN_IPHONE_PPI);
}

function guessApproxPxPerMeter() {
  return findMatch(KNOWN_ANDROID_PPI);
}

export function loadPxPerMeter() {
  const stored = Number(localStorage.getItem(STORAGE_KEY));
  if (Number.isFinite(stored) && stored > 0) return stored;
  return guessExactPxPerMeter() || guessApproxPxPerMeter() || DEFAULT_PX_PER_METER;
}

export function savePxPerMeter(value) {
  localStorage.setItem(STORAGE_KEY, String(value));
}

// Only an explicit calibration or an exact (iPhone) device match are trusted enough to skip
// the mandatory calibration gate before first VR entry -- an approximate (Android) match is
// not, since a wrong guess there is still common enough to produce the "overlapping, off
// screen" failure mode real calibration exists to prevent.
export function isCalibrated() {
  return localStorage.getItem(STORAGE_KEY) !== null || guessExactPxPerMeter() !== null;
}

// For the Settings readout, so the user can tell whether the current px/meter came from their
// own calibration, an exact device match, an approximate one, or the generic fallback.
export function pxPerMeterSource() {
  if (localStorage.getItem(STORAGE_KEY) !== null) return "calibrated";
  if (guessExactPxPerMeter() !== null) return "known-device-exact";
  if (guessApproxPxPerMeter() !== null) return "known-device-approximate";
  return "default";
}

// Wires up a calibration overlay. `root` must contain:
//   .calib-track (drag area), .calib-left, .calib-right (handles), .calib-readout (text)
// Returns a controller with show()/hide() and fires onDone(pxPerMeter) when the user confirms.
export function initCalibration(root, onDone) {
  const track = root.querySelector(".calib-track");
  const left = root.querySelector(".calib-left");
  const right = root.querySelector(".calib-right");
  const readout = root.querySelector(".calib-readout");
  const confirmBtn = root.querySelector(".calib-confirm");

  const dpr = window.devicePixelRatio || 1;
  let leftFrac = 0.15;
  let rightFrac = 0.85;

  function trackWidthCss() {
    return track.getBoundingClientRect().width;
  }

  function render() {
    left.style.left = `${leftFrac * 100}%`;
    right.style.left = `${rightFrac * 100}%`;
    const widthDevicePx = Math.max(1, (rightFrac - leftFrac) * trackWidthCss() * dpr);
    const pxPerMeter = widthDevicePx / CARD_WIDTH_M;
    readout.textContent = `${Math.round(pxPerMeter)} device px/meter — drag the bars to the edges of a card held on screen`;
    root._pendingPxPerMeter = pxPerMeter;
  }

  function makeDraggable(handle, isLeft) {
    let dragging = false;
    const onMove = (clientX) => {
      const rect = track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      if (isLeft) leftFrac = Math.min(frac, rightFrac - 0.02);
      else rightFrac = Math.max(frac, leftFrac + 0.02);
      render();
    };
    handle.addEventListener("pointerdown", (e) => {
      dragging = true;
      handle.setPointerCapture(e.pointerId);
    });
    handle.addEventListener("pointermove", (e) => {
      if (dragging) onMove(e.clientX);
    });
    handle.addEventListener("pointerup", () => {
      dragging = false;
    });
  }

  makeDraggable(left, true);
  makeDraggable(right, false);
  confirmBtn.addEventListener("click", () => {
    savePxPerMeter(root._pendingPxPerMeter);
    onDone(root._pendingPxPerMeter);
  });

  render();
  return {
    show() {
      root.hidden = false;
      render();
    },
    hide() {
      root.hidden = true;
    },
  };
}
