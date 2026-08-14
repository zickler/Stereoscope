// Browsers expose no reliable API for a screen's physical size, so "auto-adjusting to the
// phone's display" is done via a one-time calibration: the user drags two handles to match
// the width of a real ISO/IEC 7810 ID-1 card (a credit card / driver's license, 85.60mm wide
// — something almost everyone has on hand) held against the screen. That yields real device
// pixels per meter, which is then reused for every scene on this device until recalibrated.

const CARD_WIDTH_M = 0.0856;
const STORAGE_KEY = "cardboardViewer.pxPerMeter";
const DEFAULT_PX_PER_METER = 401 / 0.0254; // ~401ppi, a reasonable modern-phone fallback

export function loadPxPerMeter() {
  const stored = Number(localStorage.getItem(STORAGE_KEY));
  return Number.isFinite(stored) && stored > 0 ? stored : DEFAULT_PX_PER_METER;
}

export function savePxPerMeter(value) {
  localStorage.setItem(STORAGE_KEY, String(value));
}

export function isCalibrated() {
  return localStorage.getItem(STORAGE_KEY) !== null;
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
