// Scans a Cardboard viewer's QR code using the camera. Prefers the native BarcodeDetector API
// (Chrome/Android); falls back to the vendored jsQR decoder (viewer/vendor/jsQR.js) for
// browsers that lack it (notably iOS Safari), so scanning works on both platforms.

export class QrScanner {
  constructor(videoEl) {
    this.video = videoEl;
    this.stream = null;
    this._rafId = null;
    this._canvas = document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  stop() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
  }

  // Resolves with the decoded QR text, or rejects if stop() is called first.
  scanOnce() {
    if ("BarcodeDetector" in window) {
      return this._scanWithBarcodeDetector();
    }
    return this._scanWithJsQR();
  }

  async _scanWithBarcodeDetector() {
    const detector = new window.BarcodeDetector({ formats: ["qr_code"] });
    return new Promise((resolve, reject) => {
      const tick = async () => {
        if (!this.stream) return reject(new Error("scan cancelled"));
        try {
          const codes = await detector.detect(this.video);
          if (codes.length > 0) {
            resolve(codes[0].rawValue);
            return;
          }
        } catch (err) {
          // Transient decode errors are expected on empty/blurry frames; keep scanning.
        }
        this._rafId = requestAnimationFrame(tick);
      };
      tick();
    });
  }

  async _scanWithJsQR() {
    if (typeof window.jsQR !== "function") {
      throw new Error("QR scanning is unavailable: neither BarcodeDetector nor jsQR is loaded");
    }
    return new Promise((resolve, reject) => {
      const tick = () => {
        if (!this.stream) return reject(new Error("scan cancelled"));
        const w = this.video.videoWidth;
        const h = this.video.videoHeight;
        if (w > 0 && h > 0) {
          this._canvas.width = w;
          this._canvas.height = h;
          this._ctx.drawImage(this.video, 0, 0, w, h);
          const imageData = this._ctx.getImageData(0, 0, w, h);
          const result = window.jsQR(imageData.data, w, h);
          if (result && result.data) {
            resolve(result.data);
            return;
          }
        }
        this._rafId = requestAnimationFrame(tick);
      };
      tick();
    });
  }
}

export function qrScanningSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
