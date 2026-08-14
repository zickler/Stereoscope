// Stereo distortion renderer: splits a canvas into left/right eye viewports and pre-warps
// each eye's source image with the inverse of the Cardboard viewer's lens pincushion
// distortion, so that after passing through the physical lens the image looks undistorted.
//
// Math: Cardboard's DeviceParams.distortion_coefficients define a pincushion function
// mapping a point on the real (rendered) screen to where the eye perceives it after the
// lens, in tan-angle units relative to the lens optical center:
//     p' = p * (1 + K1 r^2 + K2 r^4 + ...),   r = |p| (tan-angle)
// To cancel that out, we pre-distort: for a fragment at real-screen tan-angle position p,
// sample the source image at the *same* forward-mapped position p' = p * (1 + K1 r^2 + ...).
// This repo's images are themselves pinhole-camera renders (see CameraRig in
// generate_perceptual_stereo.py), so pixel (x, y) already corresponds to tan-angle
// ((x - cx) / focal_px, (y - cy) / focal_px) — the same units — which lets us map distorted
// tan-angle directly to source pixel coordinates without any extra calibration step.

const VERTEX_SRC = `
attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SRC = `
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_opticalCenterPx;
uniform float u_pxPerMeter;
uniform float u_screenToLensDistance;
uniform vec4 u_distortion;
uniform float u_srcFocalPx;
uniform vec2 u_srcCenterPx;
uniform vec2 u_srcSizePx;

void main() {
  vec2 offsetPx = gl_FragCoord.xy - u_opticalCenterPx;
  vec2 offsetM = offsetPx / u_pxPerMeter;
  vec2 tanAngle = offsetM / u_screenToLensDistance;
  float r2 = dot(tanAngle, tanAngle);
  float k = 1.0 + u_distortion.x * r2 + u_distortion.y * r2 * r2
                + u_distortion.z * r2 * r2 * r2 + u_distortion.w * r2 * r2 * r2 * r2;
  vec2 distortedTan = tanAngle * k;
  // gl_FragCoord.y increases upward (WebGL window coords); image row-y increases downward.
  vec2 srcPx = vec2(
    u_srcCenterPx.x + distortedTan.x * u_srcFocalPx,
    u_srcCenterPx.y - distortedTan.y * u_srcFocalPx
  );
  vec2 uv = srcPx / u_srcSizePx;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    gl_FragColor = texture2D(u_tex, uv);
  }
}
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader compile failed: ${log}`);
  }
  return shader;
}

export class GlViewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false }) ||
      canvas.getContext("experimental-webgl", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL is not available on this device/browser");
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`program link failed: ${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;

    this.locations = {
      position: gl.getAttribLocation(program, "a_position"),
      tex: gl.getUniformLocation(program, "u_tex"),
      opticalCenterPx: gl.getUniformLocation(program, "u_opticalCenterPx"),
      pxPerMeter: gl.getUniformLocation(program, "u_pxPerMeter"),
      screenToLensDistance: gl.getUniformLocation(program, "u_screenToLensDistance"),
      distortion: gl.getUniformLocation(program, "u_distortion"),
      srcFocalPx: gl.getUniformLocation(program, "u_srcFocalPx"),
      srcCenterPx: gl.getUniformLocation(program, "u_srcCenterPx"),
      srcSizePx: gl.getUniformLocation(program, "u_srcSizePx"),
    };

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    this.leftTexture = gl.createTexture();
    this.rightTexture = gl.createTexture();
    this.leftMeta = null;
    this.rightMeta = null;
  }

  resize() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  _uploadTexture(texture, image) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  }

  // meta: {focalPx, cx, cy, width, height}
  setEyeImage(eye, image, meta) {
    this._uploadTexture(eye === "left" ? this.leftTexture : this.rightTexture, image);
    if (eye === "left") this.leftMeta = meta;
    else this.rightMeta = meta;
  }

  // profile: decoded CardboardProfile (see cardboard-profile.js)
  // pxPerMeter: calibrated device-pixel density
  render(profile, pxPerMeter) {
    const gl = this.gl;
    if (!this.leftMeta || !this.rightMeta) return;
    this.resize();

    const W = this.canvas.width;
    const H = this.canvas.height;
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const interLensPx = profile.interLensDistance * pxPerMeter;
    const trayToLensPx = profile.trayToLensDistance * pxPerMeter;
    let opticalCenterY;
    if (profile.verticalAlignment === "TOP") {
      opticalCenterY = H - trayToLensPx; // gl_FragCoord.y grows upward; TOP tray = near screen top
    } else if (profile.verticalAlignment === "CENTER") {
      opticalCenterY = H / 2;
    } else {
      opticalCenterY = trayToLensPx; // BOTTOM tray, measured up from the bottom edge
    }

    const distortion = profile.distortionCoefficients;
    const distVec = [distortion[0] || 0, distortion[1] || 0, distortion[2] || 0, distortion[3] || 0];

    gl.useProgram(this.program);
    gl.enableVertexAttribArray(this.locations.position);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1i(this.locations.tex, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(this.locations.pxPerMeter, pxPerMeter);
    gl.uniform1f(this.locations.screenToLensDistance, profile.screenToLensDistance);
    gl.uniform4fv(this.locations.distortion, distVec);

    const eyes = [
      { eye: "left", x: 0, opticalCenterX: W / 2 - interLensPx / 2, texture: this.leftTexture, meta: this.leftMeta },
      { eye: "right", x: W / 2, opticalCenterX: W / 2 + interLensPx / 2, texture: this.rightTexture, meta: this.rightMeta },
    ];
    for (const e of eyes) {
      gl.viewport(e.x, 0, W / 2, H);
      gl.bindTexture(gl.TEXTURE_2D, e.texture);
      gl.uniform2f(this.locations.opticalCenterPx, e.opticalCenterX, opticalCenterY);
      gl.uniform1f(this.locations.srcFocalPx, e.meta.focalPx);
      gl.uniform2f(this.locations.srcCenterPx, e.meta.cx, e.meta.cy);
      gl.uniform2f(this.locations.srcSizePx, e.meta.width, e.meta.height);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
}
