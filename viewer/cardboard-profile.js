// Decoder for Google Cardboard viewer QR-code profiles.
//
// The QR code printed on/with a "Works with Google Cardboard" viewer encodes a URL of the
// form `https://google.com/cardboard/cfg?p=<base64url protobuf bytes>`. The protobuf payload
// is a `cardboard.DeviceParams` message, schema (proto2, LITE_RUNTIME):
//   https://github.com/googlevr/cardboard/blob/master/proto/cardboard_device.proto
//
// There is no official Google web/JS SDK (developers.google.com/cardboard only ships
// Android/iOS/Unity SDKs), so this is a from-scratch decoder of that wire format. Field
// numbers below were cross-checked by hand-decoding a real "Cardboard I/O 2015" QR payload
// byte-for-byte against the .proto source; see the commit that added this file for the
// verification transcript.
//
//   1  vendor                          string
//   2  model                           string
//   3  screen_to_lens_distance         float   (meters)
//   4  inter_lens_distance             float   (meters)
//   5  left_eye_field_of_view_angles   packed float[4]  (left, right, bottom, top; degrees)
//   6  tray_to_lens_distance           float   (meters)
//   7  distortion_coefficients         packed float[]   (pincushion polynomial K1, K2, ...)
//  11  vertical_alignment              enum    (0=BOTTOM, 1=CENTER, 2=TOP)
//  12  primary_button                  enum    (0=NONE, 1=MAGNET, 2=TOUCH, 3=INDIRECT_TOUCH)
//
// Unknown field numbers (e.g. field 10, seen in the wild but absent from the current .proto)
// are decoded generically and ignored.

const VERTICAL_ALIGNMENT = ["BOTTOM", "CENTER", "TOP"];
const BUTTON_TYPE = ["NONE", "MAGNET", "TOUCH", "INDIRECT_TOUCH"];

// Cardboard v1 nominal defaults (used until a real viewer QR is scanned).
export const DEFAULT_PROFILE = Object.freeze({
  vendor: "Google",
  model: "Cardboard v1 (default — no viewer scanned yet)",
  screenToLensDistance: 0.042,
  interLensDistance: 0.06,
  trayToLensDistance: 0.035,
  fovAngles: [40, 40, 40, 40],
  distortionCoefficients: [0.441, 0.156],
  verticalAlignment: "BOTTOM",
  primaryButton: "MAGNET",
});

function readVarint(bytes, pos) {
  let result = 0;
  let shift = 0;
  for (;;) {
    const b = bytes[pos++];
    result |= (b & 0x7f) << shift;
    if (!(b & 0x80)) break;
    shift += 7;
  }
  return [result >>> 0, pos];
}

// Generic protobuf wire-format scan: returns {fieldNumber: [rawValue, ...]}, where rawValue is
// a number for wire types 0/1/5, or a Uint8Array for wire type 2 (length-delimited).
function decodeProtoFields(bytes) {
  const fields = {};
  let pos = 0;
  while (pos < bytes.length) {
    let tag;
    [tag, pos] = readVarint(bytes, pos);
    const fieldNo = tag >>> 3;
    const wireType = tag & 0x7;
    let value;
    if (wireType === 0) {
      [value, pos] = readVarint(bytes, pos);
    } else if (wireType === 1) {
      value = new DataView(bytes.buffer, bytes.byteOffset + pos, 8).getFloat64(0, true);
      pos += 8;
    } else if (wireType === 2) {
      let len;
      [len, pos] = readVarint(bytes, pos);
      value = bytes.subarray(pos, pos + len);
      pos += len;
    } else if (wireType === 5) {
      value = new DataView(bytes.buffer, bytes.byteOffset + pos, 4).getFloat32(0, true);
      pos += 4;
    } else {
      throw new Error(`unsupported protobuf wire type ${wireType} (field ${fieldNo})`);
    }
    (fields[fieldNo] || (fields[fieldNo] = [])).push(value);
  }
  return fields;
}

function unpackFloats(chunk) {
  const dv = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  const out = [];
  for (let i = 0; i + 4 <= chunk.byteLength; i += 4) out.push(dv.getFloat32(i, true));
  return out;
}

export function decodeDeviceParams(bytes) {
  const fields = decodeProtoFields(bytes);
  const str = (n) => (fields[n] ? new TextDecoder().decode(fields[n][0]) : undefined);
  const num = (n) => (fields[n] ? fields[n][0] : undefined);
  const floats = (n) => (fields[n] ? unpackFloats(fields[n][0]) : undefined);

  const profile = { ...DEFAULT_PROFILE };
  if (str(1) !== undefined) profile.vendor = str(1);
  if (str(2) !== undefined) profile.model = str(2);
  if (num(3) !== undefined) profile.screenToLensDistance = num(3);
  if (num(4) !== undefined) profile.interLensDistance = num(4);
  const fov = floats(5);
  if (fov && fov.length === 4) profile.fovAngles = fov;
  if (num(6) !== undefined) profile.trayToLensDistance = num(6);
  const dist = floats(7);
  if (dist && dist.length > 0) profile.distortionCoefficients = dist;
  if (num(11) !== undefined) profile.verticalAlignment = VERTICAL_ALIGNMENT[num(11)] || "BOTTOM";
  if (num(12) !== undefined) profile.primaryButton = BUTTON_TYPE[num(12)] || "MAGNET";
  return profile;
}

function base64UrlToBytes(param) {
  const b64 = param.replace(/-/g, "+").replace(/_/g, "/").replace(/\s+/g, "");
  const padded = b64 + "=".repeat((4 - (b64.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Some viewer manufacturers (e.g. Homido) print a QR code that encodes a shortened link
// (goo.gl/xxxxx) which itself redirects to the real .../cardboard/cfg?p=... URL, rather than
// encoding that URL directly. A page hosted on GitHub Pages can't follow that redirect itself
// (the browser's fetch() is blocked by CORS — google.com doesn't grant this origin permission
// to read the response), so this is surfaced as a distinct error the UI can offer a real,
// user-driven fix for: open the link in a normal browser tab (which isn't subject to CORS)
// and paste back whatever URL it lands on.
export class UnresolvedLinkError extends Error {
  constructor(url) {
    super(`"${url}" looks like a link but has no cardboard config in it — it may need to be opened first`);
    this.url = url;
  }
}

// Matches a bare "hostname.tld/..." with no scheme, e.g. "goo.gl/GvHq4R". Deliberately
// requires a dot before the first path/query separator: base64url payloads (this function's
// other candidate interpretation) use the alphabet [A-Za-z0-9_-], which never contains ".",
// so this can't misfire on a raw pasted payload -- the WHATWG URL parser is otherwise happy to
// treat *any* such string as a valid (bogus) hostname, which would misclassify every raw paste.
const BARE_HOSTNAME_RE = /^[a-z0-9-]+(\.[a-z0-9-]+)+(?=[/?#]|$)/i;

function tryParseUrl(text) {
  try {
    return new URL(text);
  } catch {
    // no scheme -- only worth retrying if it actually looks like a bare hostname+path.
  }
  if (!BARE_HOSTNAME_RE.test(text)) return null;
  try {
    return new URL(`https://${text}`);
  } catch {
    return null;
  }
}

// Accepts a full scanned QR URL (http(s)://google.com/cardboard/cfg?p=..., the newer
// https://arvr.google.com/cardboard/... form, or a bare "google.com/cardboard/cfg?p=..." with
// no scheme), or a bare base64 `p` payload.
export function parseCardboardProfileText(text) {
  const trimmed = text.trim();
  const url = tryParseUrl(trimmed);
  let param = trimmed;
  if (url) {
    const p = url.searchParams.get("p");
    if (p) {
      param = p;
    } else {
      throw new UnresolvedLinkError(url.toString());
    }
  }
  const bytes = base64UrlToBytes(param);
  return decodeDeviceParams(bytes);
}
