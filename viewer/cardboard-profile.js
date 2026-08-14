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

// Cache of short links already resolved by hand (e.g. via `curl -sIL <url>`) to their real
// cardboard/cfg?p=... payload, for viewer QR codes that encode a shortener link this page has
// no way to resolve itself at runtime (see UnresolvedLinkError below) -- following the redirect
// with a real browser tab doesn't work either, since google.com/cardboard/cfg itself does a
// client-side redirect to a query-string-free landing page before a human can copy anything.
// A physical product's QR code never changes, so each entry here is permanently valid once
// captured. Add new entries as: '<the exact scanned URL>': '<the resolved p= payload>'.
//
// This batch covers every viewer QR code listed at
// https://homido.com/en/les-principaux-qr-codes-pour-masques-vr-mobiles/ as of 2026-08, each
// resolved and decoded once to confirm the payload is sane before caching it.
//
// NOTE on accuracy: cross-checking against a real physical Homido Mini unit found its own
// QR-encoded distortion/screen-to-lens constants render noticeably worse (through the actual
// lenses) than the generic Cardboard v1 DEFAULT_PROFILE. Manufacturer-published constants
// aren't guaranteed accurate -- if a profile looks wrong once scanned, try "Reset to default
// profile" before assuming the app is at fault.
const KNOWN_SHORT_LINKS = {
  "https://goo.gl/GvHq4R": // Homido Mini -- see accuracy note above
    "CgZIT01JRE8SC0hvbWlkbyBtaW5pHYcWWT0ltvN9PSoQAABIQgAASEIAAEhCAABIQlgCNSlcDz06CI_C9T3NzMw9UABgAg",
  "https://goo.gl/6kjKrQ": // Homido ("VRStream"/"Homido")
    "CghWUlN0cmVhbRIGSG9taWRvHVg5ND0lj8J1PVgANbbz_Tw6CM3MTD6PwvU9UABgAA",
  "https://goo.gl/RvfCln": // Homido Grab ("HOMIDO"/"GRAB")
    "CgZIT01JRE8SBEdSQUIdCtcjPSWPwnU9KhAAAEhCAABIQgAASEIAAEhCWAE1KVwPPToIexSuPs3MzD1QAGAD",
  "https://goo.gl/cfliQY": // Homido V2
    "CgZIb21pZG8SCUhvbWlkbyBWMh2WQws9JbbzfT0qEAAASEIAAEhCAABIQgAASEJYATUpXA89Oghcj0I-CtejPVAAYAI",
  "https://goo.gl/tpw3Mc": // Homido Prime ("Homido"/"V3-62")
    "CghIb21pZG_CrhIFVjMtNjIdokU2PSW28309KhAAADRCAAA0QgAANEIAADRCWAA1KVwPPToIuB4FPs3MzD5QAGAC",
  "https://goo.gl/q5nR6m": // Archos VR / Colorcross (decodes as vendor "Rady")
    "CgRSYWR5EgRSYWR5HcP1KD4lzczMPSoQAABIQgAASEIAAEhCAABIQlgBNSlcDz06CArXIzwK1yM8UAFgAA",
  "https://goo.gl/Rw4kwC": // "Cardboard Official V1" ("Unofficial Cardboard Inc."/"The Classic")
    "ChlVbm9mZmljaWFsIENhcmRib2FyZCBJbmMuEgtUaGUgQ2xhc3NpYx3sUTg9JY_CdT0qEAAASEIAAEhCAABIQgAASEJYADVQjRc9OggAAAAAAAAAAFABYAE",
  "https://goo.gl/viHg5c": // "Cardboard Official V2" (decodes as vendor "VR Stream"/"Homido")
    "CglWUiBTdHJlYW0SBkhvbWlkbx1YOTQ9JY_CdT0qEAAASEIAAEhCAABIQgAASEJYATUpXA89OgjNzEw-zczMPVAAYAI",
  "https://goo.gl/t9yhyz": // "Cardboard DOMO" / "I am Cardboard V2" / "VR Box" (all the same link)
    "CgtWUlN0cmVhbS5mchIUQ2FyZGJvYXJkIFYyIG9mZmljZWwdd74fPSVvEoM9KhAAAEhCAABIQgAASEIAAEhCWAA1KVwPPToIexSuPs3MDD9QAGAD",
  "https://goo.gl/IN9uDu": // Freefly VR
    "CgtWUlN0cmVhbS5mchIKRnJlZUZseSBWUh0K1yM9JWiRbT0qEAAASEIAAEhCAABIQgAASEJYATUpXA89Ogh7FK4-CtejPVAAYAI",
  "https://goo.gl/zZbJU3": // I am Cardboard Giant
    "Cg5JIEFNIENhcmRib2FyZBIbSUFDIEdpYW50IENhcmRib2FyZCBIZWFkc2V0HX9qPD0lbxKDPSoQAABIQgAASEIAAEhCAABIQlgANVg5ND06CHE9ij5xPYo-UAFgAQ",
  "https://goo.gl/EzHx9W": // Merge VR
    "CgtWUlN0cmVhbS5mchIITWVyZ2UgVlIdAisHPSUj23k9KhAAAEhCAABIQgAASEIAAEhCWAA1KVwPPToIrkdhPo_C9bxQAGAC",
  "https://goo.gl/m6sseN": // One Plus Cardboard
    "CgdPbmVQbHVzEhVDYXJkYm9hcmQgVmlld2VyIHYxLjEdKVwPPSWPwnU9KhAAAEhCAABIQgAASEIAAEhCWAA1KVwPPToIAAAAAAAAAABQAGAD",
  "https://goo.gl/4amlp5": // VR Shinecon / KiX (decodes as vendor "Adaptive Designs"/"VRKiX")
    "ChBBZGFwdGl2ZSBEZXNpZ25zEgVWUktpWB228_08Jc3MTD0qEAAASEIAAEhCAABIQgAASEJYADUpXA89OggK1yM8CtcjvFAAYAA",
  "https://goo.gl/R1YBVa": // Wearality Sky
    "CglXZWFyYWxpdHkSEVdlYXJhbGl0eSBTa3kgMC4xHY_C9TwlAiuHPSoQAACWQgAAlkIAAJZCAACWQlgBNSlcDz06CI_C9T2amRk-UABgAg",
  "https://goo.gl/vvTUK3": // Zeiss VR One / VR One GX ("Carl Zeiss AG"/"VR ONE")
    "Cg1DYXJsIFplaXNzIEFHEgZWUiBPTkUdUI0XPSW28309KhAAAEhCAABIQgAASEIAAEhCWAE1KVwPPToIzczMPQAAgD9QAGAA",
};

// Some viewer manufacturers (e.g. Homido) print a QR code that encodes a shortened link
// (goo.gl/xxxxx) which itself redirects to the real .../cardboard/cfg?p=... URL, rather than
// encoding that URL directly. A page hosted on GitHub Pages can't follow that redirect itself
// (the browser's fetch() is blocked by CORS — google.com doesn't grant this origin permission
// to read the response), so unless it's a link already cached in KNOWN_SHORT_LINKS above, this
// is surfaced as a distinct error carrying the URL so the UI can point the user at it.
export class UnresolvedLinkError extends Error {
  constructor(url) {
    super(`"${url}" looks like a link but has no cardboard config in it — it may need to be opened first`);
    this.url = url;
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
      const known = KNOWN_SHORT_LINKS[url.toString()];
      if (known) {
        param = known;
      } else {
        throw new UnresolvedLinkError(url.toString());
      }
    }
  }
  const bytes = base64UrlToBytes(param);
  return decodeDeviceParams(bytes);
}

// Small Cloudflare Worker (see cloudflare_worker.js at the repo root) that follows a
// shortener redirect chain server-side, where this page's own fetch() would be blocked by
// CORS, and returns the last hop that still carries a query string (some chains, notably
// Google's own goo.gl -> cardboard/cfg -> arvr.google.com chain, end on a landing page that
// strips it). Only resolves hosts on the Worker's own allowlist (common URL shorteners) --
// it's not a general-purpose proxy.
const SHORT_LINK_RESOLVER_URL = "https://cardboard-resolver.zickler.workers.dev/";

// Like parseCardboardProfileText, but falls back to the resolver above for a short link
// that isn't already in KNOWN_SHORT_LINKS, instead of immediately giving up with
// UnresolvedLinkError. If the resolver call itself fails for any reason (network error,
// host not on its allowlist, etc.), re-throws the *original* UnresolvedLinkError so the
// existing "open this link yourself" UI fallback still works.
export async function resolveCardboardProfileText(text) {
  try {
    return parseCardboardProfileText(text);
  } catch (err) {
    if (!(err instanceof UnresolvedLinkError)) throw err;
    try {
      const res = await fetch(`${SHORT_LINK_RESOLVER_URL}?url=${encodeURIComponent(err.url)}`);
      if (!res.ok) throw new Error(`resolver returned HTTP ${res.status}`);
      const resolved = await res.json();
      if (resolved.error) throw new Error(resolved.error);
      return parseCardboardProfileText(resolved.bestUrl);
    } catch {
      throw err;
    }
  }
}
