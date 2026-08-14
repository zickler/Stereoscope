# Cardboard Stereoscope

A web-based viewer for viewing perceptual-stereopsis stimuli through any
Google-Cardboard-compatible viewer, inspired by
[vision.seas.harvard.edu/stereoscope](https://vision.seas.harvard.edu/stereoscope/).
Live at: https://zickler.github.io/Stereoscope/viewer/

Unlike a flat side-by-side viewer, this scans your specific Cardboard-style
headset's QR code and pre-warps each eye's image with the inverse of that
viewer's lens distortion (decoded from the QR's embedded `DeviceParams`
protobuf), calibrated to your phone's actual display density.

## Using it

1. Open the live URL above **in Safari on iPhone** (required for true
   fullscreen — see "iOS fullscreen" below) or any browser on Android.
2. Tap the gear icon (⚙) → **Scan viewer's QR code**, point the camera at
   your Cardboard viewer's QR code. If it's a viewer not already recognized,
   see "Unknown QR codes" below.
3. Pick a scene from the gallery → **Enter VR**. The first time on a given
   phone, you'll be walked through a one-time display calibration (drag two
   bars to match a credit card's width against the screen) — browsers have
   no API for a screen's true physical size, so this stands in for it.
4. Put the phone in your viewer and look.

### iOS fullscreen

iOS Safari doesn't support the Fullscreen API for arbitrary page content, so
opening the link directly in Safari always shows the URL bar, which throws
off the lens-alignment math. Instead: **Share → Add to Home Screen**, then
launch it from that home-screen icon — that runs it in a real fullscreen
standalone mode.

### Unknown QR codes

Some viewers' QR codes encode a shortened link (`goo.gl/...`) rather than
the device-config URL directly, which this static site can't resolve itself
(a browser can't follow a cross-origin redirect without the target's
permission). A small Cloudflare Worker (`cloudflare_worker.js`, deployed at
`cardboard-resolver.zickler.workers.dev`) resolves those server-side
automatically — most common viewers are also pre-cached directly in
`viewer/cardboard-profile.js` so they resolve instantly with no network call
at all.

## How it's built

- `viewer/` — the app itself: a plain HTML/JS/CSS page (no framework, no
  build step). `cardboard-profile.js` decodes the QR's protobuf payload,
  `gl-viewer.js` does the actual per-eye lens-distortion WebGL rendering,
  `calibration.js` handles the display-calibration flow, `qr-scan.js` drives
  the camera.
- `viewer/scenes.json` — the scene gallery, fetched at runtime rather than
  hardcoded, so new scenes never require touching viewer code.
- `<scene>_outputs/` — the actual stereo images (`_cyclopean.png` for the
  gallery thumbnail, `_left.png`/`_right.png` for the two eyes). Only these
  three files per scene are kept here; ground-truth disparity/segmentation
  data and vector `.svg` versions stay in the source research repo and
  aren't needed by the viewer.
- `generate_perceptual_stereo.py` / `build_manifest.py` — copied in from the
  source research repo. `build_manifest.py` scans `*_outputs/` directories
  and (re)writes `viewer/scenes.json`.
- `cloudflare_worker.js` — source of truth for the deployed Worker (see
  above). Cloudflare's dashboard editor doesn't pull from git, so if you
  change this file, paste the new contents into the Worker's online editor
  and redeploy manually.

## Adding or updating a scene's images

This repo doesn't generate images itself — it just hosts a pruned copy of
whatever the source research repo produced. After regenerating a scene
there:

```bash
# In the source research repo:
python3 generate_perceptual_stereo.py <scene>
python3 build_manifest.py   # only needed if camera params changed, or a scene was added/removed

# Copy just what the viewer needs into this repo:
cp <scene>_outputs/<stem>_cyclopean.png \
   <scene>_outputs/<stem>_left.png \
   <scene>_outputs/<stem>_right.png \
   /path/to/Stereoscope/<scene>_outputs/
cp viewer/scenes.json /path/to/Stereoscope/viewer/scenes.json   # if it changed

# In this repo:
git add <scene>_outputs viewer/scenes.json
git commit -m "Update <scene> images"
git push origin main
```

GitHub Pages rebuilds automatically after the push, usually within a
minute. If you don't see the update on a phone that has this page **added
to its home screen**, that standalone instance runs its own separate cache
— force-quit it from the app switcher and relaunch (see the source repo's
notes for other options).
