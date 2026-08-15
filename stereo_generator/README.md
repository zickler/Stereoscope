# generate_perceptual_stereo.py

Generates vector-graphics (SVG) and rasterized (PNG) stereo stimuli for three
classic perceptual-stereopsis displays plus a custom disparity-defined
figure-ground scene, along with per-scene ground truth (disparity maps,
segmentation, unsigned distance field). Python 3, standard library only —
no numpy/PIL/OpenCV required.

Camera convention (rectified stereo rig):

```
x_left  = x_cyclopean + disparity / 2
x_right = x_cyclopean - disparity / 2
disparity = focal_px * baseline / Z          (larger disparity = nearer)
```

## Papers reproduced

| Scene key | Paper | Figure(s) |
|---|---|---|
| `gillam2002` | Gillam, B., & Nakayama, K. (2002). *Subjective Contours at Line Terminations Depend on Scene Layout Analysis, Not Image Processing.* Journal of Experimental Psychology: Human Perception and Performance, 28(1), 43–53. (`SubjectiveContours-Gillam2002.pdf`) | Figure 9A — dotted near plane abutting a farther line texture ("forest" of independently placed lines, or the "planar" single tilted surface consistent with the subjective-contour percept the paper reports). |
| `belhumeur1996_dimple` / `belhumeur1996_pimple` | Belhumeur (1996), Figure 2 half-occlusion geometry. Derivation notes: `Belhumeur1996_3Dgeometry.pdf` (see also companion `Belhumeur1996_3Dgeometry.md`) | Figure 2 — the "pimple" (near disc in front of a mid-ground stadium) and "dimple" (stadium cutout revealing a far disc) half-occlusion configurations. |
| `andersonnakayama1994_wallpaper` | Anderson, B. L., & Nakayama, K. (1994). *Toward a General Theory of Stereopsis: Binocular Matching, Occluding Contours, and Fusion.* Psychological Review, 101(3), 414–445. (`AndersonNakayama1994.pdf`) | Figure 6 — the ambiguous wallpaper illusion: a periodic stripe pattern shifted by one half-cycle between the eyes, whose front/behind depth relative to the dotted surround is genuinely ambiguous, demonstrated across 4 surround-luminance conditions. |

## Custom scenes

| Scene key | Description |
|---|---|
| `svg_square` | Not from a published figure. A classic disparity-defined figure-ground display: a fronto-parallel square nearer than a fronto-parallel background plane, so the square is shifted by a different amount than the background between the two eyes. Both are textured from user-provided SVG files (defaulting to `cow_pattern.svg` for the square, `wavy_lines.svg` for the background) — any SVG matching the expected structure works. For the PNG outputs, source paths are parsed into polygons once and transformed analytically (crop window, tile offset, scale) before rasterizing at final resolution, so zooming in/out never produces resampling pixelation. The `.svg` outputs embed the source files verbatim via nested `<svg viewBox>` elements instead, reproducing them exactly. |

## Output files

Every scene writes a full suite under `--output-dir` (defaults to
`<scene-prefix>_outputs`), with filenames `<stem>_<kind>.<ext>` (`<stem>`
defaults to the scene prefix):

```
<stem>_cyclopean.svg / .png    cyclopean view
<stem>_left.svg / .png         physical left-eye view
<stem>_right.svg / .png        physical right-eye view
<stem>_crossed.svg / .png      side-by-side pair for crossed fusion
<stem>_parallel.svg / .png     side-by-side pair, ordinary L/R order
<stem>_disp.npy                cyclopean disparity map, float32
<stem>_disp_left.npy           left-view disparity map, float32
<stem>_seg.npy                 cyclopean segmentation, uint8
<stem>_seg_left.npy            left-view segmentation, uint8
<stem>_udf.npy                 unsigned distance field from the cyclopean SEG boundary, float32
<stem>_udf_left.npy            unsigned distance field from the left-view SEG boundary, float32
<stem>_udf_preview.svg / .png  vector/raster preview of the cyclopean SEG boundary
<stem>_meta.json               full scene/camera parameters
<stem>_summary.svg             left/cyclopean/right images, then left-view and cyclopean
                                (disp, seg, UDF) panels, in one labeled sheet
```

`gillam2002`, `belhumeur1996_{dimple,pimple}`, and `svg_square` each produce
exactly one `<stem>`. **`andersonnakayama1994_wallpaper` is ambiguous between two
depth interpretations of the exact same raw display** — "front" (wallpaper
nearer, occludes surround) and "behind" (wallpaper farther, seen through the
surround's aperture) — so there is no single "correct" ground-truth disparity
to pick, and both are always written. The two interpretations render
identically (same images — this really is one raw display, not two), so
those (and `_summary.svg`) are written once under the plain `<stem>`. But
per Anderson & Nakayama's own Figure 4 half-occlusion analysis, the two
interpretations disagree about which surface owns the border stripe on each
side of the patch: "front" treats it as a half-occlusion revealing the
farther surround (so the patch reads one stripe width narrower on each
side), while "behind" keeps the full patch width. That makes disparity,
segmentation, and the UDF derived from it all genuinely percept-dependent,
so `<stem>_front_disp.npy` / `_disp_left.npy` / `_seg.npy` / `_seg_left.npy` /
`_udf.npy` / `_udf_left.npy` / `_udf_preview.svg` / `_udf_preview.png` and
their `<stem>_behind_*` counterparts are each written separately per
interpretation, while
`<stem>_meta.json` records both interpretations' `wallpaper_z` /
`wallpaper_disparity_px` / `disparity_region_cyclopean_px` under a nested
`"variants"` key. Its default output-dir/stem also splice in the
surround-luminance preset (see below), so different luminance runs never
collide.

## Generating the figures

### Gillam & Nakayama 2002, Figure 9A

```bash
# Default: lines coplanar on a single tilted plane (the percept the paper reports)
python3 generate_perceptual_stereo.py gillam2002

# Original "forest" variant: each line at its own independent depth
python3 generate_perceptual_stereo.py gillam2002 --gillam-lines-mode forest
```

### Belhumeur 1996, Figure 2

```bash
python3 generate_perceptual_stereo.py belhumeur1996_dimple
python3 generate_perceptual_stereo.py belhumeur1996_pimple
```

### Anderson & Nakayama 1994, Figure 6 (all 4 surround-luminance variations)

Each run already emits both the front and behind interpretation; loop over
the 4 luminance presets with a fixed `--wallpaper-dot-seed` so all four share
one consistent surround-dot layout:

```bash
for lum in dark mid_dark mid_light light; do
  python3 generate_perceptual_stereo.py andersonnakayama1994_wallpaper \
      --wallpaper-surround-luminance "$lum" --wallpaper-dot-seed 0
done
```

This writes four output directories (`andersonnakayama1994_wallpaper_dark_outputs`,
`..._mid_dark_outputs`, `..._mid_light_outputs`, `..._light_outputs`), each
containing one shared set of images/summary plus a `_front`/`_behind` pair
of disparity/segmentation/UDF files (see "Output files" above).

### SVG-Square (custom)

```bash
python3 generate_perceptual_stereo.py svg-square

# Zoom in on the square texture (fewer, bigger-looking spots) and reposition its crop window
python3 generate_perceptual_stereo.py svg-square --svgsquare-crop-scale 0.3 --svgsquare-crop-x-frac 0.2

# Swap in different SVG textures (each must fit the same shape this script expects: a
# <pattern>-filled canvas for the square's texture, or a flat <path> list for the background's)
python3 generate_perceptual_stereo.py svg-square \
    --svgsquare-square-texture my_texture.svg --svgsquare-background-texture my_background.svg
```

## Optional arguments

These tables summarize every flag and its default; `python3 generate_perceptual_stereo.py
--help` is the always-up-to-date source of truth (see below) if this drifts.

### Shared by every scene

| Flag | Default | Meaning |
|---|---|---|
| `--scene` | *(none)* | Alternative to the positional `scene` argument; overrides it if both given. |
| `--output-dir` | `<scene-prefix>_outputs` | Output folder. |
| `--stem` | scene prefix | Output filename stem; values without the scene prefix get it prepended automatically. |
| `--image-size H W` | `512 512` | Rendered image height and width, in pixels. |
| `--focal-px` | `300.0` | Camera focal length, in pixels. |
| `--baseline` | `0.10` | Stereo baseline (eye separation), in world units. |
| `--pair-gap-frac` | `0.16` | Gap between the two images in the `_crossed`/`_parallel` pair sheets, as a fraction of image width. |
| `--show-plane-outline` | off | Draw the (otherwise invisible) reference plane's outline, where applicable, for debugging. |
| `--show-boundary-preview` | off | Draw the otherwise-subjective plane-top boundary in the separate SVGs, where applicable, for debugging. |
| `--summary-columns` | `3` | Number of columns in the `_summary.svg` panel grid. |
| `--no-summary` | off | Skip writing the `_summary.svg` sheet. |

### `gillam2002`

| Flag | Default | Meaning |
|---|---|---|
| `--plane-z` | `1.40` | Depth of the dotted near plane. |
| `--gillam-lines-mode` | `planar` | `forest` (each of the 7 lines at its own independent depth from `--line-zs`) or `planar` (all 7 coplanar on one tilted plane; see `--plane-slant-*`). |
| `--line-zs` | `2.25,2.05,2.55,1.95,2.35,2.70,2.15` | Forest mode only. 7 comma-separated line depths, each greater than `--plane-z`. |
| `--plane-slant-tilt-x-deg` | `20.0` | Planar mode only. Tilt about the horizontal axis; positive tips the far (top) edge away from the camera. |
| `--plane-slant-tilt-y-deg` | `0.0` | Planar mode only. Tilt about the vertical axis; positive tips the right edge away from the camera. |
| `--plane-slant-ref-z` | `1.95` | Planar mode only. Depth at the nearest point of the forest plane's near edge; must exceed `--plane-z`. |
| `--plane-top-frac` | `0.385` | Top boundary of the dotted plane, as a fraction of image height. |
| `--dot-radius-frac` | `0.0105` | Dot radius as a fraction of `min(image height, width)`. |
| `--line-stroke-frac` | `0.0060` | Line stroke width as a fraction of `min(image height, width)`. |

### `belhumeur1996_dimple` / `belhumeur1996_pimple`

| Flag | Default | Meaning |
|---|---|---|
| `--figure2-d1-px` | 7% of `min(H, W)` | Foreground disc disparity d1, in pixels. |
| `--figure2-dimg-px` | 20% of `min(H, W)` | Rendered disc diameter D_img, in pixels. |
| `--figure2-w` | `0.12` | Relative monocular crescent width w, as a fraction of D_img. |
| `--figure2-gray` | `96` | 0–255 grayscale value for the far-plane grey surface. |

### `andersonnakayama1994_wallpaper`

| Flag | Default | Meaning |
|---|---|---|
| `--wallpaper-surround-luminance` | `dark` | One of `dark`/`mid_dark`/`mid_light`/`light`; approximates the paper's 4 background conditions (shifts perceptual bias behind → bistable → front). Both `front`/`behind` ground truth are always generated regardless. |
| `--wallpaper-stripe-light-gray` | `235` | 0–255 grayscale value for light stripes. |
| `--wallpaper-stripe-dark-gray` | `40` | 0–255 grayscale value for dark stripes. |
| `--wallpaper-patch-width-frac` | `0.20` | Total wallpaper patch width (all stripes combined) as a fraction of image height. Stripe width is this divided by `--wallpaper-stripe-count`, which also fixes the interocular shift (one stripe width = one half cycle) that makes front/behind equally valid — not independently configurable. |
| `--wallpaper-stripe-count` | `10` | Number of stripes (Figure 6 uses 10). |
| `--wallpaper-patch-height-frac` | `0.10` | Patch height as a fraction of image height. |
| `--wallpaper-surround-z` | `1.00` | Reference depth for the surround; bounds how large the stripe-width shift can be. |
| `--wallpaper-dot-count` | `500` | Number of sparse black dots scattered over the surround. |
| `--wallpaper-dot-radius-frac` | `0.005` | Surround dot radius as a fraction of `min(H, W)`. |
| `--wallpaper-dot-seed` | `0` | Random seed for the surround dot layout (deterministic). |

### `svg_square`

| Flag | Default | Meaning |
|---|---|---|
| `--svgsquare-square-size-frac` | `0.20` | Square size as a fraction of `min(image height, width)`. |
| `--svgsquare-square-z` | `1.2` | Depth of the near square; must be nearer than `--svgsquare-background-z`. |
| `--svgsquare-background-z` | `1.6` | Depth of the far background plane. |
| `--svgsquare-square-texture` | `cow_pattern.svg` | SVG file textured onto the square (must have a `<pattern>` fill). |
| `--svgsquare-background-texture` | `wavy_lines.svg` | SVG file textured onto the background, scaled to fill the frame. |
| `--svgsquare-crop-x-frac` | `0.5` | Horizontal position (0–1) of the square's texture crop window within the available margin. |
| `--svgsquare-crop-y-frac` | `0.5` | Vertical position (0–1) of that crop window. |
| `--svgsquare-crop-scale` | `0.6` | Crop window size as a fraction of the square (texture-native units), scaled to fill the square. `1.0` crops at native scale (no resizing); `<1` zooms in (bigger/coarser spots); `>1` zooms out (smaller/finer spots). |

## Further help / customizing parameters

Full, current flag documentation (every default and every override) is
always available from argparse directly:

```bash
python3 generate_perceptual_stereo.py --help
```

A few things worth knowing when reading it:

- The positional `scene` argument (or `--scene`) accepts short aliases in
  addition to the canonical keys above — e.g. `gillam`/`figure9a`,
  `wallpaper`/`wallpaper1994`/`andersonnakayama`, `svgsquare`/`svg-square`.
  See `SCENE_ALIASES` in the script for the full mapping.
- Flags are namespaced by scene: `--plane-*`/`--line-*`/`--gillam-*` affect
  `gillam2002` only, `--figure2-*` affects the Belhumeur scenes only,
  `--wallpaper-*` affects the Anderson & Nakayama scene only, and
  `--svgsquare-*` affects the svg-square scene only; all four families share
  the generic camera/output flags (`--image-size`, `--focal-px`,
  `--baseline`, `--output-dir`, `--stem`, `--pair-gap-frac`,
  `--summary-columns`, `--no-summary`, etc.).
- To change a scene's registered defaults (rather than pass flags every
  time), edit the corresponding `parser.add_argument(...)` default in
  `main()`, or the numeric defaults inside the scene's `make_*_scene`/
  `build_*_scene` function if the value isn't exposed as a flag.
- New scenes are added by writing a `make_*_scene`/`build_*_scene` pair and
  registering a `SceneSpec` in `SCENE_SPECS` (see `andersonnakayama1994_wallpaper`
  for an example of a family that emits multiple ground-truth variants per
  invocation via `variant_build_scenes`, and `prefix_suffix` for splicing a
  run-specific token into the output naming). `svg_square` is an example of a
  scene textured from external SVG art rather than procedural primitives —
  see `_load_flat_svg_polygons`/`_load_pattern_svg_polygons` and
  `_read_svg_inner_markup` if building another one.
