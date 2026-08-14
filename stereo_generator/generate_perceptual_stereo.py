#!/usr/bin/env python3
"""
Generate vector-graphics stereo stimuli for perceptual-science 3D scenes.

The first registered scene is modeled on Figure 9A of Gillam & Nakayama's
Subjective Contours paper. Additional scenes can be added by registering a new
SceneSpec and a scene-builder function.

The scene is defined in 3D and projected into a rectified stereo rig matching
the convention used by generate_stereo_slant.py:

    x_left  = x_cyclopean + disparity / 2
    x_right = x_cyclopean - disparity / 2
    disparity = focal_px * baseline / Z

Figure 9A is intended for crossed fusion. Accordingly, this script writes
separate physical left/right eye images, plus a crossed-fusion stereo sheet in
which the right-eye image is placed in the left column and the left-eye image is
placed in the right column.

Outputs are consolidated directly under --output-dir, where <stem> defaults
to the selected scene prefix:
    <stem>_cyclopean.svg       cyclopean SVG
    <stem>_left.svg            physical left-eye SVG
    <stem>_right.svg           physical right-eye SVG
    <stem>_crossed.svg         pair sheet for crossed fusion
    <stem>_parallel.svg        pair sheet in ordinary left/right order
    <stem>_cyclopean.png       rasterized cyclopean image
    <stem>_left.png            rasterized physical left-eye image
    <stem>_right.png           rasterized physical right-eye image
    <stem>_crossed.png         rasterized crossed-fusion pair
    <stem>_parallel.png        rasterized parallel-order pair
    <stem>_disp.npy            cyclopean disparity map, float32 .npy
    <stem>_disp_left.npy       left-view disparity map, float32 .npy
    <stem>_seg.npy             cyclopean segmentation, uint8 .npy
    <stem>_udf.npy             unsigned distance field from SEG, float32 .npy
    <stem>_udf_preview.svg     vector preview of segmentation boundaries
    <stem>_udf_preview.png     rasterized UDF preview
    <stem>_meta.json           scene/camera JSON
    <stem>_summary.svg         labeled all-outputs SVG

The script intentionally uses only the Python standard library so the SVG,
PNG, and .npy outputs can be generated without PIL, OpenCV, or NumPy installed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import html
import json
import math
import os
import random
import re
import struct
import sys
import zlib
from array import array
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SCENE_KEY = "gillam2002"
GILLAM_PLANAR_LINES_LABEL = 2
WALLPAPER_SURROUND_LABEL = 1
WALLPAPER_STRIPE_LABEL = 2

# Illustrative 0-255 gray values approximating the 4 background conditions in
# Anderson & Nakayama (1994) Figure 6 (dark -> behind-biased, light -> front-biased,
# middle two relatively bistable). The paper does not publish exact luminance
# values, so these are reasonable stand-ins, not a precise replication.
WALLPAPER_SURROUND_LUMINANCE_PRESETS = {
    "dark": 25,
    "mid_dark": 95,
    "mid_light": 160,
    "light": 230,
}


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Point2:
    x: float
    y: float


@dataclass(frozen=True)
class Dot:
    center: Point3
    radius_world: float


@dataclass(frozen=True)
class LineSegment3D:
    p0: Point3
    p1: Point3
    z: float
    stroke_px: float
    label: int


@dataclass(frozen=True)
class SlantedPlane:
    """A plane N.X = k in world coordinates, used to place the Gillam 2002
    "forest" lines coplanar on a single slanted surface behind the dotted
    plane. Built from a tilt about the horizontal axis (tilt_x_deg, positive
    tips the far edge away from the camera), a tilt about the vertical axis
    (tilt_y_deg, positive tips the right edge away from the camera), and the
    reference depth at the nearest point along the near (boundary) edge.
    """

    nx: float
    ny: float
    nz: float
    k: float
    tilt_x_deg: float
    tilt_y_deg: float
    ref_z: float


@dataclass(frozen=True)
class Scene:
    width: int
    height: int
    plane_z: float
    plane_top_y: float
    dot_radius_px: float
    line_stroke_px: float
    dots: List[Dot]
    lines: List[LineSegment3D]
    mode: str = "forest"
    lines_plane: Optional[SlantedPlane] = None


@dataclass(frozen=True)
class HalfOcclusionScene:
    width: int
    height: int
    center: Point2
    variant: str
    percept_label: str
    disc_diameter_px: float
    crescent_fraction: float
    crescent_width_px: float
    disc_z: float
    stadium_z: float
    background_z: float
    disc_disparity_px: float
    stadium_disparity_px: float
    background_disparity_px: float
    near_center_offset_px: float
    far_center_offset_px: float
    near_z: float
    far_z: float
    near_disparity_px: float
    far_disparity_px: float
    near_radius_px: float
    near_stadium_half_length_px: float
    far_disc_radius_px: float
    far_stadium_half_length_px: float
    stroke_px: float
    ring_radii_px: Tuple[float, ...]
    gray_color: Tuple[int, int, int]


@dataclass(frozen=True)
class WallpaperScene:
    """A single self-consistent depth interpretation of an Anderson & Nakayama
    1994 Figure 6 ambiguous wallpaper pattern: alternating light/dark stripes
    (the "wallpaper"), shifted by exactly one stripe width (one half cycle)
    between the two eyes, embedded in a gray surround. Because the shift
    equals half the stripe period, either sign of the shift equally well
    explains the raw pixels -- this codebase generates the "front" (wallpaper
    nearer, occluding the surround) and "behind" (wallpaper farther, occluded
    by the surround's aperture) interpretations as two separate WallpaperScene
    instances, each fully self-consistent on its own.
    """

    width: int
    height: int
    percept: str
    surround_gray: int
    stripe_light_gray: int
    stripe_dark_gray: int
    stripe_width_px: float
    stripe_count: int
    patch_left_px: float
    patch_right_px: float
    patch_top_px: float
    patch_bottom_px: float
    surround_z: float
    surround_disparity_px: float
    wallpaper_z: float
    wallpaper_disparity_px: float
    shift_px: float
    luminance_preset: str
    dots: Tuple[Point2, ...]
    dot_radius_px: float
    dot_seed: int


@dataclass(frozen=True)
class NpyArray:
    shape: Tuple[int, ...]
    descr: str
    data: array


@dataclass(frozen=True)
class SvgAsset:
    path: str
    width: float
    height: float
    data_uri: str


@dataclass(frozen=True)
class ImageAsset:
    width: int
    height: int
    data_uri: str


@dataclass(frozen=True)
class Panel:
    title: str
    subtitle: str
    asset: Optional[object] = None
    text_lines: Optional[List[str]] = None


@dataclass(frozen=True)
class CameraRig:
    width: int
    height: int
    focal_px: float = 300.0
    baseline: float = 0.10

    @property
    def cx(self) -> float:
        return 0.5 * (self.width - 1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.height - 1)

    def disparity(self, z: float) -> float:
        return self.focal_px * self.baseline / z

    def image_to_world(self, x: float, y: float, z: float) -> Point3:
        return Point3(
            x=(x - self.cx) * z / self.focal_px,
            y=(y - self.cy) * z / self.focal_px,
            z=z,
        )

    def project(self, point: Point3, eye: str) -> Point2:
        if eye == "cyclopean":
            camera_x = 0.0
        elif eye == "left":
            camera_x = -0.5 * self.baseline
        elif eye == "right":
            camera_x = 0.5 * self.baseline
        else:
            raise ValueError(f"unknown eye {eye!r}")

        return Point2(
            x=self.focal_px * (point.x - camera_x) / point.z + self.cx,
            y=self.focal_px * point.y / point.z + self.cy,
        )


@dataclass(frozen=True)
class SceneSpec:
    key: str
    prefix: str
    summary_title: str
    source: str
    fusion: str
    build_scene: Optional[Callable[[CameraRig, argparse.Namespace], object]] = None
    # Used instead of build_scene by families that must emit multiple, mutually
    # exclusive ground-truth interpretations of the same kind of display in one
    # invocation (e.g. the ambiguous wallpaper illusion's front/behind pair).
    # Each entry's output gets written under "<stem>_<variant key>".
    variant_build_scenes: Optional[Dict[str, Callable[[CameraRig, argparse.Namespace], object]]] = None
    # For families whose naming should reflect a run-specific CLI choice (e.g. the
    # wallpaper surround-luminance preset), returns the extra token to splice into
    # the prefix: "<prefix>_<token>". Return "" / None for no extra token.
    prefix_suffix: Optional[Callable[[argparse.Namespace], Optional[str]]] = None

    def effective_prefix(self, args: argparse.Namespace) -> str:
        if self.prefix_suffix is not None:
            suffix = self.prefix_suffix(args)
            if suffix:
                return f"{self.prefix}_{suffix}"
        return self.prefix

    def default_output_dir(self, args: argparse.Namespace) -> str:
        return f"{self.effective_prefix(args)}_outputs"


def ensure_output_dir(out_root: str) -> None:
    os.makedirs(out_root, exist_ok=True)


def output_paths(out_root: str, stem: str) -> Dict[str, str]:
    return {
        "cyclopean": os.path.join(out_root, f"{stem}_cyclopean.svg"),
        "left": os.path.join(out_root, f"{stem}_left.svg"),
        "right": os.path.join(out_root, f"{stem}_right.svg"),
        "crossed": os.path.join(out_root, f"{stem}_crossed.svg"),
        "parallel": os.path.join(out_root, f"{stem}_parallel.svg"),
        "cyclopean_png": os.path.join(out_root, f"{stem}_cyclopean.png"),
        "left_png": os.path.join(out_root, f"{stem}_left.png"),
        "right_png": os.path.join(out_root, f"{stem}_right.png"),
        "crossed_png": os.path.join(out_root, f"{stem}_crossed.png"),
        "parallel_png": os.path.join(out_root, f"{stem}_parallel.png"),
        "disp": os.path.join(out_root, f"{stem}_disp.npy"),
        "disp_left": os.path.join(out_root, f"{stem}_disp_left.npy"),
        "seg": os.path.join(out_root, f"{stem}_seg.npy"),
        "field": os.path.join(out_root, f"{stem}_udf.npy"),
        "udf": os.path.join(out_root, f"{stem}_udf_preview.svg"),
        "udf_png": os.path.join(out_root, f"{stem}_udf_preview.png"),
        "metadata": os.path.join(out_root, f"{stem}_meta.json"),
        "summary": os.path.join(out_root, f"{stem}_summary.svg"),
    }


def _format_points(points: Sequence[Point2]) -> str:
    return " ".join(f"{p.x:.3f},{p.y:.3f}" for p in points)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )


def _svg_defs() -> str:
    return (
        "  <defs>\n"
        "    <style>\n"
        "      .dot { fill: #000; }\n"
        "      .line { stroke: #000; stroke-linecap: butt; vector-effect: non-scaling-stroke; }\n"
        "      .plane { fill: #fff; stroke: none; }\n"
        "      .boundary-preview { stroke: #d62728; stroke-width: 1.5; stroke-linecap: square; opacity: 0.70; }\n"
        "    </style>\n"
        "  </defs>\n"
    )


def _scene_svg_body(
    rig: CameraRig,
    scene: Scene,
    eye: str,
    *,
    show_plane_outline: bool = False,
    show_boundary_preview: bool = False,
    indent: str = "  ",
) -> str:
    w, h = scene.width, scene.height
    plane_z = scene.plane_z
    x_margin = 0.35 * w
    y_bottom = h + 0.35 * h

    plane_corners_cyc = [
        (-x_margin, scene.plane_top_y),
        (w + x_margin, scene.plane_top_y),
        (w + x_margin, y_bottom),
        (-x_margin, y_bottom),
    ]
    plane_corners = [
        rig.project(rig.image_to_world(x, y, plane_z), eye)
        for x, y in plane_corners_cyc
    ]

    stroke = "#bfbfbf" if show_plane_outline else "none"
    stroke_width = "1" if show_plane_outline else "0"
    lines = [
        f'{indent}<rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>\n',
        (
            f'{indent}<polygon class="plane" points="{_format_points(plane_corners)}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>\n'
        ),
    ]

    for dot in scene.dots:
        p = rig.project(dot.center, eye)
        r = rig.focal_px * dot.radius_world / dot.center.z
        lines.append(f'{indent}<circle class="dot" cx="{p.x:.3f}" cy="{p.y:.3f}" r="{r:.3f}"/>\n')

    for seg in scene.lines:
        p0 = rig.project(seg.p0, eye)
        p1 = rig.project(seg.p1, eye)
        lines.append(
            f'{indent}<line class="line" x1="{p0.x:.3f}" y1="{p0.y:.3f}" '
            f'x2="{p1.x:.3f}" y2="{p1.y:.3f}" '
            f'stroke-width="{seg.stroke_px:.3f}"/>\n'
        )

    if show_boundary_preview:
        left = rig.project(rig.image_to_world(-x_margin, scene.plane_top_y, plane_z), eye)
        right = rig.project(rig.image_to_world(w + x_margin, scene.plane_top_y, plane_z), eye)
        lines.append(
            f'{indent}<line class="boundary-preview" x1="{left.x:.3f}" y1="{left.y:.3f}" '
            f'x2="{right.x:.3f}" y2="{right.y:.3f}"/>\n'
        )

    return "".join(lines)


def half_occlusion_projected_center(scene: HalfOcclusionScene, eye: str, plane: str) -> Point2:
    if plane == "near":
        disparity = scene.near_disparity_px
        center_offset = scene.near_center_offset_px
    elif plane == "far":
        disparity = scene.far_disparity_px
        center_offset = scene.far_center_offset_px
    else:
        raise ValueError(f"unknown plane {plane!r}")

    if eye == "cyclopean":
        offset_x = 0.0
    elif eye == "left":
        offset_x = 0.5 * disparity
    elif eye == "right":
        offset_x = -0.5 * disparity
    else:
        raise ValueError(f"unknown eye {eye!r}")

    return Point2(scene.center.x + center_offset + offset_x, scene.center.y)


def half_occlusion_ring_center(scene: HalfOcclusionScene, eye: str) -> Point2:
    return half_occlusion_projected_center(scene, eye, "near" if scene.variant == "pimple" else "far")


def _half_occlusion_svg_body(
    scene: HalfOcclusionScene,
    eye: str,
    *,
    indent: str = "  ",
) -> str:
    parts = [
        f'{indent}<rect x="0" y="0" width="{scene.width}" height="{scene.height}" fill="#fff"/>\n'
    ]

    if scene.variant == "pimple":
        far_center = half_occlusion_projected_center(scene, eye, "far")
        near_center = half_occlusion_projected_center(scene, eye, "near")
        stadium_x = far_center.x - scene.far_stadium_half_length_px
        stadium_y = far_center.y - scene.near_radius_px
        stadium_w = 2.0 * scene.far_stadium_half_length_px
        stadium_h = 2.0 * scene.near_radius_px
        gray = scene.gray_color
        parts.append(
            f'{indent}<rect x="{stadium_x:.3f}" y="{stadium_y:.3f}" '
            f'width="{stadium_w:.3f}" height="{stadium_h:.3f}" '
            f'rx="{scene.near_radius_px:.3f}" ry="{scene.near_radius_px:.3f}" '
            f'fill="rgb({gray[0]},{gray[1]},{gray[2]})"/>\n'
        )
        parts.append(
            f'{indent}<circle cx="{near_center.x:.3f}" cy="{near_center.y:.3f}" '
            f'r="{scene.near_radius_px:.3f}" fill="#fff"/>\n'
        )
    elif scene.variant == "dimple":
        near_center = half_occlusion_projected_center(scene, eye, "near")
        far_center = half_occlusion_projected_center(scene, eye, "far")
        stadium_x = near_center.x - scene.near_stadium_half_length_px
        stadium_y = near_center.y - scene.near_radius_px
        stadium_w = 2.0 * scene.near_stadium_half_length_px
        stadium_h = 2.0 * scene.near_radius_px
        gray = scene.gray_color
        parts.append(
            f'{indent}<rect x="{stadium_x:.3f}" y="{stadium_y:.3f}" '
            f'width="{stadium_w:.3f}" height="{stadium_h:.3f}" '
            f'rx="{scene.near_radius_px:.3f}" ry="{scene.near_radius_px:.3f}" '
            f'fill="rgb({gray[0]},{gray[1]},{gray[2]})"/>\n'
        )
        parts.append(
            f'{indent}<circle cx="{far_center.x:.3f}" cy="{far_center.y:.3f}" '
            f'r="{scene.far_disc_radius_px:.3f}" fill="#fff"/>\n'
        )
    else:
        raise ValueError(f"unknown half-occlusion variant {scene.variant!r}")

    ring_center = half_occlusion_ring_center(scene, eye)
    for radius in scene.ring_radii_px:
        parts.append(
            f'{indent}<circle cx="{ring_center.x:.3f}" cy="{ring_center.y:.3f}" '
            f'r="{radius:.3f}" fill="none" stroke="#000" stroke-width="{scene.stroke_px:.3f}"/>\n'
        )
    return "".join(parts)


def write_half_occlusion_scene_svg(path: str, scene: HalfOcclusionScene, eye: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(scene.width, scene.height))
        f.write(_svg_defs())
        f.write(_half_occlusion_svg_body(scene, eye))
        f.write("</svg>\n")


def _wallpaper_svg_body(scene: WallpaperScene, eye: str, *, indent: str = "  ") -> str:
    w, h = scene.width, scene.height
    surround = f"rgb({scene.surround_gray},{scene.surround_gray},{scene.surround_gray})"
    window_left, window_right = _wallpaper_window_bounds(scene, eye)
    stripe_bounds = _wallpaper_stripe_bounds_eye(scene, eye)

    parts = [f'{indent}<rect x="0" y="0" width="{w}" height="{h}" fill="{surround}"/>\n']
    for center in _wallpaper_dot_centers_eye(scene, eye):
        parts.append(
            f'{indent}<circle cx="{center.x:.3f}" cy="{center.y:.3f}" r="{scene.dot_radius_px:.3f}" fill="#000"/>\n'
        )
    for i in range(scene.stripe_count):
        stripe_left = max(stripe_bounds[i], window_left)
        stripe_right = min(stripe_bounds[i + 1], window_right)
        if stripe_left >= stripe_right:
            continue
        gray = scene.stripe_light_gray if i % 2 == 0 else scene.stripe_dark_gray
        parts.append(
            f'{indent}<rect x="{stripe_left:.3f}" y="{scene.patch_top_px:.3f}" '
            f'width="{(stripe_right - stripe_left):.3f}" '
            f'height="{(scene.patch_bottom_px - scene.patch_top_px):.3f}" '
            f'fill="rgb({gray},{gray},{gray})"/>\n'
        )
    return "".join(parts)


def write_wallpaper_scene_svg(path: str, scene: WallpaperScene, eye: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(scene.width, scene.height))
        f.write(_svg_defs())
        f.write(_wallpaper_svg_body(scene, eye))
        f.write("</svg>\n")


def write_scene_svg(
    path: str,
    rig: CameraRig,
    scene: object,
    eye: str,
    *,
    show_plane_outline: bool = False,
    show_boundary_preview: bool = False,
) -> None:
    if isinstance(scene, HalfOcclusionScene):
        write_half_occlusion_scene_svg(path, scene, eye)
        return
    if isinstance(scene, WallpaperScene):
        write_wallpaper_scene_svg(path, scene, eye)
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(scene.width, scene.height))
        f.write(_svg_defs())
        f.write(_scene_svg_body(
            rig,
            scene,
            eye,
            show_plane_outline=show_plane_outline,
            show_boundary_preview=show_boundary_preview,
        ))
        f.write("</svg>\n")


def write_half_occlusion_pair_svg(
    path: str,
    scene: HalfOcclusionScene,
    *,
    crossed: bool,
    gap: int,
) -> None:
    w, h = scene.width, scene.height
    total_w = 2 * w + gap
    left_column_eye = "right" if crossed else "left"
    right_column_eye = "left" if crossed else "right"
    note = (
        "Crossed fusion: left column is the right-eye image; "
        "right column is the left-eye image."
        if crossed else
        "Parallel order: left column is the left-eye image; right column is the right-eye image."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(total_w, h))
        f.write(f"  <!-- {note} -->\n")
        f.write(_svg_defs())
        f.write('  <rect x="0" y="0" width="100%" height="100%" fill="#fff"/>\n')
        f.write('  <g transform="translate(0,0)">\n')
        f.write(_half_occlusion_svg_body(scene, left_column_eye, indent="    "))
        f.write("  </g>\n")
        f.write(f'  <g transform="translate({w + gap},0)">\n')
        f.write(_half_occlusion_svg_body(scene, right_column_eye, indent="    "))
        f.write("  </g>\n")
        f.write("</svg>\n")


def write_wallpaper_pair_svg(
    path: str,
    scene: WallpaperScene,
    *,
    crossed: bool,
    gap: int,
) -> None:
    w, h = scene.width, scene.height
    total_w = 2 * w + gap
    left_column_eye = "right" if crossed else "left"
    right_column_eye = "left" if crossed else "right"
    note = (
        "Crossed fusion: left column is the right-eye image; "
        "right column is the left-eye image."
        if crossed else
        "Parallel order: left column is the left-eye image; right column is the right-eye image."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(total_w, h))
        f.write(f"  <!-- {note} -->\n")
        f.write(_svg_defs())
        f.write('  <rect x="0" y="0" width="100%" height="100%" fill="#fff"/>\n')
        f.write('  <g transform="translate(0,0)">\n')
        f.write(_wallpaper_svg_body(scene, left_column_eye, indent="    "))
        f.write("  </g>\n")
        f.write(f'  <g transform="translate({w + gap},0)">\n')
        f.write(_wallpaper_svg_body(scene, right_column_eye, indent="    "))
        f.write("  </g>\n")
        f.write("</svg>\n")


def write_pair_svg(
    path: str,
    rig: CameraRig,
    scene: object,
    *,
    crossed: bool,
    gap: int,
    show_plane_outline: bool = False,
) -> None:
    if isinstance(scene, HalfOcclusionScene):
        write_half_occlusion_pair_svg(path, scene, crossed=crossed, gap=gap)
        return
    if isinstance(scene, WallpaperScene):
        write_wallpaper_pair_svg(path, scene, crossed=crossed, gap=gap)
        return

    w, h = scene.width, scene.height
    total_w = 2 * w + gap
    left_column_eye = "right" if crossed else "left"
    right_column_eye = "left" if crossed else "right"
    note = (
        "Crossed fusion: left column is the right-eye image; "
        "right column is the left-eye image."
        if crossed else
        "Parallel order: left column is the left-eye image; right column is the right-eye image."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(total_w, h))
        f.write(f"  <!-- {note} -->\n")
        f.write(_svg_defs())
        f.write('  <rect x="0" y="0" width="100%" height="100%" fill="#fff"/>\n')
        f.write('  <g transform="translate(0,0)">\n')
        f.write(_scene_svg_body(
            rig,
            scene,
            left_column_eye,
            show_plane_outline=show_plane_outline,
            indent="    ",
        ))
        f.write("  </g>\n")
        f.write(f'  <g transform="translate({w + gap},0)">\n')
        f.write(_scene_svg_body(
            rig,
            scene,
            right_column_eye,
            show_plane_outline=show_plane_outline,
            indent="    ",
        ))
        f.write("  </g>\n")
        f.write("</svg>\n")


def build_slanted_plane(
    rig: CameraRig,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    ref_z: float,
    boundary_y: float,
    image_width: int,
) -> SlantedPlane:
    """Build the plane N.X = k for the Gillam 2002 planar "forest" region.

    Starts from a fronto-parallel normal facing the camera, tilts it about
    the horizontal (x) axis by tilt_x_deg (positive recedes upward, away from
    the camera) and then about the vertical (y) axis by tilt_y_deg (positive
    recedes to the right). ref_z is the depth at whichever point along the
    boundary row (the near edge, abutting the dotted plane) sits nearest the
    camera -- with tilt_y_deg == 0 that is the entire boundary row.
    """
    ax = math.radians(-tilt_x_deg)
    by = math.radians(tilt_y_deg)

    # Fronto-parallel normal, facing the camera.
    nx0, ny0, nz0 = 0.0, 0.0, -1.0
    # Rotate about the x-axis.
    nx1 = nx0
    ny1 = ny0 * math.cos(ax) - nz0 * math.sin(ax)
    nz1 = ny0 * math.sin(ax) + nz0 * math.cos(ax)
    # Rotate about the y-axis.
    nx = nx1 * math.cos(by) + nz1 * math.sin(by)
    ny = ny1
    nz = -nx1 * math.sin(by) + nz1 * math.cos(by)

    f = rig.focal_px
    cx, cy = rig.cx, rig.cy

    def denom_at(x: float, y: float) -> float:
        return nx * (x - cx) / f + ny * (y - cy) / f + nz

    denom_left = denom_at(0.0, boundary_y)
    denom_right = denom_at(image_width - 1.0, boundary_y)
    x_near = 0.0 if abs(denom_left) >= abs(denom_right) else (image_width - 1.0)
    denom_near = denom_at(x_near, boundary_y)
    if abs(denom_near) < 1e-9:
        raise ValueError("slanted plane is degenerate (near-edge-on) for the given tilt angles")
    k = ref_z * denom_near

    return SlantedPlane(nx=nx, ny=ny, nz=nz, k=k, tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg, ref_z=ref_z)


def slanted_plane_depth(plane: SlantedPlane, rig: CameraRig, x: float, y: float, eye: str) -> float:
    if eye == "cyclopean":
        camera_x = 0.0
    elif eye == "left":
        camera_x = -0.5 * rig.baseline
    elif eye == "right":
        camera_x = 0.5 * rig.baseline
    else:
        raise ValueError(f"unknown eye {eye!r}")

    f = rig.focal_px
    cx, cy = rig.cx, rig.cy
    denom = plane.nx * (x - cx) / f + plane.ny * (y - cy) / f + plane.nz
    if abs(denom) < 1e-9:
        raise ValueError("slanted plane ray is degenerate (parallel to the plane) at the requested pixel")
    numer = plane.k - plane.nx * camera_x
    return numer / denom


def validate_slanted_plane_region(
    plane: SlantedPlane,
    rig: CameraRig,
    plane_z: float,
    *,
    image_width: int,
    top_y: float,
    boundary_y: float,
) -> None:
    for eye in ("cyclopean", "left", "right"):
        for x in (0.0, image_width - 1.0):
            for y in (top_y, boundary_y):
                z = slanted_plane_depth(plane, rig, x, y, eye)
                if not (z > plane_z):
                    raise ValueError(
                        "the slanted forest plane must stay strictly farther than --plane-z "
                        f"everywhere in the forest region; got z={z:.4f} at eye={eye}, x={x:.1f}, "
                        f"y={y:.1f}. Reduce --plane-slant-tilt-x-deg/--plane-slant-tilt-y-deg or "
                        "increase --plane-slant-ref-z."
                    )


def make_figure9a_scene(
    rig: CameraRig,
    *,
    mode: str = "planar",
    plane_z: float,
    line_zs: Sequence[float],
    plane_top_frac: float,
    dot_radius_frac: float,
    line_stroke_frac: float,
    plane_slant_tilt_x_deg: float = 20.0,
    plane_slant_tilt_y_deg: float = 0.0,
    plane_slant_ref_z: float = 1.95,
) -> Scene:
    if mode not in ("forest", "planar"):
        raise ValueError(f"gillam lines mode must be 'forest' or 'planar', got {mode!r}")
    if mode == "forest":
        if any(z <= plane_z for z in line_zs):
            raise ValueError("every line depth must be farther than the plane depth")
        if len(line_zs) != 7:
            raise ValueError("Figure 9A uses exactly seven line segments")

    w, h = rig.width, rig.height
    scale = min(w, h)
    plane_top_y = plane_top_frac * h
    dot_radius_px = max(1.0, dot_radius_frac * scale)
    line_stroke_px = max(1.0, line_stroke_frac * scale)

    # Irregular dot texture copied in spirit from Figure 9A: a white plane whose
    # top edge is suggested by the line terminators and by the absence of dots
    # above the boundary.
    dot_norm = [
        (0.285, 0.445), (0.350, 0.430), (0.485, 0.430), (0.600, 0.420),
        (0.675, 0.410), (0.760, 0.430),
        (0.315, 0.500), (0.405, 0.490), (0.485, 0.475), (0.560, 0.485),
        (0.635, 0.490), (0.725, 0.485), (0.815, 0.490),
        (0.260, 0.565), (0.335, 0.565), (0.455, 0.550), (0.545, 0.550),
        (0.625, 0.575), (0.710, 0.555), (0.790, 0.560),
        (0.295, 0.640), (0.385, 0.690), (0.470, 0.625), (0.525, 0.615),
        (0.615, 0.685), (0.690, 0.640),
    ]
    dots = [
        Dot(
            center=rig.image_to_world(x_frac * w, y_frac * h, plane_z),
            radius_world=dot_radius_px * plane_z / rig.focal_px,
        )
        for x_frac, y_frac in dot_norm
    ]

    boundary = plane_top_y
    line_norm = [
        ((0.305, boundary / h), (0.385, 0.095)),
        ((0.395, boundary / h), (0.425, 0.165)),
        ((0.515, boundary / h), (0.475, 0.100)),
        ((0.610, boundary / h), (0.550, 0.055)),
        ((0.680, boundary / h), (0.700, 0.155)),
        ((0.745, boundary / h), (0.805, 0.165)),
        ((0.830, boundary / h), (0.805, 0.185)),
    ]

    lines_plane: Optional[SlantedPlane] = None
    if mode == "planar":
        lines_plane = build_slanted_plane(
            rig,
            tilt_x_deg=plane_slant_tilt_x_deg,
            tilt_y_deg=plane_slant_tilt_y_deg,
            ref_z=plane_slant_ref_z,
            boundary_y=boundary,
            image_width=w,
        )
        validate_slanted_plane_region(
            lines_plane, rig, plane_z, image_width=w, top_y=0.0, boundary_y=boundary,
        )

    lines = []
    depth_source = line_zs if mode == "forest" else [None] * len(line_norm)
    for i, (((x0f, y0f), (x1f, y1f)), z) in enumerate(zip(line_norm, depth_source), start=2):
        x0, y0 = x0f * w, y0f * h
        x1, y1 = x1f * w, y1f * h
        if mode == "planar":
            z0 = slanted_plane_depth(lines_plane, rig, x0, y0, "cyclopean")
            z1 = slanted_plane_depth(lines_plane, rig, x1, y1, "cyclopean")
        else:
            z0 = z1 = z
        if z0 <= plane_z or z1 <= plane_z:
            raise ValueError("every line depth must be farther than the plane depth")
        p0 = rig.image_to_world(x0, y0, z0)
        p1 = rig.image_to_world(x1, y1, z1)
        lines.append(LineSegment3D(
            p0=p0,
            p1=p1,
            z=0.5 * (z0 + z1),
            stroke_px=line_stroke_px,
            label=i,
        ))

    return Scene(
        width=w,
        height=h,
        plane_z=plane_z,
        plane_top_y=plane_top_y,
        dot_radius_px=dot_radius_px,
        line_stroke_px=line_stroke_px,
        dots=dots,
        lines=lines,
        mode=mode,
        lines_plane=lines_plane,
    )


def make_belhumeur1996_figure2_scene(
    rig: CameraRig,
    *,
    variant: str,
    percept_label: str,
    d1_px: float,
    disc_diameter_px: float,
    crescent_fraction: float,
    gray_value: int,
) -> HalfOcclusionScene:
    disc_disparity = d1_px
    if disc_diameter_px <= 0.0:
        raise ValueError("Belhumeur Figure 2 disc diameter D_img must be positive")
    if crescent_fraction <= 0.0:
        raise ValueError("Belhumeur Figure 2 relative crescent width w must be positive")

    crescent_width_px = crescent_fraction * disc_diameter_px
    stadium_disparity = disc_disparity - crescent_width_px
    background_disparity = stadium_disparity - crescent_width_px
    if disc_disparity <= 0.0:
        raise ValueError("Belhumeur Figure 2 foreground disc disparity d1 must be positive")
    if stadium_disparity <= 0.0:
        raise ValueError("Belhumeur Figure 2 stadium/hole disparity d2 must be positive")
    if background_disparity <= 0.0:
        raise ValueError(
            "Belhumeur Figure 2 Section 3 scheme requires d1 > 2*w*D_img so that d3 = d1 - 2*w*D_img stays positive"
        )

    w, h = rig.width, rig.height
    center = Point2(0.5 * (w - 1), 0.5 * (h - 1))
    near_radius = 0.5 * disc_diameter_px
    stroke_px = max(1.0, 0.0048 * min(w, h))
    gray = max(0, min(255, gray_value))
    disc_z = rig.focal_px * rig.baseline / disc_disparity
    stadium_z = rig.focal_px * rig.baseline / stadium_disparity
    background_z = rig.focal_px * rig.baseline / background_disparity
    stadium_half_length = near_radius + 0.5 * crescent_width_px

    # Section 3 uses W = w * D_img, so the disparity ladder is:
    #   d2 = d1 - W
    #   d3 = d1 - 2W
    # while the projected disc/hole diameter stays equal to D_img.
    if variant == "pimple":
        near_center_offset = 0.0
        far_center_offset = 0.0
        near_stadium_half_length = 0.0
        far_disc_radius = 0.0
        far_stadium_half_length = stadium_half_length
        ring_host_radius = near_radius
        near_z = disc_z
        far_z = stadium_z
        near_disparity = disc_disparity
        far_disparity = stadium_disparity
    elif variant == "dimple":
        near_center_offset = 0.0
        far_center_offset = 0.0
        far_disc_radius = near_radius
        near_stadium_half_length = stadium_half_length
        far_stadium_half_length = stadium_half_length
        ring_host_radius = far_disc_radius
        near_z = stadium_z
        far_z = background_z
        near_disparity = stadium_disparity
        far_disparity = background_disparity
    else:
        raise ValueError(f"unknown Belhumeur Figure 2 variant {variant!r}")

    ring_radii = tuple(ring_host_radius * frac for frac in (1.0 / 3.0, 2.0 / 3.0, 1.0))

    return HalfOcclusionScene(
        width=w,
        height=h,
        center=center,
        variant=variant,
        percept_label=percept_label,
        disc_diameter_px=disc_diameter_px,
        crescent_fraction=crescent_fraction,
        crescent_width_px=crescent_width_px,
        disc_z=disc_z,
        stadium_z=stadium_z,
        background_z=background_z,
        disc_disparity_px=disc_disparity,
        stadium_disparity_px=stadium_disparity,
        background_disparity_px=background_disparity,
        near_center_offset_px=near_center_offset,
        far_center_offset_px=far_center_offset,
        near_z=near_z,
        far_z=far_z,
        near_disparity_px=near_disparity,
        far_disparity_px=far_disparity,
        near_radius_px=near_radius,
        near_stadium_half_length_px=near_stadium_half_length,
        far_disc_radius_px=far_disc_radius,
        far_stadium_half_length_px=far_stadium_half_length,
        stroke_px=stroke_px,
        ring_radii_px=ring_radii,
        gray_color=(gray, gray, gray),
    )


def _wallpaper_eye_offset(disparity_px: float, eye: str) -> float:
    if eye == "cyclopean":
        return 0.0
    elif eye == "left":
        return 0.5 * disparity_px
    elif eye == "right":
        return -0.5 * disparity_px
    raise ValueError(f"unknown eye {eye!r}")


def _wallpaper_window_bounds(scene: WallpaperScene, eye: str) -> Tuple[float, float]:
    """The nearer surface owns the occluding contour: for 'front' that's the
    wallpaper's own projected edges; for 'behind' it's the surround's aperture
    edges. Returns the visible [left, right) window for this eye in pixels.
    """
    anchor_disparity = scene.wallpaper_disparity_px if scene.percept == "front" else scene.surround_disparity_px
    offset = _wallpaper_eye_offset(anchor_disparity, eye)
    return scene.patch_left_px + offset, scene.patch_right_px + offset


def _wallpaper_stripe_bounds_eye(scene: WallpaperScene, eye: str) -> List[float]:
    """Stripe boundary positions for this eye. The whole stripe pattern is a
    rigid block at the wallpaper's own depth, so every boundary shares the same
    offset regardless of percept -- only the visible window (see
    _wallpaper_window_bounds) differs between 'front' and 'behind'.
    """
    offset = _wallpaper_eye_offset(scene.wallpaper_disparity_px, eye)
    return [
        scene.patch_left_px + i * scene.stripe_width_px + offset
        for i in range(scene.stripe_count + 1)
    ]


def _wallpaper_dot_centers_eye(scene: WallpaperScene, eye: str) -> List[Point2]:
    # Dots texture the one unambiguous surround surface, so they always shift
    # by the surround's own disparity, regardless of percept.
    offset = _wallpaper_eye_offset(scene.surround_disparity_px, eye)
    return [Point2(p.x + offset, p.y) for p in scene.dots]


def _sample_wallpaper_dots(
    rig: CameraRig,
    *,
    patch_left_px: float,
    patch_right_px: float,
    patch_top_px: float,
    patch_bottom_px: float,
    max_wallpaper_disparity_px: float,
    dot_count: int,
    dot_radius_px: float,
    dot_seed: int,
) -> Tuple[Point2, ...]:
    """Sample sparse dot centers (in nominal/cyclopean pixel space) over the
    whole frame, rejecting any that would fall on or near the wallpaper patch
    in ANY eye view under EITHER percept. Using max_wallpaper_disparity_px
    (the larger of the front/behind disparities) for the exclusion margin, and
    the same seed for both percepts, means front and behind scenes built with
    matching parameters get an identical dot layout, as befits a texture that
    belongs to the one unambiguous surround surface.
    """
    w, h = rig.width, rig.height
    half_shift = 0.5 * max_wallpaper_disparity_px
    excl_left = patch_left_px - half_shift - dot_radius_px
    excl_right = patch_right_px + half_shift + dot_radius_px
    excl_top = patch_top_px - dot_radius_px
    excl_bottom = patch_bottom_px + dot_radius_px

    rng = random.Random(dot_seed)
    dots: List[Point2] = []
    attempts = 0
    max_attempts = max(1000, dot_count * 200)
    while len(dots) < dot_count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(0.0, w)
        y = rng.uniform(0.0, h)
        if excl_left <= x <= excl_right and excl_top <= y <= excl_bottom:
            continue
        dots.append(Point2(x, y))
    return tuple(dots)


def make_wallpaper_scene(
    rig: CameraRig,
    *,
    percept: str,
    surround_z: float,
    surround_gray: int,
    stripe_light_gray: int,
    stripe_dark_gray: int,
    stripe_width_frac: float,
    stripe_count: int,
    patch_height_frac: float,
    luminance_preset: str,
    dot_count: int,
    dot_radius_frac: float,
    dot_seed: int,
) -> WallpaperScene:
    if percept not in ("front", "behind"):
        raise ValueError(f"wallpaper percept must be 'front' or 'behind', got {percept!r}")
    if stripe_count < 2:
        raise ValueError("--wallpaper-stripe-count must be at least 2")

    w, h = rig.width, rig.height
    scale = min(w, h)
    stripe_width_px = max(1.0, stripe_width_frac * scale)
    # Anderson & Nakayama (1994): "shifting the wallpaper pattern in one of the
    # two eyes by one stripe width, or one half cycle." This exact half-cycle
    # shift is what makes the front and behind interpretations equally valid.
    shift_px = stripe_width_px

    surround_disparity_px = rig.disparity(surround_z)
    if percept == "front":
        wallpaper_disparity_px = surround_disparity_px + shift_px
    else:
        wallpaper_disparity_px = surround_disparity_px - shift_px
        if wallpaper_disparity_px <= 0.0:
            raise ValueError(
                "the 'behind' wallpaper percept requires the surround's disparity to exceed "
                "the stripe-width shift; increase --wallpaper-surround-z (moves the surround "
                "closer) or decrease --wallpaper-stripe-width-frac"
            )
    wallpaper_z = rig.focal_px * rig.baseline / wallpaper_disparity_px

    total_width_px = stripe_count * stripe_width_px
    patch_height_px = patch_height_frac * h
    cx, cy = rig.cx, rig.cy
    patch_left_px = cx - 0.5 * total_width_px
    patch_right_px = cx + 0.5 * total_width_px
    patch_top_px = cy - 0.5 * patch_height_px
    patch_bottom_px = cy + 0.5 * patch_height_px

    dot_radius_px = max(1.0, dot_radius_frac * scale)
    # Always exclude using the "front" (surround + shift) disparity, the larger
    # of the two candidates, regardless of this instance's own percept -- see
    # _sample_wallpaper_dots.
    dots = _sample_wallpaper_dots(
        rig,
        patch_left_px=patch_left_px,
        patch_right_px=patch_right_px,
        patch_top_px=patch_top_px,
        patch_bottom_px=patch_bottom_px,
        max_wallpaper_disparity_px=surround_disparity_px + shift_px,
        dot_count=dot_count,
        dot_radius_px=dot_radius_px,
        dot_seed=dot_seed,
    )

    return WallpaperScene(
        width=w,
        height=h,
        percept=percept,
        surround_gray=surround_gray,
        stripe_light_gray=stripe_light_gray,
        stripe_dark_gray=stripe_dark_gray,
        stripe_width_px=stripe_width_px,
        stripe_count=stripe_count,
        patch_left_px=patch_left_px,
        patch_right_px=patch_right_px,
        patch_top_px=patch_top_px,
        patch_bottom_px=patch_bottom_px,
        surround_z=surround_z,
        surround_disparity_px=surround_disparity_px,
        wallpaper_z=wallpaper_z,
        wallpaper_disparity_px=wallpaper_disparity_px,
        shift_px=shift_px,
        luminance_preset=luminance_preset,
        dots=dots,
        dot_radius_px=dot_radius_px,
        dot_seed=dot_seed,
    )


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / length_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def _rgb_canvas(width: int, height: int, color: Tuple[int, int, int] = (255, 255, 255)) -> array:
    return array("B", color) * (width * height)


def _paint_pixel_rgb(
    pixels: array,
    width: int,
    height: int,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    *,
    alpha: float = 1.0,
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = 3 * (y * width + x)
    if alpha >= 1.0:
        pixels[idx] = color[0]
        pixels[idx + 1] = color[1]
        pixels[idx + 2] = color[2]
        return
    inv = 1.0 - alpha
    pixels[idx] = int(round(alpha * color[0] + inv * pixels[idx]))
    pixels[idx + 1] = int(round(alpha * color[1] + inv * pixels[idx + 1]))
    pixels[idx + 2] = int(round(alpha * color[2] + inv * pixels[idx + 2]))


def _draw_filled_circle_rgb(
    pixels: array,
    width: int,
    height: int,
    center: Point2,
    radius: float,
    color: Tuple[int, int, int],
) -> None:
    xmin = max(0, int(math.floor(center.x - radius)))
    xmax = min(width - 1, int(math.ceil(center.x + radius)))
    ymin = max(0, int(math.floor(center.y - radius)))
    ymax = min(height - 1, int(math.ceil(center.y + radius)))
    r2 = radius * radius

    for y in range(ymin, ymax + 1):
        py = y + 0.5
        dy = py - center.y
        for x in range(xmin, xmax + 1):
            px = x + 0.5
            dx = px - center.x
            if dx * dx + dy * dy <= r2:
                _paint_pixel_rgb(pixels, width, height, x, y, color)


def _draw_line_rgb(
    pixels: array,
    width: int,
    height: int,
    p0: Point2,
    p1: Point2,
    stroke_px: float,
    color: Tuple[int, int, int],
    *,
    alpha: float = 1.0,
) -> None:
    half = max(0.5, 0.5 * stroke_px)
    xmin = max(0, int(math.floor(min(p0.x, p1.x) - half - 1)))
    xmax = min(width - 1, int(math.ceil(max(p0.x, p1.x) + half + 1)))
    ymin = max(0, int(math.floor(min(p0.y, p1.y) - half - 1)))
    ymax = min(height - 1, int(math.ceil(max(p0.y, p1.y) + half + 1)))

    for y in range(ymin, ymax + 1):
        py = y + 0.5
        for x in range(xmin, xmax + 1):
            px = x + 0.5
            if _point_segment_distance(px, py, p0.x, p0.y, p1.x, p1.y) <= half:
                _paint_pixel_rgb(pixels, width, height, x, y, color, alpha=alpha)


def _overlay_rect_rgb(
    pixels: array,
    width: int,
    height: int,
    x0: int,
    y0: int,
    rect_w: int,
    rect_h: int,
    color: Tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    xmax = min(width, x0 + rect_w)
    ymax = min(height, y0 + rect_h)
    for y in range(max(0, y0), ymax):
        for x in range(max(0, x0), xmax):
            _paint_pixel_rgb(pixels, width, height, x, y, color, alpha=alpha)


def _point_in_circle(px: float, py: float, center: Point2, radius: float) -> bool:
    dx = px - center.x
    dy = py - center.y
    return dx * dx + dy * dy <= radius * radius


def _point_in_stadium(px: float, py: float, center: Point2, half_length: float, radius: float) -> bool:
    if half_length < radius:
        half_length = radius
    dx = abs(px - center.x)
    dy = abs(py - center.y)
    if dy > radius or dx > half_length:
        return False
    inner_half = half_length - radius
    if dx <= inner_half:
        return True
    cap_dx = dx - inner_half
    return cap_dx * cap_dx + dy * dy <= radius * radius


def rasterize_half_occlusion_scene_rgb(scene: HalfOcclusionScene, eye: str) -> array:
    w, h = scene.width, scene.height
    pixels = _rgb_canvas(w, h)
    near_center = half_occlusion_projected_center(scene, eye, "near")
    far_center = half_occlusion_projected_center(scene, eye, "far")

    if scene.variant == "pimple":
        _draw_filled_stadium_rgb(
            pixels,
            w,
            h,
            far_center,
            scene.far_stadium_half_length_px,
            scene.near_radius_px,
            scene.gray_color,
        )
        _draw_filled_circle_rgb(pixels, w, h, near_center, scene.near_radius_px, (255, 255, 255))
    elif scene.variant == "dimple":
        _draw_filled_stadium_rgb(
            pixels,
            w,
            h,
            near_center,
            scene.near_stadium_half_length_px,
            scene.near_radius_px,
            scene.gray_color,
        )
        _draw_filled_circle_rgb(pixels, w, h, far_center, scene.far_disc_radius_px, (255, 255, 255))
    else:
        raise ValueError(f"unknown half-occlusion variant {scene.variant!r}")

    ring_center = half_occlusion_ring_center(scene, eye)
    for radius in scene.ring_radii_px:
        _draw_circle_outline_rgb(pixels, w, h, ring_center, radius, scene.stroke_px, (0, 0, 0))

    return pixels


def _draw_filled_stadium_rgb(
    pixels: array,
    width: int,
    height: int,
    center: Point2,
    half_length: float,
    radius: float,
    color: Tuple[int, int, int],
) -> None:
    xmin = max(0, int(math.floor(center.x - half_length)))
    xmax = min(width - 1, int(math.ceil(center.x + half_length)))
    ymin = max(0, int(math.floor(center.y - radius)))
    ymax = min(height - 1, int(math.ceil(center.y + radius)))

    for y in range(ymin, ymax + 1):
        py = y + 0.5
        for x in range(xmin, xmax + 1):
            px = x + 0.5
            if _point_in_stadium(px, py, center, half_length, radius):
                _paint_pixel_rgb(pixels, width, height, x, y, color)


def _draw_circle_outline_rgb(
    pixels: array,
    width: int,
    height: int,
    center: Point2,
    radius: float,
    stroke_px: float,
    color: Tuple[int, int, int],
) -> None:
    half = max(0.5, 0.5 * stroke_px)
    outer = radius + half
    inner = max(0.0, radius - half)
    outer2 = outer * outer
    inner2 = inner * inner
    xmin = max(0, int(math.floor(center.x - outer)))
    xmax = min(width - 1, int(math.ceil(center.x + outer)))
    ymin = max(0, int(math.floor(center.y - outer)))
    ymax = min(height - 1, int(math.ceil(center.y + outer)))

    for y in range(ymin, ymax + 1):
        py = y + 0.5
        dy = py - center.y
        for x in range(xmin, xmax + 1):
            px = x + 0.5
            dx = px - center.x
            d2 = dx * dx + dy * dy
            if inner2 <= d2 <= outer2:
                _paint_pixel_rgb(pixels, width, height, x, y, color)


def _fill_rect_rgb(
    pixels: array,
    width: int,
    height: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: Tuple[int, int, int],
) -> None:
    xmin = max(0, int(math.floor(x0)))
    xmax = min(width, int(math.ceil(x1)))
    ymin = max(0, int(math.floor(y0)))
    ymax = min(height, int(math.ceil(y1)))
    if xmin >= xmax or ymin >= ymax:
        return
    row_span = array("B", color) * (xmax - xmin)
    for y in range(ymin, ymax):
        start = 3 * (y * width + xmin)
        pixels[start:start + 3 * (xmax - xmin)] = row_span


def rasterize_wallpaper_scene_rgb(scene: WallpaperScene, eye: str) -> array:
    w, h = scene.width, scene.height
    pixels = _rgb_canvas(w, h, (scene.surround_gray,) * 3)

    for center in _wallpaper_dot_centers_eye(scene, eye):
        _draw_filled_circle_rgb(pixels, w, h, center, scene.dot_radius_px, (0, 0, 0))

    window_left, window_right = _wallpaper_window_bounds(scene, eye)
    stripe_bounds = _wallpaper_stripe_bounds_eye(scene, eye)

    for i in range(scene.stripe_count):
        stripe_left = max(stripe_bounds[i], window_left)
        stripe_right = min(stripe_bounds[i + 1], window_right)
        if stripe_left >= stripe_right:
            continue
        gray = scene.stripe_light_gray if i % 2 == 0 else scene.stripe_dark_gray
        _fill_rect_rgb(
            pixels, w, h, stripe_left, stripe_right, scene.patch_top_px, scene.patch_bottom_px, (gray,) * 3,
        )

    return pixels


def rasterize_wallpaper_maps(
    scene: WallpaperScene,
    eye: str,
    *,
    include_segmentation: bool,
) -> Tuple[array, array]:
    w, h = scene.width, scene.height
    n = w * h
    disp = array("f", [scene.surround_disparity_px]) * n
    seg = array("B", [0]) * n
    if include_segmentation:
        seg = array("B", [WALLPAPER_SURROUND_LABEL]) * n

    window_left, window_right = _wallpaper_window_bounds(scene, eye)
    xmin = max(0, int(math.floor(window_left)))
    xmax = min(w, int(math.ceil(window_right)))
    ymin = max(0, int(math.floor(scene.patch_top_px)))
    ymax = min(h, int(math.ceil(scene.patch_bottom_px)))
    if xmin < xmax and ymin < ymax:
        span = xmax - xmin
        disp_span = array("f", [scene.wallpaper_disparity_px]) * span
        seg_span = array("B", [WALLPAPER_STRIPE_LABEL]) * span
        for y in range(ymin, ymax):
            start = y * w + xmin
            disp[start:start + span] = disp_span
            if include_segmentation:
                seg[start:start + span] = seg_span

    return disp, seg


def rasterize_half_occlusion_maps(
    scene: HalfOcclusionScene,
    eye: str,
    *,
    include_segmentation: bool,
) -> Tuple[array, array]:
    w, h = scene.width, scene.height
    disp = array("f", [0.0]) * (w * h)
    seg = array("B", [0]) * (w * h)
    near_center = half_occlusion_projected_center(scene, eye, "near")
    far_center = half_occlusion_projected_center(scene, eye, "far")

    for y in range(h):
        py = y + 0.5
        row = y * w
        for x in range(w):
            px = x + 0.5
            idx = row + x
            if scene.variant == "pimple":
                if _point_in_circle(px, py, near_center, scene.near_radius_px):
                    disp[idx] = scene.near_disparity_px
                    if include_segmentation:
                        seg[idx] = 1
                elif _point_in_stadium(px, py, far_center, scene.far_stadium_half_length_px, scene.near_radius_px):
                    disp[idx] = scene.far_disparity_px
                    if include_segmentation:
                        seg[idx] = 2
            elif scene.variant == "dimple":
                if _point_in_stadium(px, py, near_center, scene.near_stadium_half_length_px, scene.near_radius_px):
                    disp[idx] = scene.far_disparity_px
                    if include_segmentation:
                        if _point_in_circle(px, py, far_center, scene.far_disc_radius_px):
                            seg[idx] = 3
                        else:
                            seg[idx] = 2
                else:
                    disp[idx] = scene.near_disparity_px
                    if include_segmentation:
                        seg[idx] = 1
            else:
                raise ValueError(f"unknown half-occlusion variant {scene.variant!r}")

    return disp, seg


def rasterize_half_occlusion_boundary_preview_rgb(
    scene: HalfOcclusionScene,
    boundary: Sequence[int],
) -> array:
    w, h = scene.width, scene.height
    pixels = rasterize_half_occlusion_scene_rgb(scene, "cyclopean")
    stride = max(1, min(w, h) // 256)
    for y in range(0, h, stride):
        row = y * w
        for x in range(0, w, stride):
            if boundary[row + x]:
                _overlay_rect_rgb(pixels, w, h, x, y, stride, stride, (214, 39, 40), alpha=0.45)
    return pixels


def rasterize_scene_rgb(
    rig: CameraRig,
    scene: object,
    eye: str,
    *,
    show_plane_outline: bool = False,
    show_boundary_preview: bool = False,
) -> array:
    if isinstance(scene, HalfOcclusionScene):
        return rasterize_half_occlusion_scene_rgb(scene, eye)
    if isinstance(scene, WallpaperScene):
        return rasterize_wallpaper_scene_rgb(scene, eye)

    w, h = scene.width, scene.height
    pixels = _rgb_canvas(w, h)

    if show_plane_outline:
        x_margin = 0.35 * w
        y_bottom = h + 0.35 * h
        corners_cyc = [
            (-x_margin, scene.plane_top_y),
            (w + x_margin, scene.plane_top_y),
            (w + x_margin, y_bottom),
            (-x_margin, y_bottom),
        ]
        corners = [
            rig.project(rig.image_to_world(x, y, scene.plane_z), eye)
            for x, y in corners_cyc
        ]
        for idx, p0 in enumerate(corners):
            _draw_line_rgb(pixels, w, h, p0, corners[(idx + 1) % len(corners)], 1.0, (191, 191, 191))

    for dot in scene.dots:
        p = rig.project(dot.center, eye)
        r = rig.focal_px * dot.radius_world / dot.center.z
        _draw_filled_circle_rgb(pixels, w, h, p, r, (0, 0, 0))

    for seg in scene.lines:
        p0 = rig.project(seg.p0, eye)
        p1 = rig.project(seg.p1, eye)
        _draw_line_rgb(pixels, w, h, p0, p1, seg.stroke_px, (0, 0, 0))

    if show_boundary_preview:
        x_margin = 0.35 * w
        left = rig.project(rig.image_to_world(-x_margin, scene.plane_top_y, scene.plane_z), eye)
        right = rig.project(rig.image_to_world(w + x_margin, scene.plane_top_y, scene.plane_z), eye)
        _draw_line_rgb(pixels, w, h, left, right, 1.5, (214, 39, 40), alpha=0.70)

    return pixels


def rasterize_pair_rgb(
    rig: CameraRig,
    scene: object,
    *,
    crossed: bool,
    gap: int,
    show_plane_outline: bool = False,
) -> Tuple[array, int, int]:
    w, h = scene.width, scene.height
    total_w = 2 * w + gap
    left_column_eye = "right" if crossed else "left"
    right_column_eye = "left" if crossed else "right"
    left_pixels = rasterize_scene_rgb(
        rig,
        scene,
        left_column_eye,
        show_plane_outline=show_plane_outline,
    )
    right_pixels = rasterize_scene_rgb(
        rig,
        scene,
        right_column_eye,
        show_plane_outline=show_plane_outline,
    )
    out = _rgb_canvas(total_w, h)

    for y in range(h):
        src_row = 3 * y * w
        left_dst = 3 * y * total_w
        right_dst = 3 * (y * total_w + w + gap)
        out[left_dst:left_dst + 3 * w] = left_pixels[src_row:src_row + 3 * w]
        out[right_dst:right_dst + 3 * w] = right_pixels[src_row:src_row + 3 * w]

    return out, total_w, h


def rasterize_boundary_preview_rgb(
    rig: CameraRig,
    scene: object,
    boundary: Sequence[int],
) -> array:
    if isinstance(scene, HalfOcclusionScene):
        return rasterize_half_occlusion_boundary_preview_rgb(scene, boundary)

    w, h = scene.width, scene.height
    pixels = rasterize_scene_rgb(rig, scene, "cyclopean")
    stride = max(1, min(w, h) // 256)
    for y in range(0, h, stride):
        row = y * w
        for x in range(0, w, stride):
            if boundary[row + x]:
                _overlay_rect_rgb(pixels, w, h, x, y, stride, stride, (214, 39, 40), alpha=0.45)
    return pixels


def rasterize_view(
    rig: CameraRig,
    scene: object,
    eye: str,
    *,
    include_segmentation: bool,
) -> Tuple[array, array]:
    if isinstance(scene, HalfOcclusionScene):
        return rasterize_half_occlusion_maps(scene, eye, include_segmentation=include_segmentation)
    if isinstance(scene, WallpaperScene):
        return rasterize_wallpaper_maps(scene, eye, include_segmentation=include_segmentation)

    w, h = scene.width, scene.height
    n = w * h
    disp = array("f", [0.0]) * n
    seg = array("B", [0]) * n

    plane_disp = rig.disparity(scene.plane_z)
    for y in range(h):
        if (y + 0.5) < scene.plane_top_y:
            continue
        row = y * w
        for x in range(w):
            idx = row + x
            disp[idx] = plane_disp
            if include_segmentation:
                seg[idx] = 1

    if getattr(scene, "mode", "forest") == "planar" and scene.lines_plane is not None:
        # The forest lines are coplanar on a single slanted surface: fill the
        # whole region with that plane's disparity (and one solid label),
        # rather than only the pixels the individual strokes happen to cover.
        # This matches the percept the paper reports for this configuration
        # -- a solid occluded surface, not a set of disjoint objects.
        plane = scene.lines_plane
        f = rig.focal_px
        cx, cy = rig.cx, rig.cy
        camera_x = {"cyclopean": 0.0, "left": -0.5 * rig.baseline, "right": 0.5 * rig.baseline}[eye]
        numer = plane.k - plane.nx * camera_x
        for y in range(h):
            py = y + 0.5
            if py >= scene.plane_top_y:
                continue
            row = y * w
            ny_term = plane.ny * (py - cy) / f
            for x in range(w):
                px = x + 0.5
                denom = plane.nx * (px - cx) / f + ny_term + plane.nz
                z = numer / denom
                idx = row + x
                disp[idx] = rig.focal_px * rig.baseline / z
                if include_segmentation:
                    seg[idx] = GILLAM_PLANAR_LINES_LABEL
    else:
        for line in scene.lines:
            p0 = rig.project(line.p0, eye)
            p1 = rig.project(line.p1, eye)
            half = 0.5 * line.stroke_px
            xmin = max(0, int(math.floor(min(p0.x, p1.x) - half - 1)))
            xmax = min(w - 1, int(math.ceil(max(p0.x, p1.x) + half + 1)))
            ymin = max(0, int(math.floor(min(p0.y, p1.y) - half - 1)))
            ymax = min(h - 1, int(math.ceil(max(p0.y, p1.y) + half + 1)))
            line_disp = rig.disparity(line.z)

            for y in range(ymin, ymax + 1):
                py = y + 0.5
                if py >= scene.plane_top_y:
                    continue
                row = y * w
                for x in range(xmin, xmax + 1):
                    px = x + 0.5
                    if _point_segment_distance(px, py, p0.x, p0.y, p1.x, p1.y) <= half:
                        idx = row + x
                        disp[idx] = line_disp
                        if include_segmentation:
                            seg[idx] = line.label

    return disp, seg


def segmentation_boundary(seg: Sequence[int], width: int, height: int) -> array:
    boundary = array("B", [0]) * (width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            value = seg[idx]
            if (
                (x > 0 and seg[idx - 1] != value) or
                (x + 1 < width and seg[idx + 1] != value) or
                (y > 0 and seg[idx - width] != value) or
                (y + 1 < height and seg[idx + width] != value)
            ):
                boundary[idx] = 1
    return boundary


def _edt_1d(values: Sequence[float]) -> List[float]:
    n = len(values)
    inf = 1.0e20
    v = [0] * n
    z = [0.0] * (n + 1)
    out = [0.0] * n
    k = 0
    v[0] = 0
    z[0] = -inf
    z[1] = inf

    for q in range(1, n):
        while True:
            denom = 2.0 * (q - v[k])
            if abs(denom) < 1e-12:
                s = inf
            else:
                s = ((values[q] + q * q) - (values[v[k]] + v[k] * v[k])) / denom
            if s > z[k]:
                break
            k -= 1
            if k < 0:
                k = 0
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        out[q] = diff * diff + values[v[k]]
    return out


def distance_transform_from_boundary(boundary: Sequence[int], width: int, height: int) -> array:
    inf = 1.0e12
    temp: List[float] = [0.0] * (width * height)

    for y in range(height):
        row = y * width
        row_values = [0.0 if boundary[row + x] else inf for x in range(width)]
        row_dist = _edt_1d(row_values)
        temp[row:row + width] = row_dist

    udf = array("f", [0.0]) * (width * height)
    for x in range(width):
        col_values = [temp[y * width + x] for y in range(height)]
        col_dist = _edt_1d(col_values)
        for y, d2 in enumerate(col_dist):
            udf[y * width + x] = math.sqrt(d2)
    return udf


def _npy_shape(shape: Sequence[int]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(s) for s in shape) + ")"


def write_npy(path: str, data: Iterable[float], shape: Sequence[int], descr: str, typecode: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = data if isinstance(data, array) and data.typecode == typecode else array(typecode, data)
    if descr.startswith("<") and sys.byteorder != "little":
        arr = array(typecode, arr)
        arr.byteswap()

    header = (
        "{'descr': '" + descr + "', 'fortran_order': False, "
        "'shape': " + _npy_shape(shape) + ", }"
    )
    header_bytes = header.encode("latin1")
    padding = (16 - ((10 + len(header_bytes) + 1) % 16)) % 16
    header_bytes += b" " * padding + b"\n"

    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(b"\x01\x00")
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        f.write(arr.tobytes())


def write_png(path: str, width: int, height: int, rgb: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes(width, height, rgb))


def write_boundary_preview_svg(
    path: str,
    rig: CameraRig,
    scene: object,
    boundary: Sequence[int],
) -> None:
    if isinstance(scene, HalfOcclusionScene):
        w, h = scene.width, scene.height
        cells = []
        stride = max(1, min(w, h) // 256)
        for y in range(0, h, stride):
            row = y * w
            for x in range(0, w, stride):
                if boundary[row + x]:
                    cells.append(
                        f'  <rect x="{x}" y="{y}" width="{stride}" height="{stride}" '
                        f'fill="#d62728" opacity="0.45"/>\n'
                    )

        with open(path, "w", encoding="utf-8") as f:
            f.write(_svg_header(w, h))
            f.write(_svg_defs())
            f.write(_half_occlusion_svg_body(scene, "cyclopean"))
            f.write("".join(cells))
            f.write("</svg>\n")
        return

    if isinstance(scene, WallpaperScene):
        w, h = scene.width, scene.height
        cells = []
        stride = max(1, min(w, h) // 256)
        for y in range(0, h, stride):
            row = y * w
            for x in range(0, w, stride):
                if boundary[row + x]:
                    cells.append(
                        f'  <rect x="{x}" y="{y}" width="{stride}" height="{stride}" '
                        f'fill="#d62728" opacity="0.45"/>\n'
                    )

        with open(path, "w", encoding="utf-8") as f:
            f.write(_svg_header(w, h))
            f.write(_svg_defs())
            f.write(_wallpaper_svg_body(scene, "cyclopean"))
            f.write("".join(cells))
            f.write("</svg>\n")
        return

    w, h = scene.width, scene.height
    cells = []
    stride = max(1, min(w, h) // 256)
    for y in range(0, h, stride):
        row = y * w
        for x in range(0, w, stride):
            if boundary[row + x]:
                cells.append(
                    f'  <rect x="{x}" y="{y}" width="{stride}" height="{stride}" '
                    f'fill="#d62728" opacity="0.45"/>\n'
                )

    with open(path, "w", encoding="utf-8") as f:
        f.write(_svg_header(w, h))
        f.write(_svg_defs())
        f.write(_scene_svg_body(rig, scene, "cyclopean"))
        f.write("".join(cells))
        f.write("</svg>\n")


def write_metadata(path: str, rig: CameraRig, scene: object, spec: SceneSpec) -> None:
    if isinstance(scene, HalfOcclusionScene):
        disc_radius_world = scene.near_radius_px * scene.disc_z / rig.focal_px
        disc_diameter_world = scene.disc_diameter_px * scene.disc_z / rig.focal_px
        stadium_half_length_px = (
            scene.far_stadium_half_length_px if scene.variant == "pimple" else scene.near_stadium_half_length_px
        )
        stadium_half_length_world = stadium_half_length_px * scene.stadium_z / rig.focal_px
        stadium_diameter_world = scene.disc_diameter_px * scene.stadium_z / rig.focal_px
        stadium_straight_length_world = scene.crescent_width_px * scene.stadium_z / rig.focal_px
        background_disc_radius_world = scene.near_radius_px * scene.background_z / rig.focal_px
        background_disc_diameter_world = scene.disc_diameter_px * scene.background_z / rig.focal_px
        background_support_half_length_world = scene.far_stadium_half_length_px * scene.background_z / rig.focal_px
        near_radius_world = scene.near_radius_px * scene.near_z / rig.focal_px
        near_stadium_half_length_world = scene.near_stadium_half_length_px * scene.near_z / rig.focal_px
        far_disc_radius_world = scene.far_disc_radius_px * scene.far_z / rig.focal_px
        far_stadium_half_length_world = scene.far_stadium_half_length_px * scene.far_z / rig.focal_px
        if scene.variant == "pimple":
            description = (
                "Belhumeur et al. Figure 2 pimple stereogram using the Section 3 control scheme "
                "(d1, D_img, w): a foreground white disc at d1 and a midground gray stadium at d2."
            )
        else:
            description = (
                "Belhumeur et al. Figure 2 dimple stereogram using the Section 3 control scheme "
                "(d1, D_img, w): a stadium cutout at d2 revealing a background white disc and gray plane at d3."
            )
        data: Dict[str, object] = {
            "camera": {
                "focal_px": rig.focal_px,
                "baseline": rig.baseline,
                "principal_point_px": [rig.cx, rig.cy],
                "convention": "x_left = x_cyclopean + disparity/2; x_right = x_cyclopean - disparity/2",
            },
            "image_size": [scene.height, scene.width],
            "figure": {
                "scene_key": spec.key,
                "source": spec.source,
                "fusion": spec.fusion,
                "crossed_pair_order": {
                    "left_column": "physical right-eye image",
                    "right_column": "physical left-eye image",
                },
            },
            "half_occlusion": {
                "variant": scene.variant,
                "percept": scene.percept_label,
                "target_center_px": [scene.center.x, scene.center.y],
                "control_d1_px": scene.disc_disparity_px,
                "control_disc_diameter_px": scene.disc_diameter_px,
                "control_w": scene.crescent_fraction,
                "crescent_width_px": scene.crescent_width_px,
                "three_depth_relation": "d_background = d_disc - 2*w*D_img = 2*d_stadium - d_disc",
                "exact_swap_relation": "d_background = 2 * d_stadium - d_disc",
                "disc_z": scene.disc_z,
                "stadium_z": scene.stadium_z,
                "background_z": scene.background_z,
                "disc_disparity_px": scene.disc_disparity_px,
                "stadium_disparity_px": scene.stadium_disparity_px,
                "background_disparity_px": scene.background_disparity_px,
                "disc_diameter_px": scene.disc_diameter_px,
                "disc_diameter_world": disc_diameter_world,
                "disc_radius_px": scene.near_radius_px,
                "disc_radius_world": disc_radius_world,
                "stadium_diameter_px": scene.disc_diameter_px,
                "stadium_diameter_world": stadium_diameter_world,
                "stadium_straight_length_px": scene.crescent_width_px,
                "stadium_straight_length_world": stadium_straight_length_world,
                "stadium_half_length_px": stadium_half_length_px,
                "stadium_half_length_world": stadium_half_length_world,
                "background_disc_diameter_px": scene.disc_diameter_px,
                "background_disc_diameter_world": background_disc_diameter_world,
                "background_disc_radius_px": scene.near_radius_px,
                "background_disc_radius_world": background_disc_radius_world,
                "background_support_half_length_px": scene.far_stadium_half_length_px,
                "background_support_half_length_world": background_support_half_length_world,
                "near_center_offset_px": scene.near_center_offset_px,
                "far_center_offset_px": scene.far_center_offset_px,
                "near_z": scene.near_z,
                "far_z": scene.far_z,
                "near_disparity_px": scene.near_disparity_px,
                "far_disparity_px": scene.far_disparity_px,
                "near_radius_px": scene.near_radius_px,
                "near_radius_world": near_radius_world,
                "near_stadium_half_length_px": scene.near_stadium_half_length_px,
                "near_stadium_half_length_world": near_stadium_half_length_world,
                "far_disc_radius_px": scene.far_disc_radius_px,
                "far_disc_radius_world": far_disc_radius_world,
                "far_stadium_half_length_px": scene.far_stadium_half_length_px,
                "far_stadium_half_length_world": far_stadium_half_length_world,
                "far_support_half_length_px": scene.far_stadium_half_length_px,
                "far_support_half_length_world": far_stadium_half_length_world,
                "ring_radii_px": list(scene.ring_radii_px),
                "stroke_px": scene.stroke_px,
                "gray_color_rgb": list(scene.gray_color),
                "description": description,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return

    if isinstance(scene, WallpaperScene):
        data = {
            "camera": {
                "focal_px": rig.focal_px,
                "baseline": rig.baseline,
                "principal_point_px": [rig.cx, rig.cy],
                "convention": "x_left = x_cyclopean + disparity/2; x_right = x_cyclopean - disparity/2",
            },
            "image_size": [scene.height, scene.width],
            "figure": {
                "scene_key": spec.key,
                "source": spec.source,
                "fusion": spec.fusion,
            },
            "wallpaper": {
                "percept": scene.percept,
                "luminance_preset": scene.luminance_preset,
                "surround_gray": scene.surround_gray,
                "stripe_light_gray": scene.stripe_light_gray,
                "stripe_dark_gray": scene.stripe_dark_gray,
                "stripe_count": scene.stripe_count,
                "stripe_width_px": scene.stripe_width_px,
                "shift_px": scene.shift_px,
                "patch_bounds_cyclopean_px": {
                    "left": scene.patch_left_px,
                    "right": scene.patch_right_px,
                    "top": scene.patch_top_px,
                    "bottom": scene.patch_bottom_px,
                },
                "surround_z": scene.surround_z,
                "surround_disparity_px": scene.surround_disparity_px,
                "wallpaper_z": scene.wallpaper_z,
                "wallpaper_disparity_px": scene.wallpaper_disparity_px,
                "surround_dot_count": len(scene.dots),
                "surround_dot_radius_px": scene.dot_radius_px,
                "surround_dot_seed": scene.dot_seed,
                "surround_dot_disparity_px": scene.surround_disparity_px,
                "description": (
                    "Anderson & Nakayama (1994) Figure 6 ambiguous wallpaper pattern. The stripe "
                    "pattern is shifted by exactly one stripe width (one half cycle) between the "
                    "eyes, which makes the 'front' (wallpaper nearer, occludes surround) and "
                    "'behind' (wallpaper farther, occluded by the surround's aperture) "
                    "interpretations equally valid explanations of the same kind of display; this "
                    "file records the '" + scene.percept + "' interpretation specifically."
                ),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return

    figure: Dict[str, object] = {
        "scene_key": spec.key,
        "source": spec.source,
        "fusion": spec.fusion,
        "lines_mode": getattr(scene, "mode", "forest"),
    }
    if spec.fusion == "crossed":
        figure["crossed_pair_order"] = {
            "left_column": "physical right-eye image",
            "right_column": "physical left-eye image",
        }

    lines_plane = getattr(scene, "lines_plane", None)

    data: Dict[str, object] = {
        "camera": {
            "focal_px": rig.focal_px,
            "baseline": rig.baseline,
            "principal_point_px": [rig.cx, rig.cy],
            "convention": "x_left = x_cyclopean + disparity/2; x_right = x_cyclopean - disparity/2",
        },
        "image_size": [scene.height, scene.width],
        "figure": figure,
        "plane": {
            "z": scene.plane_z,
            "disparity_px": rig.disparity(scene.plane_z),
            "top_boundary_y_px_cyclopean": scene.plane_top_y,
            "texture": "black dots on an otherwise white plane",
        },
        "dots": [
            {
                "center_world": [dot.center.x, dot.center.y, dot.center.z],
                "center_cyclopean_px": [
                    rig.project(dot.center, "cyclopean").x,
                    rig.project(dot.center, "cyclopean").y,
                ],
                "radius_px": rig.focal_px * dot.radius_world / dot.center.z,
            }
            for dot in scene.dots
        ],
        "lines": [
            {
                "label": line.label,
                "z": line.z,
                "disparity_px": rig.disparity(line.z),
                "p0_z": line.p0.z,
                "p1_z": line.p1.z,
                "p0_disparity_px": rig.disparity(line.p0.z),
                "p1_disparity_px": rig.disparity(line.p1.z),
                "p0_world": [line.p0.x, line.p0.y, line.p0.z],
                "p1_world": [line.p1.x, line.p1.y, line.p1.z],
                "p0_cyclopean_px": [
                    rig.project(line.p0, "cyclopean").x,
                    rig.project(line.p0, "cyclopean").y,
                ],
                "p1_cyclopean_px": [
                    rig.project(line.p1, "cyclopean").x,
                    rig.project(line.p1, "cyclopean").y,
                ],
                "stroke_px": line.stroke_px,
            }
            for line in scene.lines
        ],
    }
    if lines_plane is not None:
        data["lines_plane"] = {
            "tilt_x_deg": lines_plane.tilt_x_deg,
            "tilt_y_deg": lines_plane.tilt_y_deg,
            "ref_z": lines_plane.ref_z,
            "normal_world": [lines_plane.nx, lines_plane.ny, lines_plane.nz],
            "plane_constant_k": lines_plane.k,
            "description": (
                "Single slanted plane N.X=k containing all seven forest line segments, "
                "positioned behind and tilted relative to the fronto-parallel dotted plane. "
                "disp/disp_left/seg/udf are filled solidly across the whole forest region "
                "with this plane's values, not just under the drawn line strokes."
            ),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_npy(path: str) -> NpyArray:
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path!r} is not a .npy file")
        major, minor = f.read(2)
        if (major, minor) == (1, 0):
            header_len = struct.unpack("<H", f.read(2))[0]
        elif (major, minor) in ((2, 0), (3, 0)):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"unsupported .npy version {(major, minor)} in {path!r}")
        header = ast.literal_eval(f.read(header_len).decode("latin1"))
        payload = f.read()

    if header.get("fortran_order"):
        raise ValueError(f"Fortran-order arrays are not supported: {path!r}")

    descr = header["descr"]
    shape = tuple(int(v) for v in header["shape"])
    typecode = {
        "|u1": "B",
        "<u1": "B",
        "|i1": "b",
        "<f4": "f",
        "<f8": "d",
    }.get(descr)
    if typecode is None:
        raise ValueError(f"unsupported dtype {descr!r} in {path!r}")

    data = array(typecode)
    data.frombytes(payload)
    if descr.startswith(">") or (descr.startswith("<") and sys.byteorder != "little"):
        data.byteswap()

    expected = 1
    for dim in shape:
        expected *= dim
    if len(data) != expected:
        raise ValueError(f"{path!r} has {len(data)} values, expected {expected}")

    return NpyArray(shape=shape, descr=descr, data=data)


def png_bytes(width: int, height: int, rgb: Sequence[int]) -> bytes:
    rgb_bytes = bytes(rgb)
    if len(rgb_bytes) != width * height * 3:
        raise ValueError("RGB byte buffer size does not match image dimensions")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) +
            kind +
            data +
            struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)
        start = y * row_bytes
        raw.extend(rgb_bytes[start:start + row_bytes])

    return (
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(bytes(raw), level=6)) +
        chunk(b"IEND", b"")
    )


def png_data_uri(width: int, height: int, rgb: Sequence[int]) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(width, height, rgb)).decode("ascii")


def png_file_data_uri(path: str) -> ImageAsset:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 24 or not raw.startswith(b"\x89PNG\r\n\x1a\n") or raw[12:16] != b"IHDR":
        raise ValueError(f"{path!r} is not a PNG file")
    width, height = struct.unpack(">II", raw[16:24])
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    return ImageAsset(width, height, uri)


def svg_data_uri(path: str) -> SvgAsset:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    width, height = parse_svg_size(text)
    uri = "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")
    return SvgAsset(path=path, width=width, height=height, data_uri=uri)


def parse_svg_size(text: str) -> Tuple[float, float]:
    viewbox = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', text)
    if viewbox:
        parts = [float(v) for v in re.split(r"[\s,]+", viewbox.group(1).strip()) if v]
        if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
            return parts[2], parts[3]

    width = re.search(r'width\s*=\s*["\']([0-9.]+)', text)
    height = re.search(r'height\s*=\s*["\']([0-9.]+)', text)
    if width and height:
        return float(width.group(1)), float(height.group(1))
    return 512.0, 512.0


def image_dims_from_shape(shape: Sequence[int]) -> Tuple[int, int]:
    if len(shape) == 2:
        return shape[1], shape[0]
    if len(shape) == 3 and shape[2] == 1:
        return shape[1], shape[0]
    raise ValueError(f"expected HxW or HxWx1 array, got shape {shape}")


def value_range(values: Sequence[float], *, positive_only: bool = False) -> Tuple[float, float]:
    if positive_only:
        vals = [float(v) for v in values if float(v) > 0.0]
    else:
        vals = [float(v) for v in values]
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def interpolate_palette(t: float, stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for idx in range(1, len(stops)):
        p0, c0 = stops[idx - 1]
        p1, c1 = stops[idx]
        if t <= p1:
            local = 0.0 if math.isclose(p0, p1) else (t - p0) / (p1 - p0)
            return (
                lerp(c0[0], c1[0], local),
                lerp(c0[1], c1[1], local),
                lerp(c0[2], c1[2], local),
            )
    return stops[-1][1]


def colorize_float_map(npy: NpyArray, *, zero_is_background: bool) -> Tuple[ImageAsset, str]:
    width, height = image_dims_from_shape(npy.shape)
    lo, hi = value_range(npy.data, positive_only=zero_is_background)
    stops = (
        (0.00, (47, 55, 142)),
        (0.25, (38, 139, 209)),
        (0.50, (45, 181, 126)),
        (0.75, (245, 210, 80)),
        (1.00, (189, 50, 38)),
    )

    rgb = bytearray()
    for value in npy.data:
        value = float(value)
        if zero_is_background and value <= 0.0:
            rgb.extend((245, 245, 245))
            continue
        t = (value - lo) / (hi - lo)
        rgb.extend(interpolate_palette(t, stops))

    subtitle = f"{width}x{height}, range {lo:.3g} to {hi:.3g}"
    return ImageAsset(width, height, png_data_uri(width, height, bytes(rgb))), subtitle


def colorize_segmentation(npy: NpyArray) -> Tuple[ImageAsset, str]:
    width, height = image_dims_from_shape(npy.shape)
    palette = [
        (255, 255, 255),
        (188, 228, 232),
        (49, 84, 170),
        (213, 94, 0),
        (0, 158, 115),
        (204, 121, 167),
        (230, 159, 0),
        (86, 180, 233),
        (0, 114, 178),
        (180, 180, 180),
    ]
    labels = sorted(set(int(v) for v in npy.data))
    rgb = bytearray()
    for value in npy.data:
        rgb.extend(palette[int(value) % len(palette)])

    subtitle = f"{width}x{height}, labels " + ", ".join(str(v) for v in labels)
    return ImageAsset(width, height, png_data_uri(width, height, bytes(rgb))), subtitle


def relative_label(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def load_metadata_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    camera = meta.get("camera", {})
    plane = meta.get("plane", {})
    lines = meta.get("lines", [])
    figure = meta.get("figure", {})
    order = figure.get("crossed_pair_order", {})
    half_occlusion = meta.get("half_occlusion")

    if isinstance(half_occlusion, dict):
        metadata_lines = [
            f"scene key: {figure.get('scene_key', 'n/a')}",
            f"source: {figure.get('source', 'n/a')}",
            f"fusion: {figure.get('fusion', 'n/a')}",
            f"variant: {half_occlusion.get('variant', 'n/a')}",
            f"percept: {half_occlusion.get('percept', 'n/a')}",
            f"image size HxW: {meta.get('image_size', 'n/a')}",
            f"target center: {half_occlusion.get('target_center_px', 'n/a')}",
            f"control d1: {half_occlusion.get('control_d1_px', half_occlusion.get('disc_disparity_px', 'n/a'))} px",
            f"control D_img: {half_occlusion.get('control_disc_diameter_px', half_occlusion.get('disc_diameter_px', 'n/a'))} px",
            f"control w: {half_occlusion.get('control_w', 'n/a')}",
            f"crescent width: {half_occlusion.get('crescent_width_px', 'n/a')} px",
            f"three-depth relation: {half_occlusion.get('three_depth_relation', half_occlusion.get('exact_swap_relation', 'n/a'))}",
            f"disc z: {half_occlusion.get('disc_z', 'n/a')}",
            f"stadium/hole z: {half_occlusion.get('stadium_z', 'n/a')}",
            f"background z: {half_occlusion.get('background_z', 'n/a')}",
            f"disc disparity d1: {half_occlusion.get('disc_disparity_px', 'n/a')} px",
            f"stadium/hole disparity d2: {half_occlusion.get('stadium_disparity_px', 'n/a')} px",
            f"background disparity d3: {half_occlusion.get('background_disparity_px', 'n/a')} px",
            f"near center offset: {half_occlusion.get('near_center_offset_px', 'n/a')}",
            f"far center offset: {half_occlusion.get('far_center_offset_px', 'n/a')}",
            f"near z: {half_occlusion.get('near_z', 'n/a')}",
            f"far z: {half_occlusion.get('far_z', 'n/a')}",
            f"near disparity: {half_occlusion.get('near_disparity_px', 'n/a')} px",
            f"far disparity: {half_occlusion.get('far_disparity_px', 'n/a')} px",
            f"disc diameter: {half_occlusion.get('disc_diameter_px', 'n/a')}",
            f"disc radius: {half_occlusion.get('disc_radius_px', half_occlusion.get('near_radius_px', 'n/a'))}",
            f"stadium straight-length: {half_occlusion.get('stadium_straight_length_px', half_occlusion.get('crescent_width_px', 'n/a'))}",
            f"stadium half-length: {half_occlusion.get('stadium_half_length_px', 'n/a')}",
            f"near stadium half-length: {half_occlusion.get('near_stadium_half_length_px', 'n/a')}",
            f"far disc radius: {half_occlusion.get('far_disc_radius_px', 'n/a')}",
            f"background support half-length: {half_occlusion.get('background_support_half_length_px', half_occlusion.get('far_support_half_length_px', half_occlusion.get('far_stadium_half_length_px', 'n/a')))}",
            f"ring radii: {half_occlusion.get('ring_radii_px', 'n/a')}",
        ]
        if order:
            metadata_lines.extend([
                f"crossed left column: {order.get('left_column', 'n/a')}",
                f"crossed right column: {order.get('right_column', 'n/a')}",
            ])
        return metadata_lines

    wallpaper = meta.get("wallpaper")
    if isinstance(wallpaper, dict):
        bounds = wallpaper.get("patch_bounds_cyclopean_px", {})
        metadata_lines = [
            f"scene key: {figure.get('scene_key', 'n/a')}",
            f"source: {figure.get('source', 'n/a')}",
            f"fusion: {figure.get('fusion', 'n/a')}",
            f"percept: {wallpaper.get('percept', 'n/a')}",
            f"surround luminance preset: {wallpaper.get('luminance_preset', 'n/a')}",
            f"focal_px: {camera.get('focal_px', 'n/a')}",
            f"baseline: {camera.get('baseline', 'n/a')}",
            f"image size HxW: {meta.get('image_size', 'n/a')}",
            f"surround gray: {wallpaper.get('surround_gray', 'n/a')}",
            f"stripe light/dark gray: {wallpaper.get('stripe_light_gray', 'n/a')}/{wallpaper.get('stripe_dark_gray', 'n/a')}",
            f"stripe count: {wallpaper.get('stripe_count', 'n/a')}",
            f"stripe width: {wallpaper.get('stripe_width_px', 0):.2f} px",
            f"shift (half cycle): {wallpaper.get('shift_px', 0):.2f} px",
            f"patch bounds cyclopean (L,R,T,B) px: {bounds.get('left', 'n/a')}, {bounds.get('right', 'n/a')}, {bounds.get('top', 'n/a')}, {bounds.get('bottom', 'n/a')}",
            f"surround z: {wallpaper.get('surround_z', 0):.3g}",
            f"surround disparity: {wallpaper.get('surround_disparity_px', 0):.3g} px",
            f"wallpaper z: {wallpaper.get('wallpaper_z', 0):.3g}",
            f"wallpaper disparity: {wallpaper.get('wallpaper_disparity_px', 0):.3g} px",
            f"surround dot count: {wallpaper.get('surround_dot_count', 'n/a')}",
            f"surround dot radius: {wallpaper.get('surround_dot_radius_px', 0):.2f} px",
            f"surround dot disparity: {wallpaper.get('surround_dot_disparity_px', 0):.3g} px",
        ]
        return metadata_lines

    line_depths = ", ".join(f"{item.get('z', 0):.2f}" for item in lines)
    line_disps = ", ".join(f"{item.get('disparity_px', 0):.2f}" for item in lines)
    lines_plane = meta.get("lines_plane")

    metadata_lines = [
        f"scene key: {figure.get('scene_key', 'n/a')}",
        f"source: {figure.get('source', 'n/a')}",
        f"fusion: {figure.get('fusion', 'n/a')}",
        f"lines mode: {figure.get('lines_mode', 'forest')}",
        f"focal_px: {camera.get('focal_px', 'n/a')}",
        f"baseline: {camera.get('baseline', 'n/a')}",
        f"principal point: {camera.get('principal_point_px', 'n/a')}",
        f"image size HxW: {meta.get('image_size', 'n/a')}",
        f"plane z: {plane.get('z', 0):.3g}",
        f"plane disparity: {plane.get('disparity_px', 0):.3g} px",
        f"line count: {len(lines)}",
        f"line mean-z values: {line_depths}",
        f"line mean-disparities: {line_disps}",
    ]
    if isinstance(lines_plane, dict):
        metadata_lines.extend([
            f"forest plane tilt x/y (deg): {lines_plane.get('tilt_x_deg', 'n/a')}/{lines_plane.get('tilt_y_deg', 'n/a')}",
            f"forest plane ref_z (nearest boundary depth): {lines_plane.get('ref_z', 'n/a')}",
        ])
    if order:
        metadata_lines.extend([
            f"crossed left column: {order.get('left_column', 'n/a')}",
            f"crossed right column: {order.get('right_column', 'n/a')}",
        ])
    return metadata_lines


def build_summary_panels(input_dir: str, stem: str) -> List[Panel]:
    paths = output_paths(input_dir, stem)
    expected_keys = (
        "cyclopean",
        "left",
        "right",
        "crossed",
        "parallel",
        "cyclopean_png",
        "left_png",
        "right_png",
        "crossed_png",
        "parallel_png",
        "disp",
        "disp_left",
        "seg",
        "field",
        "udf",
        "udf_png",
        "metadata",
    )
    missing = [paths[key] for key in expected_keys if not os.path.exists(paths[key])]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"missing expected output files:\n  {formatted}")

    disp_asset, disp_subtitle = colorize_float_map(read_npy(paths["disp"]), zero_is_background=True)
    left_disp_asset, left_disp_subtitle = colorize_float_map(read_npy(paths["disp_left"]), zero_is_background=True)
    seg_asset, seg_subtitle = colorize_segmentation(read_npy(paths["seg"]))
    field_asset, field_subtitle = colorize_float_map(read_npy(paths["field"]), zero_is_background=False)

    return [
        Panel("Cyclopean image", relative_label(paths["cyclopean"], input_dir), svg_data_uri(paths["cyclopean"])),
        Panel("Physical left image", relative_label(paths["left"], input_dir), svg_data_uri(paths["left"])),
        Panel("Physical right image", relative_label(paths["right"], input_dir), svg_data_uri(paths["right"])),
        Panel("Crossed-fusion pair", relative_label(paths["crossed"], input_dir), svg_data_uri(paths["crossed"])),
        Panel("Parallel-order pair", relative_label(paths["parallel"], input_dir), svg_data_uri(paths["parallel"])),
        Panel("Cyclopean PNG", relative_label(paths["cyclopean_png"], input_dir), png_file_data_uri(paths["cyclopean_png"])),
        Panel("Physical left PNG", relative_label(paths["left_png"], input_dir), png_file_data_uri(paths["left_png"])),
        Panel("Physical right PNG", relative_label(paths["right_png"], input_dir), png_file_data_uri(paths["right_png"])),
        Panel("Crossed pair PNG", relative_label(paths["crossed_png"], input_dir), png_file_data_uri(paths["crossed_png"])),
        Panel("Parallel pair PNG", relative_label(paths["parallel_png"], input_dir), png_file_data_uri(paths["parallel_png"])),
        Panel("Cyclopean disparity", disp_subtitle, disp_asset),
        Panel("Left-view disparity", left_disp_subtitle, left_disp_asset),
        Panel("Segmentation map", seg_subtitle, seg_asset),
        Panel("UDF field", field_subtitle, field_asset),
        Panel("UDF boundary preview", relative_label(paths["udf"], input_dir), svg_data_uri(paths["udf"])),
        Panel("UDF preview PNG", relative_label(paths["udf_png"], input_dir), png_file_data_uri(paths["udf_png"])),
        Panel("Scene metadata", relative_label(paths["metadata"], input_dir), text_lines=load_metadata_lines(paths["metadata"])),
    ]


def panel_svg(panel: Panel, x: int, y: int, width: int, height: int) -> str:
    pad = 14
    title_h = 24
    subtitle_h = 18
    content_x = x + pad
    content_y = y + pad + title_h + subtitle_h + 6
    content_w = width - 2 * pad
    content_h = height - 2 * pad - title_h - subtitle_h - 6

    parts = [
        f'<g transform="translate({x},{y})">\n',
        f'  <rect x="0" y="0" width="{width}" height="{height}" rx="7" fill="#fff" stroke="#c9ced6"/>\n',
        f'  <text x="{pad}" y="{pad + 16}" class="panel-title">{html.escape(panel.title)}</text>\n',
        f'  <text x="{pad}" y="{pad + 38}" class="panel-subtitle">{html.escape(panel.subtitle)}</text>\n',
    ]

    if panel.asset is not None:
        asset = panel.asset
        asset_w = float(asset.width)
        asset_h = float(asset.height)
        scale = min(content_w / asset_w, content_h / asset_h)
        draw_w = asset_w * scale
        draw_h = asset_h * scale
        draw_x = content_x - x + 0.5 * (content_w - draw_w)
        draw_y = content_y - y + 0.5 * (content_h - draw_h)
        parts.append(
            f'  <image x="{draw_x:.3f}" y="{draw_y:.3f}" width="{draw_w:.3f}" '
            f'height="{draw_h:.3f}" href="{asset.data_uri}" preserveAspectRatio="xMidYMid meet"/>\n'
        )
    elif panel.text_lines is not None:
        text_x = content_x - x
        text_y = content_y - y + 16
        parts.append(f'  <text x="{text_x}" y="{text_y}" class="metadata-text">\n')
        for idx, line in enumerate(panel.text_lines):
            dy = 0 if idx == 0 else 20
            parts.append(f'    <tspan x="{text_x}" dy="{dy}">{html.escape(line)}</tspan>\n')
        parts.append("  </text>\n")

    parts.append("</g>\n")
    return "".join(parts)


def write_summary_svg(
    path: str,
    panels: Sequence[Panel],
    *,
    input_dir: str,
    stem: str,
    summary_title: str,
    columns: int,
) -> None:
    panel_w = 420
    panel_h = 350
    gap = 22
    margin = 30
    title_h = 76
    rows = math.ceil(len(panels) / columns)
    width = 2 * margin + columns * panel_w + (columns - 1) * gap
    height = title_h + margin + rows * panel_h + (rows - 1) * gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n',
        "  <defs>\n",
        "    <style>\n",
        "      text { font-family: Helvetica, Arial, sans-serif; fill: #1f2933; }\n",
        "      .title { font-size: 28px; font-weight: 700; }\n",
        "      .subtitle { font-size: 13px; fill: #53606f; }\n",
        "      .panel-title { font-size: 16px; font-weight: 700; }\n",
        "      .panel-subtitle { font-size: 11px; fill: #6b7280; }\n",
        "      .metadata-text { font-size: 12px; fill: #2e3440; }\n",
        "    </style>\n",
        "  </defs>\n",
        '  <rect x="0" y="0" width="100%" height="100%" fill="#f4f6f8"/>\n',
        f'  <text x="{margin}" y="38" class="title">{html.escape(summary_title)}</text>\n',
        (
            f'  <text x="{margin}" y="62" class="subtitle">stem: {html.escape(stem)} '
            f'| input: {html.escape(os.path.abspath(input_dir))}</text>\n'
        ),
    ]

    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = margin + col * (panel_w + gap)
        y = title_h + row * (panel_h + gap)
        parts.append(panel_svg(panel, x, y, panel_w, panel_h))

    parts.append("</svg>\n")

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def parse_line_zs(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("--line-zs must contain exactly seven comma-separated depths")
    return values


def build_gillam2002_scene(rig: CameraRig, args: argparse.Namespace) -> Scene:
    return make_figure9a_scene(
        rig,
        mode=args.gillam_lines_mode,
        plane_z=args.plane_z,
        line_zs=args.line_zs,
        plane_top_frac=args.plane_top_frac,
        dot_radius_frac=args.dot_radius_frac,
        line_stroke_frac=args.line_stroke_frac,
        plane_slant_tilt_x_deg=args.plane_slant_tilt_x_deg,
        plane_slant_tilt_y_deg=args.plane_slant_tilt_y_deg,
        plane_slant_ref_z=args.plane_slant_ref_z,
    )


def build_belhumeur1996_dimple_scene(rig: CameraRig, args: argparse.Namespace) -> HalfOcclusionScene:
    return make_belhumeur1996_figure2_scene(
        rig,
        variant="dimple",
        percept_label="dimple",
        d1_px=abs(args.figure2_d1_px),
        disc_diameter_px=args.figure2_dimg_px,
        crescent_fraction=args.figure2_w,
        gray_value=args.figure2_gray,
    )


def build_belhumeur1996_pimple_scene(rig: CameraRig, args: argparse.Namespace) -> HalfOcclusionScene:
    return make_belhumeur1996_figure2_scene(
        rig,
        variant="pimple",
        percept_label="pimple",
        d1_px=abs(args.figure2_d1_px),
        disc_diameter_px=args.figure2_dimg_px,
        crescent_fraction=args.figure2_w,
        gray_value=args.figure2_gray,
    )


def _build_wallpaper_scene(rig: CameraRig, args: argparse.Namespace, percept: str) -> WallpaperScene:
    return make_wallpaper_scene(
        rig,
        percept=percept,
        surround_z=args.wallpaper_surround_z,
        surround_gray=WALLPAPER_SURROUND_LUMINANCE_PRESETS[args.wallpaper_surround_luminance],
        stripe_light_gray=args.wallpaper_stripe_light_gray,
        stripe_dark_gray=args.wallpaper_stripe_dark_gray,
        stripe_width_frac=args.wallpaper_stripe_width_frac,
        stripe_count=args.wallpaper_stripe_count,
        patch_height_frac=args.wallpaper_patch_height_frac,
        luminance_preset=args.wallpaper_surround_luminance,
        dot_count=args.wallpaper_dot_count,
        dot_radius_frac=args.wallpaper_dot_radius_frac,
        dot_seed=args.wallpaper_dot_seed,
    )


def build_andersonnakayama1994_wallpaper_front_scene(rig: CameraRig, args: argparse.Namespace) -> WallpaperScene:
    return _build_wallpaper_scene(rig, args, "front")


def build_andersonnakayama1994_wallpaper_behind_scene(rig: CameraRig, args: argparse.Namespace) -> WallpaperScene:
    return _build_wallpaper_scene(rig, args, "behind")


SCENE_SPECS: Dict[str, SceneSpec] = {
    "gillam2002": SceneSpec(
        key="gillam2002",
        prefix="gillam2002",
        summary_title="Gillam 2002 Figure 9A Output Summary",
        source="Gillam & Nakayama Subjective Contours paper, Figure 9A",
        fusion="crossed",
        build_scene=build_gillam2002_scene,
    ),
    "belhumeur1996_dimple": SceneSpec(
        key="belhumeur1996_dimple",
        prefix="belhumeur1996_dimple",
        summary_title="Belhumeur 1996 Figure 2 Dimple Summary",
        source="Belhumeur et al. Stereo 1996, Figure 2 half-occlusion demo",
        fusion="crossed",
        build_scene=build_belhumeur1996_dimple_scene,
    ),
    "belhumeur1996_pimple": SceneSpec(
        key="belhumeur1996_pimple",
        prefix="belhumeur1996_pimple",
        summary_title="Belhumeur 1996 Figure 2 Pimple Summary",
        source="Belhumeur et al. Stereo 1996, Figure 2 half-occlusion demo",
        fusion="crossed",
        build_scene=build_belhumeur1996_pimple_scene,
    ),
    "andersonnakayama1994_wallpaper": SceneSpec(
        key="andersonnakayama1994_wallpaper",
        prefix="andersonnakayama1994_wallpaper",
        summary_title="Anderson & Nakayama 1994 Figure 6 Wallpaper Summary",
        source="Anderson & Nakayama, Toward a General Theory of Stereopsis, Figure 6",
        fusion="crossed",
        variant_build_scenes={
            "front": build_andersonnakayama1994_wallpaper_front_scene,
            "behind": build_andersonnakayama1994_wallpaper_behind_scene,
        },
        prefix_suffix=lambda args: args.wallpaper_surround_luminance,
    ),
}

SCENE_ALIASES = {
    "gillam": "gillam2002",
    "gillam2002": "gillam2002",
    "figure9a": "gillam2002",
    "belhumeur1996dimple": "belhumeur1996_dimple",
    "belhumeur1996pimple": "belhumeur1996_pimple",
    "andersonnakayama1994wallpaper": "andersonnakayama1994_wallpaper",
    "andersonnakayama1994": "andersonnakayama1994_wallpaper",
    "wallpaper1994": "andersonnakayama1994_wallpaper",
    "wallpaper": "andersonnakayama1994_wallpaper",
    "andersonnakayama": "andersonnakayama1994_wallpaper",
}


def available_scene_names() -> str:
    return ", ".join(sorted(SCENE_SPECS))


def canonical_scene_key(text: Optional[str]) -> str:
    raw = DEFAULT_SCENE_KEY if text is None else text
    key = "".join(ch for ch in raw.strip().lower() if ch.isalnum())
    return SCENE_ALIASES.get(key, key)


def resolve_scene_spec(text: Optional[str]) -> SceneSpec:
    key = canonical_scene_key(text)
    spec = SCENE_SPECS.get(key)
    if spec is None:
        raise ValueError(f"unknown scene {text!r}; available scenes: {available_scene_names()}")
    return spec


def normalize_stem(stem: Optional[str], prefix: str) -> str:
    if stem is None:
        return prefix
    stem = stem.strip().replace(os.sep, "_")
    if not stem:
        return prefix
    if stem == prefix or stem.startswith(f"{prefix}_"):
        return stem
    return f"{prefix}_{stem}"


def generate_scene_outputs(
    rig: CameraRig,
    scene: object,
    spec: SceneSpec,
    args: argparse.Namespace,
    out_root: str,
    stem: str,
    summary_title: str,
) -> None:
    width, height = rig.width, rig.height
    paths = output_paths(out_root, stem)
    ensure_output_dir(out_root)

    write_scene_svg(
        paths["left"], rig, scene, "left",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )
    write_scene_svg(
        paths["right"], rig, scene, "right",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )
    write_scene_svg(
        paths["cyclopean"], rig, scene, "cyclopean",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )

    pair_gap = int(round(args.pair_gap_frac * width))
    write_pair_svg(
        paths["crossed"], rig, scene, crossed=True, gap=pair_gap, show_plane_outline=args.show_plane_outline,
    )
    write_pair_svg(
        paths["parallel"], rig, scene, crossed=False, gap=pair_gap, show_plane_outline=args.show_plane_outline,
    )

    left_rgb = rasterize_scene_rgb(
        rig, scene, "left",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )
    right_rgb = rasterize_scene_rgb(
        rig, scene, "right",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )
    cyclopean_rgb = rasterize_scene_rgb(
        rig, scene, "cyclopean",
        show_plane_outline=args.show_plane_outline, show_boundary_preview=args.show_boundary_preview,
    )
    crossed_rgb, crossed_width, crossed_height = rasterize_pair_rgb(
        rig, scene, crossed=True, gap=pair_gap, show_plane_outline=args.show_plane_outline,
    )
    parallel_rgb, parallel_width, parallel_height = rasterize_pair_rgb(
        rig, scene, crossed=False, gap=pair_gap, show_plane_outline=args.show_plane_outline,
    )

    disp_cyc, seg_cyc = rasterize_view(rig, scene, "cyclopean", include_segmentation=True)
    disp_left, _ = rasterize_view(rig, scene, "left", include_segmentation=False)
    boundary = segmentation_boundary(seg_cyc, width, height)
    udf = distance_transform_from_boundary(boundary, width, height)
    udf_preview_rgb = rasterize_boundary_preview_rgb(rig, scene, boundary)

    write_png(paths["cyclopean_png"], width, height, cyclopean_rgb)
    write_png(paths["left_png"], width, height, left_rgb)
    write_png(paths["right_png"], width, height, right_rgb)
    write_png(paths["crossed_png"], crossed_width, crossed_height, crossed_rgb)
    write_png(paths["parallel_png"], parallel_width, parallel_height, parallel_rgb)
    write_npy(paths["disp"], disp_cyc, (height, width), "<f4", "f")
    write_npy(paths["disp_left"], disp_left, (height, width), "<f4", "f")
    write_npy(paths["seg"], seg_cyc, (height, width), "|u1", "B")
    write_npy(paths["field"], udf, (height, width, 1), "<f4", "f")
    write_boundary_preview_svg(paths["udf"], rig, scene, boundary)
    write_png(paths["udf_png"], width, height, udf_preview_rgb)
    write_metadata(paths["metadata"], rig, scene, spec)

    if not args.no_summary:
        panels = build_summary_panels(out_root, stem)
        write_summary_svg(
            paths["summary"],
            panels,
            input_dir=out_root,
            stem=stem,
            summary_title=summary_title,
            columns=args.summary_columns,
        )

    print(f"Generated scene              '{spec.key}' (stem '{stem}')")
    print(f"Saved outputs in             '{out_root}'")
    print(f"  cyclopean SVG              '{os.path.basename(paths['cyclopean'])}'")
    print(f"  physical left SVG          '{os.path.basename(paths['left'])}'")
    print(f"  physical right SVG         '{os.path.basename(paths['right'])}'")
    print(f"  crossed-fusion pair        '{os.path.basename(paths['crossed'])}'")
    print(f"  parallel-order pair        '{os.path.basename(paths['parallel'])}'")
    print(f"  cyclopean PNG preview      '{os.path.basename(paths['cyclopean_png'])}'")
    print(f"  physical left PNG preview  '{os.path.basename(paths['left_png'])}'")
    print(f"  physical right PNG preview '{os.path.basename(paths['right_png'])}'")
    print(f"  crossed-pair PNG preview   '{os.path.basename(paths['crossed_png'])}'")
    print(f"  parallel-pair PNG preview  '{os.path.basename(paths['parallel_png'])}'")
    print(f"  cyclopean disparity        '{os.path.basename(paths['disp'])}'")
    print(f"  left disparity             '{os.path.basename(paths['disp_left'])}'")
    print(f"  segmentation map           '{os.path.basename(paths['seg'])}'")
    print(f"  UDF field                  '{os.path.basename(paths['field'])}'")
    print(f"  UDF boundary preview       '{os.path.basename(paths['udf'])}'")
    print(f"  UDF preview PNG            '{os.path.basename(paths['udf_png'])}'")
    print(f"  metadata                   '{os.path.basename(paths['metadata'])}'")
    if not args.no_summary:
        print(f"  all-outputs summary        '{os.path.basename(paths['summary'])}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a full stereo-output suite for a named perceptual-science 3D scene."
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help=f"Scene text string to generate. Available scenes: {available_scene_names()}.",
    )
    parser.add_argument(
        "--scene",
        dest="scene_option",
        default=None,
        help="Scene text string to generate. Overrides the positional scene argument.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Scene output folder. Defaults to <scene-prefix>_outputs.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem. Values without the scene prefix are prefixed automatically.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--focal-px", type=float, default=300.0)
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--plane-z", type=float, default=1.40)
    parser.add_argument(
        "--gillam-lines-mode",
        choices=["forest", "planar"],
        default="planar",
        help=(
            "Gillam 2002 upper-region line layout: 'forest' (original -- each of the seven "
            "line segments fronto-parallel at its own pseudorandom depth from --line-zs) or "
            "'planar' (default -- all seven lines coplanar on a single slanted plane, behind "
            "and tilted relative to the dotted plane; see --plane-slant-*)."
        ),
    )
    parser.add_argument(
        "--line-zs",
        type=parse_line_zs,
        default=parse_line_zs("2.25,2.05,2.55,1.95,2.35,2.70,2.15"),
        help="Forest mode only. Seven comma-separated line depths. Each must be greater than --plane-z.",
    )
    parser.add_argument(
        "--plane-slant-tilt-x-deg",
        type=float,
        default=20.0,
        help=(
            "Planar mode only. Tilt of the forest plane about the horizontal axis, in degrees. "
            "Positive tips the far (top) edge away from the camera, so the plane recedes as it "
            "goes up the image."
        ),
    )
    parser.add_argument(
        "--plane-slant-tilt-y-deg",
        type=float,
        default=0.0,
        help=(
            "Planar mode only. Tilt of the forest plane about the vertical axis, in degrees. "
            "Positive tips the right edge away from the camera."
        ),
    )
    parser.add_argument(
        "--plane-slant-ref-z",
        type=float,
        default=1.95,
        help=(
            "Planar mode only. Depth at the nearest point along the forest plane's near "
            "(boundary) edge, i.e. how close that edge comes to the dotted plane. Must be "
            "greater than --plane-z."
        ),
    )
    parser.add_argument("--plane-top-frac", type=float, default=0.385)
    parser.add_argument("--dot-radius-frac", type=float, default=0.0105)
    parser.add_argument("--line-stroke-frac", type=float, default=0.0060)
    parser.add_argument(
        "--figure2-d1-px",
        type=float,
        default=None,
        help="Belhumeur Figure 2 foreground disc disparity d1 in pixels. Defaults to 7%% of the smaller image dimension.",
    )
    parser.add_argument(
        "--figure2-dimg-px",
        type=float,
        default=None,
        help="Belhumeur Figure 2 rendered disc diameter D_img in pixels. Defaults to 20%% of the smaller image dimension.",
    )
    parser.add_argument(
        "--figure2-w",
        type=float,
        default=0.12,
        help="Belhumeur Figure 2 relative monocular crescent width w, expressed as a fraction of D_img.",
    )
    parser.add_argument(
        "--figure2-gray",
        type=int,
        default=96,
        help="0-255 grayscale value for the Belhumeur Figure 2 far-plane grey surface.",
    )
    parser.add_argument(
        "--wallpaper-surround-luminance",
        choices=sorted(WALLPAPER_SURROUND_LUMINANCE_PRESETS),
        default="dark",
        help=(
            "Anderson & Nakayama 1994 Figure 6: surround (background) luminance preset. "
            "Approximates the paper's 4 background conditions, which shift the perceptual bias "
            "from predominantly-behind (dark) through bistable (mid_dark/mid_light) to "
            "predominantly-front (light); stripe luminances stay fixed regardless of this "
            "setting. Both the front and behind ground-truth interpretations are always "
            "generated, independent of this choice."
        ),
    )
    parser.add_argument(
        "--wallpaper-stripe-light-gray",
        type=int,
        default=235,
        help="0-255 grayscale value for the wallpaper pattern's light stripes.",
    )
    parser.add_argument(
        "--wallpaper-stripe-dark-gray",
        type=int,
        default=40,
        help="0-255 grayscale value for the wallpaper pattern's dark stripes.",
    )
    parser.add_argument(
        "--wallpaper-stripe-width-frac",
        type=float,
        default=0.040,
        help=(
            "Wallpaper stripe width as a fraction of min(image height, width). This also fixes "
            "the interocular shift (one stripe width = one half cycle), which is what makes the "
            "front/behind interpretations equally valid -- it is not independently configurable."
        ),
    )
    parser.add_argument(
        "--wallpaper-stripe-count",
        type=int,
        default=10,
        help="Number of stripes in the wallpaper pattern (Figure 6 uses 10).",
    )
    parser.add_argument(
        "--wallpaper-patch-height-frac",
        type=float,
        default=0.20,
        help=(
            "Wallpaper patch height as a fraction of image height. Paired with the default "
            "stripe count and width, this gives a patch roughly 2:1 wide:tall (spanning ~40%% of "
            "the frame width), matching Figure 6."
        ),
    )
    parser.add_argument(
        "--wallpaper-surround-z",
        type=float,
        default=1.00,
        help="Reference depth for the surround; also bounds how large the stripe-width shift can be.",
    )
    parser.add_argument(
        "--wallpaper-dot-count",
        type=int,
        default=300,
        help="Number of sparse black dots scattered over the surround, as in the paper's figures.",
    )
    parser.add_argument(
        "--wallpaper-dot-radius-frac",
        type=float,
        default=0.005,
        help="Surround dot radius as a fraction of min(image height, width).",
    )
    parser.add_argument(
        "--wallpaper-dot-seed",
        type=int,
        default=0,
        help="Random seed for the surround dot layout (deterministic; same seed -> same dots).",
    )
    parser.add_argument("--pair-gap-frac", type=float, default=0.16)
    parser.add_argument("--show-plane-outline", action="store_true")
    parser.add_argument(
        "--show-boundary-preview",
        action="store_true",
        help="Draw the otherwise subjective plane top boundary in the separate SVGs for debugging.",
    )
    parser.add_argument("--summary-columns", type=int, default=3)
    parser.add_argument("--no-summary", action="store_true")

    args = parser.parse_args()
    try:
        if args.scene_option:
            spec = resolve_scene_spec(args.scene_option)
        elif args.scene:
            spec = resolve_scene_spec(args.scene)
        else:
            spec = resolve_scene_spec(DEFAULT_SCENE_KEY)
    except ValueError as exc:
        parser.error(str(exc))

    height, width = args.image_size
    if height <= 0 or width <= 0:
        raise ValueError("--image-size must be positive")
    if args.summary_columns <= 0:
        raise ValueError("--summary-columns must be positive")
    if args.figure2_w <= 0.0:
        raise ValueError("--figure2-w must be positive")

    scale = float(min(height, width))
    if args.figure2_dimg_px is None:
        args.figure2_dimg_px = 0.20 * scale
    if args.figure2_d1_px is None:
        args.figure2_d1_px = max(18.0, 0.07 * scale)
    if args.figure2_dimg_px <= 0.0:
        raise ValueError("--figure2-dimg-px must be positive")
    if args.figure2_d1_px <= 0.0:
        raise ValueError("--figure2-d1-px must be positive")
    if args.figure2_d1_px <= 2.0 * args.figure2_w * args.figure2_dimg_px:
        raise ValueError("--figure2-d1-px must satisfy d1 > 2*w*D_img")

    rig = CameraRig(width=width, height=height, focal_px=args.focal_px, baseline=args.baseline)
    out_root = args.output_dir or spec.default_output_dir(args)
    stem = normalize_stem(args.stem, spec.effective_prefix(args))

    if spec.variant_build_scenes:
        for variant_key, build_fn in spec.variant_build_scenes.items():
            scene = build_fn(rig, args)
            variant_stem = f"{stem}_{variant_key}"
            variant_title = f"{spec.summary_title} ({variant_key})"
            generate_scene_outputs(rig, scene, spec, args, out_root, variant_stem, variant_title)
    else:
        scene = spec.build_scene(rig, args)
        generate_scene_outputs(rig, scene, spec, args, out_root, stem, spec.summary_title)


if __name__ == "__main__":
    main()
