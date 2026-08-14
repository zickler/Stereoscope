#!/usr/bin/env python3
"""
Scan `*_outputs` directories for generated stereo scenes and emit viewer/scenes.json.

Run this after generating or regenerating any scene with generate_perceptual_stereo.py so the
web viewer (viewer/index.html) picks up new or changed scenes -- it fetches scenes.json at
runtime rather than hardcoding a scene list, so adding a scene never requires touching viewer
code, only regenerating this manifest (and redeploying/copying the new output directory
wherever the viewer is hosted).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator, List, Optional, Tuple


def find_stems(repo_root: str) -> Iterator[Tuple[str, str, str]]:
    for entry in sorted(os.listdir(repo_root)):
        dir_path = os.path.join(repo_root, entry)
        if not os.path.isdir(dir_path) or not entry.endswith("_outputs"):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith("_meta.json"):
                continue
            stem = fname[: -len("_meta.json")]
            left = os.path.join(dir_path, f"{stem}_left.png")
            right = os.path.join(dir_path, f"{stem}_right.png")
            if os.path.exists(left) and os.path.exists(right):
                yield dir_path, stem, fname


def build_manifest(repo_root: str, base_url: Optional[str]) -> List[dict]:
    entries = []
    for dir_path, stem, meta_fname in find_stems(repo_root):
        with open(os.path.join(dir_path, meta_fname), encoding="utf-8") as f:
            meta = json.load(f)
        camera = meta["camera"]
        height, width = meta["image_size"]
        cx, cy = camera["principal_point_px"]
        figure = meta.get("figure", {})
        dir_name = os.path.basename(dir_path)

        def asset_url(suffix: str) -> str:
            rel = f"{dir_name}/{stem}{suffix}"
            if base_url:
                return f"{base_url.rstrip('/')}/{rel}"
            return f"../{rel}"

        entries.append(
            {
                "id": stem,
                "title": stem.replace("_", " "),
                "source": figure.get("source", ""),
                "thumbnail": asset_url("_cyclopean.png"),
                "leftImage": asset_url("_left.png"),
                "rightImage": asset_url("_right.png"),
                "focalPx": camera["focal_px"],
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to scan for *_outputs directories (default: this script's directory).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the manifest (default: <repo-root>/viewer/scenes.json).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "If set, emit absolute asset URLs under this base instead of paths relative to "
            "viewer/index.html -- use this when the *_outputs directories are hosted "
            "separately from the viewer (e.g. a different bucket/CDN than GitHub Pages)."
        ),
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.repo_root, "viewer", "scenes.json")
    entries = build_manifest(args.repo_root, args.base_url)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(entries)} scene(s) to {out_path}")


if __name__ == "__main__":
    main()
