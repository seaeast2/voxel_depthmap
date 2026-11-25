#!/usr/bin/env python3
"""Freeze five same-camera views per ETH3D scene using RGB-only COLMAP tracks."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parents[1]
P0 = RUN / "P0_20261124-001"
ASSETS = P0 / "assets/datasets/eth3d_laser"
EXTRACTED = ASSETS / "extracted"
ARCHIVES = ASSETS / "archives"
OUT = STAGE / "results/frames"
SCENES = ("office", "pipes", "courtyard", "delivery_area", "kicker")
sys.path.insert(0, str(P0 / "code"))
from prepare_eth3d_rgb import extract_7z, parse_calibration, sha256  # noqa: E402


def scene_manifest(scene: str) -> dict:
    root = EXTRACTED / scene
    undist_dir = root / "dslr_calibration_undistorted"
    distorted_dir = root / "dslr_calibration_jpg"
    cameras, images = parse_calibration(undist_dir)
    raw_cameras, raw_images = parse_calibration(distorted_dir)
    names = sorted(images)
    feature_tracks = [images[name]["track_ids"] for name in names]
    centers = []
    for name in names:
        w2c = np.asarray(images[name]["world_to_camera"], dtype=np.float64)
        centers.append(-np.linalg.solve(w2c[:, :3], w2c[:, 3]))
    n = len(names)
    overlap = np.zeros((n, n), dtype=np.int32)
    baseline = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            overlap[i, j] = overlap[j, i] = len(feature_tracks[i] & feature_tracks[j])
            baseline[i, j] = baseline[j, i] = np.linalg.norm(centers[i] - centers[j])

    proposals = []
    for inds in itertools.combinations(range(n), 5):
        if len({images[names[i]]["camera_id"] for i in inds}) != 1:
            continue
        pairs = [(inds[a], inds[b]) for a in range(5) for b in range(a + 1, 5)]
        pair_overlap = [int(overlap[i, j]) for i, j in pairs]
        pair_baselines = [float(baseline[i, j]) for i, j in pairs]
        if min(pair_baselines) < 1e-4:
            continue
        proposals.append((min(pair_overlap), sum(pair_overlap), tuple(names[i] for i in inds),
                          inds, pair_overlap, pair_baselines))
    if not proposals:
        raise RuntimeError(f"no eligible same-camera five-view group: {scene}")
    proposals.sort(key=lambda x: (-x[0], -x[1], x[2]))
    min_shared, total_shared, selected_names, inds, pair_counts, baselines = proposals[0]
    if min_shared < 50:
        raise RuntimeError(f"minimum pair shared tracks <50 for {scene}: {min_shared}")

    frames = []
    for frame_index, name in enumerate(selected_names):
        meta = images[name]
        raw = raw_images[name]
        camera = cameras[meta["camera_id"]]
        original_camera = raw_cameras[raw["camera_id"]]
        if camera["model"] != "PINHOLE" or original_camera["model"] != "THIN_PRISM_FISHEYE":
            raise RuntimeError(f"unexpected camera model for {scene}/{name}")
        pose_delta = np.max(np.abs(np.asarray(meta["world_to_camera"]) - np.asarray(raw["world_to_camera"])))
        if pose_delta >= 1e-4:
            raise RuntimeError(f"distorted/undistorted camera pose mismatch {scene}/{name}: {pose_delta}")
        image_path = root / "images" / meta["name"]
        with Image.open(image_path) as im:
            if im.size != (camera["width"], camera["height"]):
                raise RuntimeError(f"image/camera dimension mismatch {scene}/{name}: {im.size}")
            if im.getexif().get(274, 1) != 1:
                raise RuntimeError(f"unexpected EXIF rotation {scene}/{name}")
        frames.append({
            "frame_index": frame_index, "name": name, "path": str(image_path),
            "size_bytes": image_path.stat().st_size, "sha256": sha256(image_path),
            "original_wh": list(Image.open(image_path).size), "camera": camera,
            "original_camera": original_camera, "world_to_camera": meta["world_to_camera"],
            "image_id": meta["image_id"], "camera_id": meta["camera_id"],
            "distorted_camera_id": raw["camera_id"], "pose_pair_max_abs_delta": float(pose_delta),
        })
    return {"run_id": "VFUSE-20261124-003", "stage_id": "P1_20261124-001", "scene": scene,
            "selection_rule": "five same-camera views maximizing minimum pairwise RGB-only shared COLMAP tracks, then total; lexical tie-break",
            "frame_count": len(frames), "all_image_count": n, "laser_gt_opened": False,
            "min_pair_shared_tracks": min_shared, "total_pair_shared_tracks": total_shared,
            "pair_shared_track_counts": pair_counts, "pair_baseline_colmap_units": baselines,
            "frames": frames}


def main() -> None:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        print(f"extracting_undistorted_rgb_calibration={scene}", flush=True)
        extract_7z(ARCHIVES / f"{scene}_dslr_undistorted.7z", EXTRACTED)
        extract_7z(ARCHIVES / f"{scene}_dslr_jpg.7z", EXTRACTED, lambda n: n.endswith(".txt"))
        frozen = scene_manifest(scene)
        out = OUT / f"{scene}.json"
        out.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        contact = Image.new("RGB", (1800, 330), "white")
        draw = ImageDraw.Draw(contact)
        for i, frame in enumerate(frozen["frames"]):
            with Image.open(frame["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((360, 300))
                contact.paste(image, (i * 360, 28))
            draw.text((i * 360 + 5, 7), frame["name"], fill="black")
        contact.save(OUT / f"{scene}_selected_rgb.png")
        print(json.dumps({k: v for k, v in frozen.items() if k != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
