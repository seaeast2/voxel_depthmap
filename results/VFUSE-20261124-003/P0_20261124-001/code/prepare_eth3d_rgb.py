#!/usr/bin/env python3
"""Extract RGB/calibration only and freeze three RGB-selected views per ETH3D scene.

Laser-depth archives are intentionally not opened by this preparation step.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


STAGE = Path(__file__).resolve().parents[1]
ASSETS = STAGE / "assets/datasets/eth3d_laser"
ARCHIVES = ASSETS / "archives"
EXTRACTED = ASSETS / "extracted"
OUTPUT = STAGE / "results/inputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_7z(archive: Path, destination: Path, accept=lambda name: True) -> list[Path]:
    lib = ct.CDLL("libarchive.so.13")
    ptr = ct.c_void_p
    signatures = {
        "archive_read_new": ([], ptr),
        "archive_read_support_filter_all": ([ptr], ct.c_int),
        "archive_read_support_format_7zip": ([ptr], ct.c_int),
        "archive_read_open_filename": ([ptr, ct.c_char_p, ct.c_size_t], ct.c_int),
        "archive_read_next_header": ([ptr, ct.POINTER(ptr)], ct.c_int),
        "archive_entry_pathname": ([ptr], ct.c_char_p),
        "archive_entry_size": ([ptr], ct.c_longlong),
        "archive_read_data": ([ptr, ptr, ct.c_size_t], ct.c_longlong),
        "archive_read_data_skip": ([ptr], ct.c_int),
        "archive_read_free": ([ptr], ct.c_int),
        "archive_error_string": ([ptr], ct.c_char_p),
    }
    for name, (args, result) in signatures.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = result
    destination.mkdir(parents=True, exist_ok=True)
    archive_reader = lib.archive_read_new()
    lib.archive_read_support_filter_all(archive_reader)
    lib.archive_read_support_format_7zip(archive_reader)
    status = lib.archive_read_open_filename(archive_reader, str(archive).encode(), 1024 * 1024)
    if status != 0:
        raise RuntimeError(f"libarchive open error: {lib.archive_error_string(archive_reader)}")

    entry = ptr()
    buffer = ct.create_string_buffer(1024 * 1024)
    written: list[Path] = []
    try:
        while True:
            status = lib.archive_read_next_header(archive_reader, ct.byref(entry))
            if status == 1:
                break
            if status < 0:
                raise RuntimeError(f"libarchive read error: {lib.archive_error_string(archive_reader)}")
            member = lib.archive_entry_pathname(entry).decode()
            size = lib.archive_entry_size(entry)
            target = (destination / member).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"archive member escapes destination: {member}")
            if not accept(member) or member.endswith("/") or size == 0:
                lib.archive_read_data_skip(archive_reader)
                continue
            if target.exists() and target.stat().st_size == size:
                written.append(target)
                lib.archive_read_data_skip(archive_reader)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                while True:
                    count = lib.archive_read_data(archive_reader, buffer, len(buffer))
                    if count == 0:
                        break
                    if count < 0:
                        raise RuntimeError(f"libarchive extract error: {lib.archive_error_string(archive_reader)}")
                    output.write(buffer.raw[:count])
            if target.stat().st_size != size:
                raise RuntimeError(f"extracted size mismatch for {member}: expected={size}")
            written.append(target)
    finally:
        lib.archive_read_free(archive_reader)
    return written


def parse_calibration(directory: Path) -> tuple[dict, dict]:
    cameras = {}
    for line in (directory / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        row = line.split()
        cameras[int(row[0])] = {
            "model": row[1], "width": int(row[2]), "height": int(row[3]),
            "params": list(map(float, row[4:])),
        }
    lines = (directory / "images.txt").read_text(encoding="utf-8").splitlines()
    images = {}
    i = 0
    while i < len(lines):
        if not lines[i].strip() or lines[i].startswith("#"):
            i += 1
            continue
        row = lines[i].split()
        if len(row) != 10:
            raise ValueError(f"unexpected COLMAP image row at {directory}/images.txt:{i+1}")
        q = np.asarray(list(map(float, row[1:5])))
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        translation = np.asarray(list(map(float, row[5:8])))
        point_row = lines[i + 1].split() if i + 1 < len(lines) else []
        tracks = {int(v) for v in point_row[2::3] if int(v) >= 0}
        name = Path(row[9]).name
        images[name] = {
            "image_id": int(row[0]), "name": row[9], "camera_id": int(row[8]),
            "world_to_camera": np.c_[rotation, translation].tolist(),
            "track_ids": tracks,
        }
        i += 2
    return cameras, images


def freeze_scene(scene: str) -> dict:
    root = EXTRACTED / scene
    rgb_camera_dir = root / "dslr_calibration_undistorted"
    raw_camera_dir = root / "dslr_calibration_jpg"
    cameras, images = parse_calibration(rgb_camera_dir)
    raw_cameras, raw_images = parse_calibration(raw_camera_dir)
    proposals = []
    names = sorted(images)
    for triple in itertools.combinations(names, 3):
        current = [images[name] for name in triple]
        if len({item["camera_id"] for item in current}) != 1:
            continue
        centers = [
            -np.linalg.solve(np.asarray(item["world_to_camera"])[:, :3],
                             np.asarray(item["world_to_camera"])[:, 3])
            for item in current
        ]
        if min(np.linalg.norm(a - b) for a, b in itertools.combinations(centers, 2)) < 1e-4:
            continue
        shared = [len(a["track_ids"] & b["track_ids"]) for a, b in itertools.combinations(current, 2)]
        proposals.append((min(shared), sum(shared), triple, shared))
    if not proposals:
        raise RuntimeError(f"no valid same-camera triple for {scene}")
    minimum_shared, total_shared, triple, shared_counts = sorted(
        proposals, key=lambda item: (-item[0], -item[1], item[2])
    )[0]
    if minimum_shared < 50:
        raise RuntimeError(f"RGB-only triple support below frozen minimum: {scene} {shared_counts}")

    frames = []
    for frame_index, name in enumerate(triple):
        rgb_meta = images[name]
        raw_meta = raw_images[name]
        camera = cameras[rgb_meta["camera_id"]]
        original_camera = raw_cameras[raw_meta["camera_id"]]
        if camera["model"] != "PINHOLE" or original_camera["model"] != "THIN_PRISM_FISHEYE":
            raise RuntimeError(f"unexpected camera model for {scene}/{name}")
        if np.max(np.abs(np.asarray(rgb_meta["world_to_camera"]) - np.asarray(raw_meta["world_to_camera"]))) >= 1e-4:
            raise RuntimeError(f"original/undistorted camera pose mismatch: {scene}/{name}")
        image_path = root / "images" / rgb_meta["name"]
        with Image.open(image_path) as image:
            if image.size != (camera["width"], camera["height"]):
                raise RuntimeError(f"image/camera dimensions mismatch: {image_path} {image.size}")
            if image.getexif().get(274, 1) != 1:
                raise RuntimeError(f"unexpected EXIF rotation: {image_path}")
            original_wh = list(image.size)
        frames.append({
            "frame_index": frame_index,
            "source_index_zero_based": names.index(name),
            "name": name,
            "path": str(image_path),
            "size_bytes": image_path.stat().st_size,
            "sha256": sha256(image_path),
            "original_wh": original_wh,
            "camera": camera,
            "original_camera": original_camera,
            "world_to_camera": rgb_meta["world_to_camera"],
            "image_id": rgb_meta["image_id"],
        })
    return {
        "scene": scene,
        "selection_rule": "maximize minimum pair shared COLMAP RGB track count; then total; lexical tie break",
        "laser_gt_opened": False,
        "frames": frames,
        "pair_shared_track_counts": shared_counts,
        "minimum_pair_shared_tracks": minimum_shared,
        "total_pair_shared_tracks": total_shared,
        "all_image_count": len(names),
    }


def main() -> None:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    for scene in ("office", "pipes"):
        rgb_archive = ARCHIVES / f"{scene}_dslr_undistorted.7z"
        jpg_archive = ARCHIVES / f"{scene}_dslr_jpg.7z"
        print(f"extracting_rgb={scene}", flush=True)
        extract_7z(rgb_archive, EXTRACTED)
        print(f"extracting_calibration={scene}", flush=True)
        extract_7z(jpg_archive, EXTRACTED, lambda name: name.endswith(".txt"))
        result = freeze_scene(scene)
        out_dir = OUTPUT / f"eth3d-{scene}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frames.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        contact = Image.new("RGB", (1200, 300), "white")
        draw = ImageDraw.Draw(contact)
        for i, frame in enumerate(result["frames"]):
            with Image.open(frame["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((400, 270))
                contact.paste(image, (i * 400, 30))
            draw.text((i * 400 + 5, 8), frame["name"], fill="black")
        contact.save(out_dir / "selected_rgb.png")
        print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2), flush=True)
    print(f"frames_json_root={OUTPUT}")


if __name__ == "__main__":
    main()
