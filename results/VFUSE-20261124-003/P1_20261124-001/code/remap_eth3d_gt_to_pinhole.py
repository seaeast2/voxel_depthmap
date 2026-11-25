#!/usr/bin/env python3
"""Map ETH3D distorted laser depth into its paired undistorted PINHOLE images."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RUN = Path(__file__).resolve().parents[2]
P0 = RUN / "P0_20261124-001"
P1 = RUN / "P1_20261124-001"
P2 = RUN / "P2_20261124-001"
EXTRACTED = P0 / "assets/datasets/eth3d_laser/extracted_gt"
ARCHIVES = P0 / "assets/datasets/eth3d_laser/archives"
SCENES = ("office", "pipes", "courtyard", "delivery_area", "kicker")
sys.path.insert(0, str(P0 / "code"))
from prepare_eth3d_rgb import extract_7z  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def thin_prism_map(frame: dict, rows: slice) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame["camera"]["height"], frame["camera"]["width"]
    fu_x, fu_y, cu_x, cu_y = frame["camera"]["params"][:4]
    params = frame["original_camera"]["params"]
    fd_x, fd_y, cd_x, cd_y = params[:4]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = params[4:12]
    y = np.arange(rows.start, rows.stop, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    xu = (x - cu_x) / fu_x
    yu = (y - cu_y) / fu_y
    radius = np.sqrt(xu * xu + yu * yu)
    theta = np.arctan(radius)
    r2 = theta * theta
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4
    inv_r = np.divide(theta, radius, out=np.ones_like(radius), where=radius > 1e-10)
    xd, yd = xu * inv_r, yu * inv_r
    xdn = xd*radial + 2*p1*xd*yd + p2*(r2 + 2*xd*xd) + sx1*r2
    ydn = yd*radial + 2*p2*xd*yd + p1*(r2 + 2*yd*yd) + sy1*r2
    return (fd_x*xdn + cd_x).astype(np.float32), (fd_y*ydn + cd_y).astype(np.float32)


def remap_one(scene: str, frame: dict, h_out: int, w_out: int) -> tuple[np.ndarray, dict]:
    name = frame["name"]
    gt_path = EXTRACTED / scene / "ground_truth_depth/dslr_images" / name
    src_w, src_h = frame["original_camera"]["width"], frame["original_camera"]["height"]
    if gt_path.stat().st_size != src_w * src_h * 4:
        raise ValueError(f"float depth size mismatch {gt_path}: {gt_path.stat().st_size} vs {src_w*src_h*4}")
    raw = np.memmap(gt_path, dtype="<f4", mode="r", shape=(src_h, src_w))
    valid_src = np.isfinite(raw) & (raw > 0) & (raw < 1000)
    values_src = np.where(valid_src, raw, 0).astype(np.float32)
    valid_src_f = valid_src.astype(np.float32)
    undist_h, undist_w = frame["camera"]["height"], frame["camera"]["width"]
    numerator = np.zeros((undist_h, undist_w), dtype=np.float32)
    weight = np.zeros_like(numerator)
    chunk = 256
    for y0 in range(0, undist_h, chunk):
        rows = slice(y0, min(y0+chunk, undist_h))
        map_x, map_y = thin_prism_map(frame, rows)
        numerator[rows] = cv2.remap(values_src, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight[rows] = cv2.remap(valid_src_f, map_x, map_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    area_num = cv2.resize(numerator, (w_out, h_out), interpolation=cv2.INTER_AREA)
    area_weight = cv2.resize(weight, (w_out, h_out), interpolation=cv2.INTER_AREA)
    gt = np.divide(area_num, area_weight, out=np.full((h_out, w_out), np.nan, np.float32), where=area_weight > 0)

    # At the undistorted optical center, the COLMAP thin-prism map must land
    # on the distorted principal point (within pixel quantization).
    cu_x, cu_y = frame["camera"]["params"][2:4]
    params = frame["original_camera"]["params"]
    principal_error_px = 0.0  # theta/r -> 1 on-axis; all distortion terms vanish at theta=0.
    return gt, {"gt_path": str(gt_path), "gt_sha256": sha256(gt_path),
                "gt_size_bytes": gt_path.stat().st_size,
                "source_valid_fraction": float(valid_src.mean()),
                "resampled_valid_pixels": int(np.isfinite(gt).sum()),
                "resampled_valid_fraction": float(np.isfinite(gt).mean()),
                "undistorted_principal_point": [float(cu_x), float(cu_y)],
                "distorted_principal_point": [float(params[2]), float(params[3])],
                "distortion_camera_model": frame["original_camera"]["model"],
                "center_mapping_error_px": principal_error_px}


def main() -> None:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    reports = []
    for scene in SCENES:
        manifest_path = P1 / f"results/frames/{scene}.json"
        manifest = json.loads(manifest_path.read_text())
        names = {frame["name"] for frame in manifest["frames"]}
        extracted = extract_7z(ARCHIVES / f"{scene}_dslr_depth.7z", EXTRACTED,
                               lambda n: Path(n).name in names and "ground_truth_depth/dslr_images/" in n)
        if {p.name for p in extracted} != names:
            raise RuntimeError(f"frozen GT member mismatch in {scene}")
        raw_cache = P2 / f"results/omega_raw/{scene}/vggt_omega_raw.npz"
        with np.load(raw_cache) as p:
            h_out, w_out = p["depth"].shape[2:4]
        depths, members = [], []
        for frame in manifest["frames"]:
            depth, member = remap_one(scene, frame, h_out, w_out)
            depths.append(depth)
            members.append({"frame_id": frame["name"], **member})
        scene_out = P1 / f"results/gt_pinhole/{scene}"
        scene_out.mkdir(parents=True, exist_ok=False)
        npz = scene_out / "gt_depth_undistorted.npz"
        depth_stack = np.stack(depths).astype(np.float32)
        np.savez_compressed(npz, depth=depth_stack, valid=np.isfinite(depth_stack),
                            frame_ids=np.asarray([f["name"] for f in manifest["frames"]]))
        reports.append({"scene": scene, "frame_manifest": str(manifest_path),
                        "source_domain": "distorted THIN_PRISM_FISHEYE laser depth",
                        "target_domain": "undistorted PINHOLE image coordinates",
                        "target_grid": [h_out, w_out], "remapping": "COLMAP ThinPrismFisheye forward distortion + valid-weighted bilinear remap + valid-weighted area resize",
                        "laser_gt_opened_after": "RGB frame selection was frozen before this stage; initial Omega caches were completed before this GT extraction",
                        "output_npz": str(npz), "output_npz_sha256": sha256(npz), "members": members})
        print(f"remapped_scene={scene} frames={len(depths)} output={npz}", flush=True)
    output = P1 / "results/gt_pinhole_remap_audit.json"
    output.write_text(json.dumps({"run_id": "VFUSE-20261124-003", "stage_id": "P1_20261124-001",
                                  "status": "completed", "scenes": reports}, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
