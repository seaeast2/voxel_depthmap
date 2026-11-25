#!/usr/bin/env python3
"""One frozen five-view VGGT-Omega inference per ETH3D scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[2]
P0 = RUN / "P0_20261124-001"
P1 = RUN / "P1_20261124-001"
STAGE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["office", "pipes", "courtyard", "delivery_area", "kicker"], required=True)
    parser.add_argument("--resolution", type=int, choices=[384, 512], default=512)
    args = parser.parse_args()
    import numpy as np
    import torch
    import sys
    sys.path.insert(0, str(P0 / "assets/source/vggt-omega"))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    manifest = P1 / f"results/frames/{args.scene}.json"
    checkpoint = P0 / "assets/checkpoints/vggt_omega_1b_512.pt"
    output = STAGE / f"results/omega_raw/{args.scene}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    frames = json.loads(manifest.read_text())["frames"]
    if len(frames) != 5:
        raise ValueError(f"expected frozen five-view selection, got {len(frames)}")
    output.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(0)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = {"run_id": "VFUSE-20261124-003", "stage_id": "P2_20261124-001",
               "scene": args.scene, "status": "running", "execution_context": "host GPU outside sandbox",
               "image_domain": "undistorted PINHOLE", "count": 5, "resolution": args.resolution,
               "seed": 0, "frame_manifest": str(manifest), "frame_manifest_sha256": sha256(manifest),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
               "source_commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6", "gt_read_by_inference": False}
    try:
        model = VGGTOmega().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model = model.to("cuda")
        images = load_and_preprocess_images([f["path"] for f in frames], mode="max_size",
                                            image_resolution=args.resolution).cuda()
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            predictions = model(images)
            ext, intr = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
        torch.cuda.synchronize()
        predictions.update(extrinsics=ext, intrinsics=intr)
        arrays = {k: v.detach().float().cpu().numpy() for k, v in predictions.items() if torch.is_tensor(v)}
        required = {"pose_enc", "images", "depth", "depth_conf", "extrinsics", "intrinsics"}
        if not required <= arrays.keys():
            raise ValueError(f"missing outputs: {required - arrays.keys()}")
        if arrays["images"].shape[:2] != (1, 5):
            raise ValueError(f"unexpected preprocessed image shape {arrays['images'].shape}")
        if any(not np.isfinite(x).all() for x in arrays.values()) or not (arrays["depth"] > 0).all():
            raise ValueError("nonfinite or nonpositive Omega output")
        out_npz = output / "vggt_omega_raw.npz"
        np.savez_compressed(out_npz, **arrays)
        with np.load(out_npz) as saved:
            for key, value in arrays.items():
                if not np.array_equal(saved[key], value):
                    raise ValueError(f"cache round-trip mismatch {key}")
        metrics.update(status="completed", gpu=torch.cuda.get_device_name(), python=os.sys.version.split()[0],
                       torch=torch.__version__, torch_cuda=torch.version.cuda,
                       forward_seconds=time.perf_counter()-forward_start, process_seconds=time.perf_counter()-started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       output_npz=str(out_npz), output_npz_sha256=sha256(out_npz),
                       frame_ids=[f["name"] for f in frames], shapes={k:list(v.shape) for k,v in arrays.items()})
    except Exception as exc:
        metrics.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        metrics.setdefault("process_seconds", time.perf_counter()-started)
        metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
