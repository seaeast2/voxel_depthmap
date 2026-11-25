#!/usr/bin/env python3
"""Run frozen V-FUSE for five references and C1-C4 confidence transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[2]
P0 = RUN / "P0_20261124-001"
P1 = RUN / "P1_20261124-001"
P2 = RUN / "P2_20261124-001"
STAGE = Path(__file__).resolve().parents[1]
CONDITIONS = ("c1_constant", "c2_minmax", "c3_percentile", "c4_rank")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confidence_maps(raw: np.ndarray, valid: np.ndarray) -> tuple[dict[str,np.ndarray],dict]:
    values = raw[valid].astype(np.float32)
    low, high = float(values.min()), float(values.max())
    q05, q95 = np.quantile(values, [0.05, 0.95]).tolist()
    sorted_values = np.sort(values)
    maps = {
        "c1_constant": valid.astype(np.float32),
        "c2_minmax": np.where(valid, np.maximum(np.clip((raw-low)/max(high-low, 1e-12), 0, 1), 1e-4), 0).astype(np.float32),
        "c3_percentile": np.where(valid, np.maximum(np.clip((raw-q05)/max(q95-q05, 1e-12), 0, 1), 1e-4), 0).astype(np.float32),
        "c4_rank": np.where(valid, np.maximum(np.searchsorted(sorted_values, raw, side="right")/len(sorted_values), 1e-4), 0).astype(np.float32),
    }
    return maps, {"valid_raw_min": low, "valid_raw_max": high, "q05": float(q05), "q95": float(q95),
                  "rank_definition": "scene-global empirical CDF using right insertion rank / valid pixel count",
                  "positive_confidence_floor": 1e-4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["office", "pipes", "courtyard", "delivery_area", "kicker"], required=True)
    args = parser.parse_args()
    import torch
    import yaml
    sys.path.insert(0, str(P0 / "assets/source/V-FUSE"))
    from src.models.fusion import Fusion
    from src.utils.common import warp

    raw_npz = P2 / f"results/omega_raw/{args.scene}/vggt_omega_raw.npz"
    manifest_path = P1 / f"results/frames/{args.scene}.json"
    checkpoint = P0 / "assets/checkpoints/vfuse_gbinet_V10.0.7.pt"
    out_root = STAGE / f"results/inference/{args.scene}"
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite stage outputs: {out_root}")
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["frames"]
    with np.load(raw_npz) as cache:
        depth = cache["depth"][0, ..., 0].astype(np.float32)
        raw_conf = cache["depth_conf"][0].astype(np.float32)
        ext = cache["extrinsics"][0].astype(np.float32)
        intr = cache["intrinsics"][0].astype(np.float32)
        images = cache["images"][0].astype(np.float32)
    n,h,w=depth.shape
    if n != 5 or len(frames) != 5:
        raise ValueError(f"expected five-view cache and manifest; got {n}")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(raw_conf)
    if not valid.all():
        raise ValueError("this frozen Omega cache is expected to have finite positive depth/confidence")
    conf_by_condition, conf_params = confidence_maps(raw_conf, valid)
    q05,q95=np.quantile(depth[valid],[0.05,0.95]).tolist()
    pad=0.10*(q95-q05)
    near=max(float(np.finfo(np.float32).eps),float(q05-pad)); far=float(q95+pad)
    if not far>near: raise ValueError("degenerate scene depth bounds")

    cfg=yaml.safe_load((P0/"assets/source/V-FUSE/configs/DTU/dtu.yaml").read_text())
    cfg["device"]="cuda:0"; cfg["eval"]["num_frame"]=5; cfg["eval"]["depth_planes"]=8; cfg["eval"]["scale"]=1.0
    cfg["swenet"]["depth_mode"]="conf"
    torch.manual_seed(0); torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
    model=Fusion(cfg,8).to("cuda:0")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(state["model_state_dict"],strict=True); model.eval()
    out_root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); run_records=[]
    for condition in CONDITIONS:
        for ref in range(5):
            order=[(ref+i)%5 for i in range(5)]
            result_dir=out_root/condition/f"ref_{ref:02d}"
            if result_dir.exists(): raise FileExistsError(result_dir)
            result_dir.mkdir(parents=True,exist_ok=False)
            d=depth[order]; c=conf_by_condition[condition][order]
            cams=np.zeros((5,2,4,4),np.float32)
            for j,idx in enumerate(order):
                cams[j,0,:3,:4]=ext[idx]; cams[j,0,3,3]=1
                cams[j,1,:3,:3]=intr[idx]
                cams[j,1,3]=np.array([near,0,256,far],np.float32)
            d_t=torch.from_numpy(d[:,None]); c_t=torch.from_numpy(c[:,None]); cam_t=torch.from_numpy(cams)
            warped_d,warped_c=warp(d_t,c_t,cam_t.clone())
            data={"depths":warped_d.unsqueeze(1).unsqueeze(0),
                  "orig_depths":d_t.unsqueeze(0),"confs":warped_c.unsqueeze(1).unsqueeze(0),
                  "orig_confs":c_t.unsqueeze(0),"images":torch.from_numpy(images[order]).unsqueeze(0),
                  "cams":cam_t.unsqueeze(0),"num_frame":5,"scene":args.scene,"ref_index":ref}
            data={k:(v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k,v in data.items()}
            torch.cuda.synchronize(); t0=time.perf_counter()
            with torch.inference_mode(): output=model(data)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
            fused=output["fused_depth"][0,0].float().cpu().numpy()
            fused_conf=output["fused_conf"][0,0].float().cpu().numpy()
            bounds=output["depth_bounds"].float().cpu().numpy()
            if not np.isfinite(fused).all() or not np.isfinite(fused_conf).all() or np.any(fused<=0):
                print("DEBUG", condition, ref, "depth_nonfinite", int((~np.isfinite(fused)).sum()), "depth_minmax", (float(np.nanmin(fused)), float(np.nanmax(fused))), "conf_nonfinite", int((~np.isfinite(fused_conf)).sum()), "conf_minmax", (float(np.nanmin(fused_conf)), float(np.nanmax(fused_conf))), "bounds_nonfinite", int((~np.isfinite(bounds)).sum()), flush=True); raise ValueError(f"invalid V-FUSE output {args.scene}/{condition}/ref{ref}")
            npz=result_dir/"v_fuse.npz"
            np.savez_compressed(npz,fused_depth=fused,fused_conf=fused_conf,depth_bounds=bounds)
            metrics={"run_id":"VFUSE-20261124-003","stage_id":"P3_20261124-003","scene":args.scene,
                     "reference_index":ref,"reference_frame":frames[ref]["name"],
                     "ordered_frame_ids":[frames[i]["name"] for i in order],"condition":condition,
                     "confidence_params":conf_params,"bounds_source":"Omega prediction Q05/Q95 + 10% span padding",
                     "bounds":[near,far],"checkpoint_sha256":sha256(checkpoint),"input_npz_sha256":sha256(raw_npz),
                     "gt_read_by_inference":False,"status":"completed","gpu":torch.cuda.get_device_name(),
                     "input_shapes":{"depths":list(data["depths"].shape),"cams":list(data["cams"].shape)},
                     "output_shapes":{"fused_depth":list(fused.shape),"fused_conf":list(fused_conf.shape),"depth_bounds":list(bounds.shape)},
                     "model_seconds":elapsed,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"output_npz":str(npz),"output_npz_sha256":sha256(npz)}
            (result_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
            run_records.append({"condition":condition,"reference_index":ref,"frame":frames[ref]["name"],
                                "model_seconds":elapsed,"output_npz_sha256":metrics["output_npz_sha256"]})
            del data,output
            torch.cuda.empty_cache()
            print(f"completed={args.scene}/{condition}/ref{ref:02d} seconds={elapsed:.3f}",flush=True)
    summary={"run_id":"VFUSE-20261124-003","stage_id":"P3_20261124-003","scene":args.scene,
             "status":"completed","checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
             "input_npz":str(raw_npz),"input_npz_sha256":sha256(raw_npz),"gpu":torch.cuda.get_device_name(),
             "conditions":list(CONDITIONS),"reference_count":5,"inference_count":len(run_records),
             "total_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
             "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"runs":run_records}
    (out_root/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
