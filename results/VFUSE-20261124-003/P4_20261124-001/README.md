# P4 oracle-camera diagnostic

V-FUSE C3-percentile inference uses ETH3D undistorted PINHOLE K and COLMAP world-to-camera poses. The scene scale is the median `GT_ref0 / Omega_ref0` over valid reference-0 pixels; this scale also converts GT camera translations into Omega depth units. Reference 0 is the only GT depth read for this fitting step. Held-out GT refs 1–4 are not read by inference. This is an oracle-camera diagnostic, not a deployable method. Per-scene scale and inference are in `results/inference/<scene>/c3_gtcamera/run_summary.json`.
