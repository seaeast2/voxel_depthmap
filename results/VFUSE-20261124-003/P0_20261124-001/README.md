# P0 assets and provenance

- VGGT-Omega source snapshot: commit `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`.
- V-FUSE source snapshot: commit `88156ea8ee823e62ce7f122c693c8b7b84278d05`.
- V-FUSE pretrained GBiNet fusion checkpoint and Omega checkpoint were copied into this run; hashes are in `results/` manifests.
- ETH3D high-res multi-view training scene archives include existing office/pipes and added courtyard/delivery_area/kicker. Archive byte sizes and SHA-256s are in `results/additional_eth3d_assets.json` and related manifests.
- Official V-FUSE data preparation says baseline MVS data is not yet provided and describes expected GBiNet depth/confidence/camera/image layout. We do not have packaged author GBiNet predictions for an in-domain control, so this control remains pending. Public ETH3D RGB/calibration/laser depth is not itself a GBiNet MVS prediction set.
- Host GPU inference uses the prior run's venv path for Python 3.11.18 / torch 2.15.0+cu132; record this reuse when recreating the environment.
