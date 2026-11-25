# P2 Omega frozen predictions

Five same-camera ETH3D views per scene were frozen by P1 before GT was opened. VGGT-Omega inference completed once per scene at `max_size=512` (512×336 output). The run used the pinned Omega checkpoint and produced five NPZ caches with metadata/hashes in `results/omega_raw/<scene>/metrics.json`. GPU: host NVIDIA GeForce RTX 5060; peak allocated memory: 5,156,147,712 bytes. Inference did not read laser GT.
