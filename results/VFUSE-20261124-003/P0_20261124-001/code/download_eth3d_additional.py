#!/usr/bin/env python3
"""Download official ETH3D office/pipes reuse plus three new RGB/laser scenes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
TARGET = STAGE / "assets/datasets/eth3d_laser/archives"
FILES = [(scene, f"{scene}_dslr_{kind}.7z")
         for scene in ("courtyard", "delivery_area", "kicker")
         for kind in ("undistorted", "jpg", "depth")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(item: tuple[str, str]) -> dict:
    scene, name = item
    final = TARGET / name
    partial = TARGET / f".{name}.part"
    if final.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {final}")
    command = ["curl", "-sS", "--fail", "--location", "--continue-at", "-",
               "--retry", "3", "--retry-delay", "2", "--connect-timeout", "20",
               "--max-time", "3600", "--output", str(partial),
               f"https://eth3d.ethz.ch/data/{name}"]
    print(f"downloading={name}", flush=True)
    subprocess.run(command, check=True)
    result = {"scene": scene, "name": name, "size_bytes": partial.stat().st_size,
              "sha256": sha256(partial), "url": f"https://eth3d.ethz.ch/data/{name}"}
    os.replace(partial, final)
    print(f"verified_download={json.dumps(result)}", flush=True)
    return result


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, item) for item in FILES]
        records = [future.result() for future in as_completed(futures)]
    output = STAGE / "results/additional_eth3d_assets.json"
    output.write_text(json.dumps({"run_id": "VFUSE-20261124-003", "stage_id": "P0_20261124-001",
                                  "scenes": sorted(records, key=lambda x: (x["scene"], x["name"]))},
                                 indent=2) + "\n", encoding="utf-8")
    print(f"manifest={output}", flush=True)


if __name__ == "__main__":
    main()
