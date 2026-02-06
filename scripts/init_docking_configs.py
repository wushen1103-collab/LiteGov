#!/usr/bin/env python
import os
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    rec_dir = root / "docking" / "receptors"

    grids = {
        "1M17": {
            "center_x": 8.9,
            "center_y": 22.4,
            "center_z": 57.3,
            "size_x": 20,
            "size_y": 20,
            "size_z": 20,
        },
        "CDK2": {
            "center_x": 16.2,
            "center_y": 29.1,
            "center_z": 34.8,
            "size_x": 22,
            "size_y": 22,
            "size_z": 22,
        },
        "HIVPR": {
            "center_x": 10.4,
            "center_y": 23.2,
            "center_z": 15.5,
            "size_x": 22,
            "size_y": 22,
            "size_z": 22,
        },
        "6LU7": {
            "center_x": -11.7,
            "center_y": 13.6,
            "center_z": 69.7,
            "size_x": 26,
            "size_y": 26,
            "size_z": 26,
        },
        "3EML": {
            "center_x": 37.2,
            "center_y": 52.1,
            "center_z": -1.9,
            "size_x": 22,
            "size_y": 22,
            "size_z": 22,
        },
        "THR": {
            "center_x": 38.6,
            "center_y": 35.5,
            "center_z": 29.2,
            "size_x": 22,
            "size_y": 22,
            "size_z": 22,
        },
        "ERa": {
            "center_x": 23.3,
            "center_y": 12.8,
            "center_z": 27.0,
            "size_x": 24,
            "size_y": 24,
            "size_z": 24,
        },
    }

    for name, params in grids.items():
        target_dir = rec_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = target_dir / "config.txt"

        lines = [
            f"center_x = {params['center_x']}",
            f"center_y = {params['center_y']}",
            f"center_z = {params['center_z']}",
            f"size_x = {params['size_x']}",
            f"size_y = {params['size_y']}",
            f"size_z = {params['size_z']}",
            "exhaustiveness = 8",
            "num_modes = 9",
        ]
        cfg_text = "\n".join(lines) + "\n"

        with open(cfg_path, "w") as f:
            f.write(cfg_text)

        print(f"[+] wrote {cfg_path}")

if __name__ == "__main__":
    main()
