"""
Storage-friendly PPG-DaLiA extractor.

PPG-DaLiA's S*.pkl files are huge (each ~1.3 GB) because they bundle the
full high-frequency CHEST sensor recording (ECG, EDA, EMG, respiration,
temperature @ 700 Hz) alongside the small WRIST signal we actually use
(BVP @ 64 Hz + accel @ 32 Hz) plus the HR ground-truth label. This script
never writes the full .pkl to disk -- it streams each entry straight out
of data.zip into memory, keeps only the wrist signal + label, and saves
that as a small compressed .npz per subject (typically a few MB, not
1.3 GB).

Usage:
    python extract_wrist_compact.py --zip "raw_data\\PPG-DaLiA\\data.zip" --out "raw_data\\PPG-DaLiA\\PPG_wrist_only" --max-subjects 6
"""

import argparse
import pickle
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-subjects", type=int, default=None,
                         help="Only process this many subjects (saves time; "
                              "5-6 is plenty for a real held-out validation set).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        pkl_names = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
        if args.max_subjects:
            pkl_names = pkl_names[:args.max_subjects]

        print(f"Processing {len(pkl_names)} subject .pkl files one at a "
              f"time (each briefly held in memory, ~1.3 GB peak, never "
              f"written to disk in full)...")

        for i, name in enumerate(pkl_names, 1):
            subject = Path(name).stem  # e.g. "S1"
            print(f"  [{i}/{len(pkl_names)}] {name} -- loading from zip stream...")
            with zf.open(name) as fh:
                data = pickle.load(fh, encoding="latin1")

            bvp = np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1)
            acc = np.asarray(data["signal"]["wrist"]["ACC"])
            label = np.asarray(data.get("label", []))

            out_path = out_dir / f"{subject}.npz"
            np.savez_compressed(out_path, bvp=bvp, acc=acc, label=label)
            size_mb = out_path.stat().st_size / (1024 ** 2)
            print(f"      -> {out_path} ({size_mb:.2f} MB, vs ~1.3 GB "
                  f"for the full .pkl we never wrote to disk)")
            del data  # free the big chest-signal dict before the next subject

    print(f"\nDone. Point PPGDALIA_DIR at: {out_dir}")
    print("(This folder now contains small .npz files, not .pkl -- the "
          "updated real_data_loader.py reads both formats automatically.)")


if __name__ == "__main__":
    main()