"""
Run this FIRST, before we write the real loader. It only PRINTS the
structure of your downloaded SisFall and PPG-DaLiA folders (file counts,
filename patterns, column counts, dict keys) -- it does not process or
copy any actual sensor readings anywhere. Paste its output back so the
real loader can be written to match your files exactly.

Usage:
    python inspect_datasets.py --sisfall "D:\\Sentryband\\raw_data\\SisFall" --ppgdalia "D:\\Sentryband\\raw_data\\PPG-DaLiA"

(Adjust the paths to wherever you extracted the two datasets. You can
also run with just one of --sisfall / --ppgdalia if you only have one
downloaded so far.)
"""

import argparse
import pickle
from pathlib import Path


def inspect_sisfall(root: str, max_files_to_preview: int = 3):
    root = Path(root)
    print("=" * 60)
    print(f"SISFALL INSPECTION: {root}")
    print("=" * 60)

    if not root.exists():
        print(f"!! Path does not exist: {root}")
        return

    all_files = list(root.rglob("*.txt")) + list(root.rglob("*.csv"))
    print(f"Total .txt/.csv files found (recursive): {len(all_files)}")

    if not all_files:
        print("No .txt/.csv files found. Listing top-level contents instead:")
        for item in sorted(root.iterdir())[:30]:
            print(f"  {'[DIR] ' if item.is_dir() else '      '}{item.name}")
        return

    print("\nSample filenames (up to 15):")
    for f in all_files[:15]:
        print(f"  {f.relative_to(root)}")

    print(f"\nPreviewing first {max_files_to_preview} files' raw content:")
    for f in all_files[:max_files_to_preview]:
        print(f"\n--- {f.relative_to(root)} ---")
        with open(f, "r", errors="replace") as fh:
            lines = [fh.readline().strip() for _ in range(3)]
        for i, line in enumerate(lines):
            if not line:
                continue
            for delim_name, delim in [("comma", ","), ("semicolon", ";"), ("whitespace", None)]:
                parts = line.split(delim) if delim else line.split()
                if len(parts) > 1:
                    print(f"  line {i}: {len(parts)} fields via {delim_name} delimiter -> {parts[:12]}")
                    break
        with open(f, "r", errors="replace") as fh:
            n_lines = sum(1 for _ in fh)
        print(f"  total lines in file: {n_lines}")


def inspect_ppg_dalia(root: str, max_files_to_preview: int = 2):
    root = Path(root)
    print("\n" + "=" * 60)
    print(f"PPG-DALIA INSPECTION: {root}")
    print("=" * 60)

    if not root.exists():
        print(f"!! Path does not exist: {root}")
        return

    pkl_files = list(root.rglob("*.pkl"))
    print(f"Total .pkl files found (recursive): {len(pkl_files)}")

    if not pkl_files:
        print("No .pkl files found. Listing top-level contents instead:")
        for item in sorted(root.iterdir())[:30]:
            print(f"  {'[DIR] ' if item.is_dir() else '      '}{item.name}")
        return

    print("\nSample filenames (up to 15):")
    for f in pkl_files[:15]:
        print(f"  {f.relative_to(root)}")

    for f in pkl_files[:max_files_to_preview]:
        print(f"\n--- Structure of {f.relative_to(root)} ---")
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh, encoding="latin1")
        except Exception as e:
            print(f"  !! Failed to load: {e}")
            continue

        def describe(obj, prefix="", depth=0, max_depth=3):
            if depth > max_depth:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    print(f"  {prefix}['{k}'] -> {type(v).__name__}", end="")
                    if hasattr(v, "shape"):
                        print(f", shape={v.shape}, dtype={getattr(v, 'dtype', '?')}")
                    elif isinstance(v, (list, tuple)):
                        print(f", len={len(v)}")
                    else:
                        print()
                    if isinstance(v, dict):
                        describe(v, prefix=prefix + f"['{k}']", depth=depth + 1, max_depth=max_depth)
            else:
                print(f"  {prefix} -> {type(obj).__name__}")

        describe(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sisfall", type=str, default=None)
    parser.add_argument("--ppgdalia", type=str, default=None)
    args = parser.parse_args()

    if args.sisfall:
        inspect_sisfall(args.sisfall)
    if args.ppgdalia:
        inspect_ppg_dalia(args.ppgdalia)

    if not args.sisfall and not args.ppgdalia:
        print("Pass at least one of --sisfall or --ppgdalia with a folder path.")