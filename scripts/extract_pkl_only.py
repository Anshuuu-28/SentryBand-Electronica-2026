"""
Selectively extracts ONLY the S*.pkl files from PPG-DaLiA's data.zip --
skipping each subject's S*_E4.zip, S*_RespiBAN.h5, S*_activity.csv,
S*_quest.csv (which src/real_data_loader.py never reads anyway). This
needs far less free disk space than a full Expand-Archive, since those
skipped files make up most of the 2.7 GB.

Usage:
    python extract_pkl_only.py --zip "D:\\Sentryband\\raw_data\\PPG-DaLiA\\data.zip" --out "D:\\Sentryband\\raw_data\\PPG-DaLiA\\PPG_FieldStudy"
"""

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = zf.namelist()
        pkl_names = [n for n in names if n.lower().endswith(".pkl")]
        print(f"Found {len(names)} total entries in the zip; "
              f"{len(pkl_names)} of them are .pkl files.")
        if not pkl_names:
            print("No .pkl entries found -- printing first 20 entry names "
                  "so we can see the real internal structure:")
            for n in names[:20]:
                print(f"  {n}")
            return

        total_bytes = sum(zf.getinfo(n).file_size for n in pkl_names)
        print(f"Extracting {len(pkl_names)} .pkl files "
              f"(~{total_bytes / (1024**2):.1f} MB total) to {out_dir} ...")

        for i, n in enumerate(pkl_names, 1):
            # Flatten into out_dir/<SubjectFolder>/<file>.pkl regardless of
            # the zip's internal nesting, so the loader's rglob("S*.pkl")
            # finds them no matter what.
            target = out_dir / Path(n).name.replace(".pkl", "") / Path(n).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  [{i}/{len(pkl_names)}] {n} -> {target}")

    print("\nDone. Point PPGDALIA_DIR at:", out_dir)


if __name__ == "__main__":
    main()