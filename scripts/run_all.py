"""Run ``build_mask`` on the entire dataset and surface failures.

Processes every XML under ``data/burkeIIA长波/<weather>/``. For each sample:
  * writes an overlay to ``outputs/<folder>/<stem>.png`` (unless ``--no-overlays``);
  * records ratio, centroid error, containment and notes in a CSV;
  * if the sample is "bad" (see ``--bad-threshold``), copies its overlay into
    ``outputs/_bad/<folder>/<stem>_r<ratio>.png`` for quick human review.

A bad sample is one whose ``|ratio - 1|`` exceeds ``--bad-threshold``
(default 0.2), OR the mask is empty / extraction raised.

Usage::

    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --bad-threshold 0.15 --no-overlays
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import traceback
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample  # noqa: E402
from scripts.extract_demo import _overlay, _qa  # noqa: E402


DATA_ROOT = Path("data/burkeIIA长波")
OUT_ROOT = Path("outputs")


def _is_bad(qa: dict, threshold: float) -> bool:
    r = qa["ratio"]
    if not np.isfinite(r):
        return True
    if qa["n_mask"] == 0:
        return True
    return abs(r - 1.0) > threshold


def _process_folder(folder: Path,
                    bad_threshold: float,
                    write_overlays: bool,
                    writer: csv.writer) -> dict:
    xmls = sorted(folder.glob("*.xml"))
    overlay_dir = OUT_ROOT / folder.name
    bad_dir = OUT_ROOT / "_bad" / folder.name
    if write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    ratios: list[float] = []
    n_bad = 0
    n_error = 0
    n_empty = 0
    bad_rows: list[tuple[str, float, str]] = []

    for i, xml in enumerate(xmls):
        try:
            sample = load_sample(xml)
            result = build_mask(sample)
        except Exception as e:
            n_error += 1
            n_bad += 1
            writer.writerow([folder.name, xml.stem, "", "", "", "", "", "",
                             f"ERROR: {e}"])
            bad_rows.append((xml.stem, float("nan"), f"ERROR: {e}"))
            continue

        qa = _qa(sample, result)
        ratios.append(qa["ratio"])
        if qa["n_mask"] == 0:
            n_empty += 1
        writer.writerow([
            folder.name, xml.stem,
            qa["n_mask"], qa["pixel_num"],
            f"{qa['ratio']:.4f}",
            f"{qa['centroid_err_norm']:.4f}"
                if np.isfinite(qa["centroid_err_norm"]) else "",
            f"{qa['containment']:.4f}"
                if np.isfinite(qa["containment"]) else "",
            qa["n_components"],
            "; ".join(result.notes),
        ])

        out_path = overlay_dir / f"{xml.stem}.png"
        if write_overlays:
            _overlay(sample, result, out_path)

        if _is_bad(qa, bad_threshold):
            n_bad += 1
            # always produce the overlay so the bad-dir is complete
            if not write_overlays:
                _overlay(sample, result, out_path)
            tag = f"r{qa['ratio']:.2f}" if np.isfinite(qa["ratio"]) else "rNAN"
            shutil.copyfile(
                out_path,
                bad_dir / f"{tag}__{xml.stem}.png",
            )
            bad_rows.append(
                (xml.stem, qa["ratio"], "; ".join(result.notes))
            )

        if (i + 1) % 50 == 0:
            print(f"  [{folder.name}] {i + 1}/{len(xmls)} processed", flush=True)

    good_ratios = [r for r in ratios if np.isfinite(r)]
    std = pstdev(good_ratios) if len(good_ratios) > 1 else 0.0
    stats = dict(
        n=len(xmls),
        n_bad=n_bad,
        n_error=n_error,
        n_empty=n_empty,
        mean=mean(good_ratios) if good_ratios else float("nan"),
        median=median(good_ratios) if good_ratios else float("nan"),
        std=std,
        min=min(good_ratios) if good_ratios else float("nan"),
        max=max(good_ratios) if good_ratios else float("nan"),
        within_0_1=sum(1 for r in good_ratios if abs(r - 1) <= 0.1),
        within_0_2=sum(1 for r in good_ratios if abs(r - 1) <= 0.2),
        bad_rows=bad_rows,
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad-threshold", type=float, default=0.2,
                    help="|ratio-1| above this = bad (default 0.2)")
    ap.add_argument("--no-overlays", action="store_true",
                    help="skip writing per-sample overlays for good samples")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="subset of folder names; default = all 8")
    args = ap.parse_args()

    OUT_ROOT.mkdir(exist_ok=True)
    # Clean the bad-dir so the listing reflects only this run.
    bad_root = OUT_ROOT / "_bad"
    if bad_root.exists():
        shutil.rmtree(bad_root)
    bad_root.mkdir()

    folders = ([DATA_ROOT / n for n in args.folders]
               if args.folders else sorted(DATA_ROOT.iterdir()))
    folders = [f for f in folders if f.is_dir()]

    csv_path = OUT_ROOT / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder", "stem", "n_mask", "pixel_num", "ratio",
                         "cent_err_norm", "containment", "n_components", "notes"])
        all_stats = {}
        for folder in folders:
            print(f"\n=== {folder.name} ({len(list(folder.glob('*.xml')))} samples) ===",
                  flush=True)
            stats = _process_folder(folder, args.bad_threshold,
                                    not args.no_overlays, writer)
            all_stats[folder.name] = stats

    # Per-folder summary.
    print("\n" + "=" * 78)
    print(f"{'folder':<30} {'n':>4} {'mean':>6} {'med':>6} {'std':>6} "
          f"{'max':>5} {'±0.1':>6} {'±0.2':>6} {'bad':>4}")
    print("-" * 78)
    total_n = total_bad = total_err = 0
    for name, s in all_stats.items():
        print(f"{name:<30} {s['n']:>4} {s['mean']:>6.3f} {s['median']:>6.3f} "
              f"{s['std']:>6.3f} {s['max']:>5.2f} "
              f"{s['within_0_1']:>4}/{s['n']:<3d} "
              f"{s['within_0_2']:>4}/{s['n']:<3d} "
              f"{s['n_bad']:>4}")
        total_n += s["n"]
        total_bad += s["n_bad"]
        total_err += s["n_error"]
    print("-" * 78)
    print(f"TOTAL samples={total_n}  bad={total_bad}  errors={total_err}")
    print(f"Per-sample CSV: {csv_path}")
    print(f"Bad overlays:   {bad_root}/<folder>/")

    # Write a compact "bad list" that's easy to scan.
    bad_list_path = OUT_ROOT / "bad_list.txt"
    with bad_list_path.open("w", encoding="utf-8") as fh:
        for name, s in all_stats.items():
            if not s["bad_rows"]:
                continue
            fh.write(f"\n=== {name}  ({len(s['bad_rows'])} bad) ===\n")
            for stem, r, notes in sorted(
                s["bad_rows"],
                key=lambda x: (-abs((x[1] or 0) - 1)
                               if np.isfinite(x[1] or 0) else -1e9),
            ):
                rs = f"{r:.3f}" if np.isfinite(r) else "NAN"
                fh.write(f"  ratio={rs:>6}  {stem}  {notes}\n")
    print(f"Bad list:       {bad_list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
