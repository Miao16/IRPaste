"""Pre-extract all target masks and save to disk cache.

Usage::

    uv run python scripts/pre_extract.py \
      --targets-root data/burkeIIA长波 \
      --cache-dir outputs/_cache \
      --resume

Produces:
    {cache_dir}/*.npz          — one per target (patch + mask + metadata)
    {cache_dir}/manifest.csv   — index: stem, view, on_horizon, cache_file
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample, classify_target  # noqa: E402
from irpaste.paste import detect_target_on_horizon, target_patch_from_sample  # noqa: E402


def _stem_hash(stem: Path) -> str:
    """Short deterministic hash of the stem path for cache filenames."""
    return hashlib.sha256(str(stem).encode()).hexdigest()[:16]


def _extract_one(stem: Path) -> dict | None:
    """Extract mask + patch for one target. Returns dict or None on failure."""
    try:
        sample = load_sample(stem)
        res = build_mask(sample)
        if res.n_mask < 40:
            return None
        patch, mask, _ = target_patch_from_sample(sample, res.mask)
        if mask.sum() < 40:
            return None
        on_horizon, sim_hr = detect_target_on_horizon(sample, res.mask)
        return dict(
            patch=patch,
            mask=mask,
            on_horizon=on_horizon,
            sim_horizon_row=sim_hr if sim_hr is not None else -1,
        )
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-root", default="data/burkeIIA长波")
    ap.add_argument("--cache-dir", default="outputs/_cache")
    ap.add_argument("--resume", action="store_true", help="skip already-cached targets")
    args = ap.parse_args()

    root = Path(args.targets_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all target stems.
    stems: list[tuple[Path, str]] = []  # (stem, view)
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            view = classify_target(xml_path)
        except Exception:
            continue
        stem = xml_path.with_suffix("")
        stems.append((stem, view))

    if not stems:
        print("No targets found.")
        return 1

    manifest_path = cache_dir / "manifest.csv"
    existing = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open("r") as fh:
            for row in csv.reader(fh):
                if row:
                    existing.add(row[0])

    t0 = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0

    with manifest_path.open("a" if args.resume else "w", newline="") as mf:
        writer = csv.writer(mf)
        if not args.resume:
            writer.writerow(["stem", "view", "on_horizon", "sim_horizon_row", "cache_file"])

        for stem, view in tqdm(stems, desc="Extracting masks", unit="target"):
            stem_str = str(stem)
            if args.resume and stem_str in existing:
                n_skip += 1
                continue

            data = _extract_one(stem)
            if data is None:
                n_fail += 1
                continue

            fname = f"{_stem_hash(stem)}.npz"
            npz_path = cache_dir / fname
            np.savez_compressed(
                npz_path,
                patch=data["patch"],
                mask=data["mask"],
                view=view,
                on_horizon=data["on_horizon"],
                sim_horizon_row=np.float32(data["sim_horizon_row"]),
                stem=stem_str,
            )
            writer.writerow([
                stem_str, view,
                int(data["on_horizon"]),
                data["sim_horizon_row"],
                fname,
            ])
            mf.flush()
            n_ok += 1

    dt = time.time() - t0
    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed in {dt:.1f}s")
    print(f"Cache: {cache_dir}/  ({n_ok} .npz files)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
