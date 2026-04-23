"""Bulk paste: generate N view-matched composites with diverse targets.

Targets are shuffled and consumed without replacement until the pool
is exhausted (then re-shuffled), so the first 2368 side composites
and 592 top composites each use a unique target. Backgrounds are
picked uniformly at random from the matching view class.

Uses the current default blend: ``laplacian`` + ``tv_smooth=True``.

Usage::

    uv run python scripts/paste_bulk.py --n 512 --seed 7

Outputs:
    outputs/_bulk/{idx:04d}_{view}_{bg}_{target}.png   — composite only
    outputs/_bulk/_contact_{k:02d}.png                 — 8×8 mosaic index
    outputs/_bulk/_manifest.csv                        — row per composite
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample                              # noqa: E402
from irpaste.paste import paste_target, load_background                   # noqa: E402
from irpaste.viewcls import classify_background, classify_target          # noqa: E402


_ORANGE = (0, 160, 255)
_CYAN   = (255, 255, 0)


def _index_bgs(root: Path):
    side, top = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        try:
            bg = load_background(p)
            v = classify_background(bg, return_info=True)
        except Exception:
            continue
        (side if v.kind == "side" else top).append(p)
    return side, top


def _index_targets(root: Path):
    side, top = [], []
    for xml in root.rglob("*.xml"):
        try:
            k = classify_target(xml)
        except Exception:
            continue
        (side if k == "side" else top).append(xml.with_suffix(""))
    return side, top


class _Shuffler:
    """Draw items without replacement; reshuffle when pool is exhausted."""
    def __init__(self, items, rng: np.random.Generator):
        self.items = list(items)
        self.rng = rng
        self._order: list[int] = []
        self._reshuffle()

    def _reshuffle(self):
        self._order = list(range(len(self.items)))
        self.rng.shuffle(self._order)

    def next(self):
        if not self.items:
            return None
        if not self._order:
            self._reshuffle()
        return self.items[self._order.pop()]


def _annotate(comp: np.ndarray, pr, tag: str) -> np.ndarray:
    bgr = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
    x, y = pr.paste_xy
    ph, pw = pr.mask_patch.shape
    cv2.rectangle(bgr, (x, y), (x + pw, y + ph), _ORANGE, 1)
    if pr.bg_view.kind == "side":
        if pr.bg_view.horizon_curve is not None:
            pts = pr.bg_view.horizon_curve.polyline(n=max(32, bgr.shape[1] // 8))
            cv2.polylines(bgr, [pts], False, _CYAN, 1, cv2.LINE_AA)
        elif pr.bg_view.horizon_row is not None:
            hr = int(pr.bg_view.horizon_row)
            cv2.line(bgr, (0, hr), (bgr.shape[1] - 1, hr), _CYAN, 1, cv2.LINE_AA)
    cv2.putText(bgr, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return bgr


def _safe_stem(name: str, n: int = 16) -> str:
    s = "".join(c for c in name if c.isascii() and (c.isalnum() or c in "._-"))
    return (s or "x")[:n]


def _build_contact_sheet(paths: list[Path], cols: int = 8, tile: int = 160) -> np.ndarray:
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
        oy, ox = (tile - im.shape[0]) // 2, (tile - im.shape[1]) // 2
        canvas[oy : oy + im.shape[0], ox : ox + im.shape[1]] = im
        tiles.append(canvas)
    if not tiles:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i : i + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-root", default="data/burkeIIA长波")
    ap.add_argument("--bg-root", default="data/background/test_1")
    ap.add_argument("--out", default="outputs/_bulk")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--side-frac", type=float, default=0.80,
                    help="fraction of composites that should be side-view")
    ap.add_argument("--no-annotate", action="store_true",
                    help="write plain composites without overlay bbox/horizon")
    ap.add_argument("--contact-rows", type=int, default=8,
                    help="rows per contact sheet")
    ap.add_argument("--contact-cols", type=int, default=8,
                    help="cols per contact sheet")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print("indexing backgrounds ...")
    bg_side, bg_top = _index_bgs(Path(args.bg_root))
    print(f"  bg: {len(bg_side)} side, {len(bg_top)} top")
    print("indexing targets ...")
    tgt_side, tgt_top = _index_targets(Path(args.targets_root))
    print(f"  targets: {len(tgt_side)} side, {len(tgt_top)} top")
    print(f"  index took {time.time() - t0:.1f}s")

    s_side = _Shuffler(tgt_side, np.random.default_rng(args.seed + 1))
    s_top  = _Shuffler(tgt_top,  np.random.default_rng(args.seed + 2))

    manifest_path = out / "_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "view", "bg", "target", "on_horizon", "paste_x", "paste_y", "out"])

        produced = 0
        attempts = 0
        written: list[Path] = []
        per_tile = max(args.contact_rows * args.contact_cols, 16)

        t_gen = time.time()
        while produced < args.n and attempts < args.n * 4:
            attempts += 1
            is_side = rng.random() < args.side_frac
            if is_side and (not bg_side or not tgt_side):
                is_side = False
            if (not is_side) and (not bg_top or not tgt_top):
                is_side = True
            kind = "side" if is_side else "top"

            tgt_stem = (s_side if is_side else s_top).next()
            bg_pool = bg_side if is_side else bg_top
            bg_path = bg_pool[int(rng.integers(len(bg_pool)))]

            try:
                sample = load_sample(tgt_stem)
                res = build_mask(sample)
                if res.n_mask < 40:
                    continue
                bg = load_background(bg_path)
                pr = paste_target(
                    sample, res.mask, bg,
                    rng=np.random.default_rng(args.seed + 1000 + produced),
                )
            except Exception as e:
                print(f"  skip {tgt_stem.name}: {e}")
                continue

            if args.no_annotate:
                img = cv2.cvtColor(pr.composite, cv2.COLOR_GRAY2BGR)
            else:
                tag = f"{kind}{' H' if pr.target_on_horizon else ''}"
                img = _annotate(pr.composite, pr, tag)

            out_name = (
                f"{produced:04d}_{kind}_{_safe_stem(bg_path.stem)}"
                f"_{_safe_stem(Path(tgt_stem).name)}.png"
            )
            out_path = out / out_name
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

            w.writerow([
                produced, kind, bg_path.name, Path(tgt_stem).name,
                int(pr.target_on_horizon), pr.paste_xy[0], pr.paste_xy[1],
                out_name,
            ])

            produced += 1
            if produced % 64 == 0:
                dt = time.time() - t_gen
                rate = produced / max(dt, 1e-3)
                eta = (args.n - produced) / max(rate, 1e-3)
                print(f"  {produced}/{args.n}   {rate:.1f}/s   ETA {eta:.0f}s")

    # Contact sheets (groups of contact_rows × contact_cols).
    print("building contact sheets ...")
    for k, start in enumerate(range(0, len(written), per_tile)):
        sheet = _build_contact_sheet(
            written[start : start + per_tile],
            cols=args.contact_cols,
            tile=160,
        )
        cv2.imwrite(str(out / f"_contact_{k:02d}.png"), sheet)

    print(
        f"wrote {produced} composites → {out}   "
        f"(total {time.time() - t0:.1f}s,   "
        f"manifest: {manifest_path.name})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
