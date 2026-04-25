"""Bulk paste: generate N view-matched composites with diverse targets.

Two-phase workflow:
  1. Pre-extract masks:  uv run python scripts/pre_extract.py --targets-root ... --cache-dir ...
  2. Bulk paste:         uv run python scripts/paste_bulk.py --cache-dir ... --bg-root ... --n 512 --seed 7

Usage::

    uv run python scripts/paste_bulk.py --n 512 --seed 7 --cache-dir outputs/_cache

Outputs:
    outputs/_bulk/{idx:06d}_{view}_{bg}_{target}_n{N}.png  — composite
    outputs/_bulk/clean/                                    — clean composite
    outputs/_bulk/vis/                                      — annotated with bbox + horizon
    outputs/_bulk/labels/                                   — YOLO HBB labels
    outputs/_bulk/_contact_{k:02d}.png                      — contact sheets
    outputs/_bulk/_manifest.csv                             — row per composite
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste.paste import (  # noqa: E402
    paste_patch,
    load_background,
    augment_background,
    _feather_alpha,
)
from irpaste.viewcls import classify_background  # noqa: E402


_ORANGE = (0, 160, 255)
_CYAN = (255, 255, 0)


def _index_bgs(root: Path):
    """Index backgrounds by view type, caching bg_view to avoid re-classification."""
    side_paths, top_paths = [], []
    side_views, top_views = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        try:
            bg = load_background(p)
            v = classify_background(bg, return_info=True, filename=p.name)
        except Exception:
            continue
        if v.kind == "side":
            side_paths.append(p)
            side_views.append(v)
        else:
            top_paths.append(p)
            top_views.append(v)
    return (side_paths, side_views), (top_paths, top_views)


def _load_manifest(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load manifest.csv, return (side_entries, top_entries)."""
    side, top = [], []
    manifest = cache_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run pre_extract.py first")
    with manifest.open("r") as fh:
        for row in csv.DictReader(fh):
            row["on_horizon"] = row["on_horizon"] == "1"
            row["sim_horizon_row"] = float(row["sim_horizon_row"])
            entry = dict(row)
            (side if row["view"] == "side" else top).append(entry)
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


def _safe_stem(name: str, n: int = 16) -> str:
    s = "".join(c for c in name if c.isascii() and (c.isalnum() or c in "._-"))
    return (s or "x")[:n]


# ---------------------------------------------------------------------------
# Annotation helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _annotate_multi(
    comp_or_bgr: np.ndarray,
    results: list,
    tags: list[str],
    crop_x: int = 0,
    crop_y: int = 0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    if comp_or_bgr.ndim == 2:
        bgr = cv2.cvtColor(comp_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = comp_or_bgr.copy()
    for pr, tag in zip(results, tags):
        ys, xs = np.where(pr.mask_patch)
        if ys.size > 0:
            x = pr.paste_xy[0] + int(xs.min())
            y = pr.paste_xy[1] + int(ys.min())
            pw = int(xs.max()) - int(xs.min())
            ph = int(ys.max()) - int(ys.min())
        else:
            x, y = pr.paste_xy
            ph, pw = pr.mask_patch.shape
        x = int(round((x - crop_x) * scale_x))
        y = int(round((y - crop_y) * scale_y))
        pw = max(1, int(round(pw * scale_x)))
        ph = max(1, int(round(ph * scale_y)))
        cv2.rectangle(bgr, (x, y), (x + pw, y + ph), _ORANGE, 1)
        cv2.putText(
            bgr, tag, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            bgr, tag, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 0), 1, cv2.LINE_AA,
        )
    if results and results[0].bg_view.kind == "side":
        bv = results[0].bg_view
        if bv.horizon_curve is not None:
            pts = bv.horizon_curve.polyline(n=max(32, bgr.shape[1] // 8))
            adj = pts.astype(np.float64)
            adj[:, 0] = (adj[:, 0] - crop_x) * scale_x
            adj[:, 1] = (adj[:, 1] - crop_y) * scale_y
            cv2.polylines(bgr, [adj.round().astype(np.int32)], False, _CYAN, 1, cv2.LINE_AA)
        elif bv.horizon_row is not None:
            hr = int(round((bv.horizon_row - crop_y) * scale_y))
            cv2.line(bgr, (0, hr), (bgr.shape[1] - 1, hr), _CYAN, 1, cv2.LINE_AA)
    view_label = f"{'Side' if results[0].bg_view.kind == 'side' else 'Top'} view  n={len(results)}" if results else f"n={len(results)}"
    cv2.putText(
        bgr, view_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        bgr, view_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return bgr


def _compute_label(
    pr, crop_x: int, crop_y: int, out_w: int, out_h: int
) -> tuple[float, float, float, float]:
    ys, xs = np.where(pr.mask_patch)
    if ys.size == 0:
        mk_x0 = 0
        mk_y0 = 0
        pw = pr.mask_patch.shape[1]
        ph = pr.mask_patch.shape[0]
    else:
        mk_x0 = int(xs.min())
        mk_y0 = int(ys.min())
        pw = int(xs.max()) - mk_x0 + 1
        ph = int(ys.max()) - mk_y0 + 1
    px = pr.paste_xy[0] + mk_x0
    py = pr.paste_xy[1] + mk_y0
    ox0 = px - crop_x
    oy0 = py - crop_y
    ox1 = ox0 + pw
    oy1 = oy0 + ph
    ox0_c = max(0, ox0)
    oy0_c = max(0, oy0)
    ox1_c = min(out_w, ox1)
    oy1_c = min(out_h, oy1)
    if ox1_c <= ox0_c or oy1_c <= oy0_c:
        return 0.5, 0.5, 0.0, 0.0
    pw_c = ox1_c - ox0_c
    ph_c = oy1_c - oy0_c
    cx_n = (ox0_c + ox1_c) / 2.0 / out_w
    cy_n = (oy0_c + oy1_c) / 2.0 / out_h
    w_n = pw_c / out_w
    h_n = ph_c / out_h
    return (
        float(np.clip(cx_n, 0.0, 1.0)),
        float(np.clip(cy_n, 0.0, 1.0)),
        float(np.clip(w_n, 0.0, 1.0)),
        float(np.clip(h_n, 0.0, 1.0)),
    )


def _write_multi_label(
    label_path: Path, labels: list[tuple[float, float, float, float]]
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cx, cy, w, h in labels:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _build_contact_sheet(
    paths: list[Path], cols: int = 8, tile: int = 160
) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="outputs/_cache",
                    help="directory with .npz cache + manifest.csv from pre_extract.py")
    ap.add_argument("--bg-root", default="data/background/test_1")
    ap.add_argument("--out", default="outputs/_bulk")
    ap.add_argument("--n", type=int, default=None,
                    help="composites to generate (omit to process every cached target once)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--side-frac", type=float, default=0.80,
                    help="fraction of composites that should be side-view")
    ap.add_argument("--contact-rows", type=int, default=8)
    ap.add_argument("--contact-cols", type=int, default=8)
    ap.add_argument("--augment-bg", action="store_true",
                    help="randomly zoom-in and smart-crop background before pasting")
    ap.add_argument("--bg-scale-max", type=float, default=1.3,
                    help="upper bound of background zoom-in factor")
    ap.add_argument("--align-axis", action="store_true",
                    help="rotate ship principal axis parallel to bg horizon")
    ap.add_argument("--ship-scale-min", type=float, default=0.55)
    ap.add_argument("--ship-scale-max", type=float, default=0.90)
    ap.add_argument("--max-ships-per-bg", type=int, default=1,
                    help="max ships per background (1-5)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_clean = out / "clean"
    out_vis = out / "vis"
    out_labels = out / "labels"
    out_clean.mkdir(exist_ok=True)
    out_vis.mkdir(exist_ok=True)
    out_labels.mkdir(exist_ok=True)

    seed = (
        args.seed
        if args.seed is not None
        else int(np.random.default_rng().integers(0, 2**31))
    )
    print(f"seed: {seed}  (pass --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed)

    t0 = time.time()

    # --- Load manifest ---
    print("loading manifest ...")
    tgt_side, tgt_top = _load_manifest(Path(args.cache_dir))
    print(f"  targets: {len(tgt_side)} side, {len(tgt_top)} top")

    # --- Index backgrounds (with view pre-computed) ---
    print("indexing backgrounds ...")
    (bg_side_paths, bg_side_views), (bg_top_paths, bg_top_views) = _index_bgs(Path(args.bg_root))
    print(f"  bg: {len(bg_side_paths)} side, {len(bg_top_paths)} top")
    print(f"  index took {time.time() - t0:.1f}s")

    # --- Build work queue ---
    if args.n is None:
        _all = tgt_side + tgt_top
        _arr = np.arange(len(_all))
        rng.shuffle(_arr)
        _work_queue = [_all[int(i)] for i in _arr]
        n_total = len(_work_queue)
        s_side = s_top = None
        print(f"  exhaustive mode: {n_total} targets queued")
    else:
        _work_queue = None
        n_total = args.n
        s_side = _Shuffler(tgt_side, np.random.default_rng(seed + 1))
        s_top = _Shuffler(tgt_top, np.random.default_rng(seed + 2))

    cache_dir = Path(args.cache_dir)
    manifest_path = out / "_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["idx", "view", "bg", "target", "on_horizon", "paste_x", "paste_y", "out"]
        )

        produced = 0
        attempts = 0
        written: list[Path] = []
        per_tile = max(args.contact_rows * args.contact_cols, 16)
        _q_iter = iter(_work_queue) if _work_queue is not None else None
        _TARGET = 512
        max_per_bg = max(1, min(args.max_ships_per_bg, 5))

        pbar = tqdm(total=n_total, desc="Pasting", unit="comp")
        while produced < n_total:
            # --- Build batch: 1 to max_per_bg ships sharing one bg ---
            n_ships_this_batch = int(rng.integers(1, max_per_bg + 1))
            batch_entries: list[dict] = []
            batch_kind = None
            batch_bg_idx = None

            for _ in range(n_ships_this_batch):
                if _q_iter is not None:
                    try:
                        entry = next(_q_iter)
                    except StopIteration:
                        break
                    kind = entry["view"]
                    bg_pool_paths = bg_side_paths if kind == "side" else bg_top_paths
                    bg_pool_views = bg_side_views if kind == "side" else bg_top_views
                    if not bg_pool_paths:
                        continue
                    bg_idx = int(rng.integers(len(bg_pool_paths)))
                else:
                    attempts += 1
                    if attempts > n_total * 4:
                        break
                    is_side = rng.random() < args.side_frac
                    if is_side and (not bg_side_paths or not tgt_side):
                        is_side = False
                    if (not is_side) and (not bg_top_paths or not tgt_top):
                        is_side = True
                    kind = "side" if is_side else "top"
                    bg_pool_paths = bg_side_paths if is_side else bg_top_paths
                    bg_pool_views = bg_side_views if is_side else bg_top_views
                    entry = (s_side if is_side else s_top).next()
                    if entry is None:
                        continue
                    bg_idx = int(rng.integers(len(bg_pool_paths)))

                if batch_bg_idx is None:
                    batch_bg_idx = bg_idx
                    batch_kind = kind
                batch_entries.append(entry)

                if _q_iter is None and attempts > n_total * 4:
                    break

            if not batch_entries:
                break

            # --- Load bg ONCE, augment ONCE, classify ONCE per batch ---
            bg_path = (bg_side_paths if batch_kind == "side" else bg_top_paths)[batch_bg_idx]
            bg_view_orig = (bg_side_views if batch_kind == "side" else bg_top_views)[batch_bg_idx]
            try:
                bg = load_background(bg_path)
            except Exception as e:
                print(f"  skip batch (bg) {bg_path.name}: {e}")
                continue

            if args.augment_bg:
                bg = augment_background(bg, rng, scale_range=(1.0, args.bg_scale_max))
                bg_view = classify_background(bg, return_info=True, filename=bg_path.name)
            else:
                bg_view = bg_view_orig

            # --- Paste ships sequentially on the shared bg ---
            results: list = []
            occupied_mask = np.zeros(bg.shape, dtype=bool)

            for i, entry in enumerate(batch_entries):
                # Load cached patch + mask.
                npz_path = cache_dir / entry["cache_file"]
                try:
                    data = np.load(str(npz_path))
                    patch = data["patch"]
                    mask = data["mask"]
                    on_horizon = bool(data["on_horizon"])
                except Exception as e:
                    print(f"  skip {entry['stem']}: {e}")
                    continue

                try:
                    pr = paste_patch(
                        patch,
                        mask,
                        bg,
                        bg_view=bg_view,
                        target_on_horizon=on_horizon,
                        rng=np.random.default_rng(seed + 1000 + produced + i),
                        align_to_horizon=args.align_axis,
                        ship_scale_range=(args.ship_scale_min, args.ship_scale_max),
                        occupied_mask=occupied_mask if i > 0 else None,
                    )
                except Exception as e:
                    print(f"  skip {entry['stem']}: {e}")
                    continue

                # --- Update composite ---
                if i == 0:
                    composite_full = pr.composite.copy()
                else:
                    x, y = pr.paste_xy
                    ph, pw = pr.mask_patch.shape
                    H_c, W_c = composite_full.shape
                    x1 = min(x + pw, W_c)
                    y1 = min(y + ph, H_c)
                    ry, rx = y1 - y, x1 - x
                    if ry > 0 and rx > 0:
                        m_roi = pr.mask_patch[:ry, :rx]
                        a = _feather_alpha(m_roi, dilate=0, sigma=0.8) * 0.9
                        roi_new = pr.composite[y:y1, x:x1].astype(np.float32)
                        roi_old = composite_full[y:y1, x:x1].astype(np.float32)
                        blended = roi_new * a + roi_old * (1.0 - a)
                        composite_full[y:y1, x:x1] = np.clip(blended, 0, 255).astype(np.uint8)

                # --- Update occupied_mask (dilate 2px for breathing room) ---
                x, y = pr.paste_xy
                ph, pw = pr.mask_patch.shape
                H_c, W_c = bg.shape
                x1 = min(x + pw, W_c)
                y1 = min(y + ph, H_c)
                ry, rx = y1 - y, x1 - x
                if ry > 0 and rx > 0:
                    m_occ = pr.mask_patch[:ry, :rx].copy()
                    m_occ_u8 = m_occ.astype(np.uint8)
                    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    m_occ_u8 = cv2.dilate(m_occ_u8, k_dil)
                    fm = np.zeros(bg.shape, dtype=bool)
                    fm[y:y1, x:x1] = m_occ_u8.astype(bool)
                    occupied_mask = occupied_mask | fm

                results.append(pr)

            if not results:
                continue

            # --- Output naming ---
            batch_idx = produced
            first_stem = batch_entries[0]["stem"]
            out_name = (
                f"{batch_idx:06d}_{batch_kind}"
                f"_{_safe_stem(bg_path.stem)}"
                f"_{_safe_stem(Path(first_stem).name)}"
                f"_n{len(results)}.png"
            )

            # --- Crop/resize to 512x512 ---
            H_full, W_full = composite_full.shape
            clean_full = cv2.cvtColor(composite_full, cv2.COLOR_GRAY2BGR)
            tags = [
                f"ship{j+1}{' H' if pr.target_on_horizon else ''}"
                for j, pr in enumerate(results)
            ]
            if H_full >= _TARGET and W_full >= _TARGET:
                _y0 = (H_full - _TARGET) // 2
                _x0 = (W_full - _TARGET) // 2
                clean_out = clean_full[_y0 : _y0 + _TARGET, _x0 : _x0 + _TARGET]
                vis_out = _annotate_multi(clean_out, results, tags, crop_x=_x0, crop_y=_y0)
            else:
                _y0, _x0 = 0, 0
                scale_x = _TARGET / max(W_full, 1)
                scale_y = _TARGET / max(H_full, 1)
                clean_out = cv2.resize(
                    clean_full, (_TARGET, _TARGET), interpolation=cv2.INTER_LINEAR
                )
                vis_out = _annotate_multi(clean_out, results, tags, scale_x=scale_x, scale_y=scale_y)

            cv2.imwrite(str(out_clean / out_name), clean_out)
            out_vis_path = out_vis / out_name
            cv2.imwrite(str(out_vis_path), vis_out)
            written.append(out_vis_path)

            # --- YOLO HBB labels ---
            out_w = clean_out.shape[1]
            out_h = clean_out.shape[0]
            label_lines: list[tuple[float, float, float, float]] = []
            for pr in results:
                cx_n, cy_n, w_n, h_n = _compute_label(pr, _x0, _y0, out_w, out_h)
                label_lines.append((cx_n, cy_n, w_n, h_n))
            lbl_path = out_labels / Path(out_name).with_suffix(".txt").name
            _write_multi_label(lbl_path, label_lines)

            # --- Manifest row ---
            target_names = ";".join(Path(e["stem"]).name for e in batch_entries)
            w.writerow([
                batch_idx,
                batch_kind,
                bg_path.name,
                target_names,
                int(results[0].target_on_horizon),
                results[0].paste_xy[0],
                results[0].paste_xy[1],
                f"clean/{out_name}",
            ])

            produced += 1
            pbar.update(1)
            pbar.set_postfix(rate=f"{produced / max(time.time() - t0, 1e-3):.1f}/s")

        pbar.close()

    # --- Contact sheets ---
    print("building contact sheets ...")
    for k, start in enumerate(range(0, len(written), per_tile)):
        sheet = _build_contact_sheet(
            written[start : start + per_tile],
            cols=args.contact_cols,
            tile=160,
        )
        cv2.imwrite(str(out / f"_contact_{k:02d}.png"), sheet)

    print(
        f"wrote {produced} composites -> {out_clean} (clean)  {out_vis} (vis)   "
        f"(total {time.time() - t0:.1f}s,   "
        f"manifest: {manifest_path.name})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
