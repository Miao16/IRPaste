"""Compare the current paste methods (alpha / laplacian) against their
TV-polished variants on the *same* (target, background, paste-site)
pairs.

Seam-quality metric
-------------------
For each composite we build a 1-px ring along the mask contour and
compute mean ``|∇composite|`` on that ring (Sobel magnitude). A clean
blend matches the surrounding bg texture → lower ring gradient is
better. We also report the mean ring gradient of the *raw* background
at the same location as an "ideal" reference.

Usage::

    uv run python scripts/compare_tv.py --n 6 --seed 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample                          # noqa: E402
from irpaste.paste import paste_target, load_background              # noqa: E402
from irpaste.viewcls import classify_background, classify_target     # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _boundary_ring(full_mask: np.ndarray, px: int = 1) -> np.ndarray:
    m = full_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


def seam_gradient(img: np.ndarray, ring: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    return float(g[ring].mean()) if ring.any() else 0.0


# --------------------------------------------------------------------------- #
# Indexing helpers (same as paste_demo)
# --------------------------------------------------------------------------- #


def _index_bgs(root: Path):
    side, top = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        bg = load_background(p)
        v = classify_background(bg, return_info=True)
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


# --------------------------------------------------------------------------- #
# Comparison panel
# --------------------------------------------------------------------------- #


_CYAN = (255, 255, 0)
_ORANGE = (0, 160, 255)


def _label(img, text, xy=(6, 18), color=(0, 255, 0)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _annotate(comp: np.ndarray, pr, method_tag: str, score: float) -> np.ndarray:
    bgr = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
    # paste bbox
    x, y = pr.paste_xy
    ph, pw = pr.mask_patch.shape
    cv2.rectangle(bgr, (x, y), (x + pw, y + ph), _ORANGE, 1)
    # horizon (curve if available)
    if pr.bg_view.kind == "side" and pr.bg_view.horizon_curve is not None:
        pts = pr.bg_view.horizon_curve.polyline(n=max(32, bgr.shape[1] // 8))
        cv2.polylines(bgr, [pts], False, _CYAN, 1, cv2.LINE_AA)
    elif pr.bg_view.kind == "side" and pr.bg_view.horizon_row is not None:
        hr = int(pr.bg_view.horizon_row)
        cv2.line(bgr, (0, hr), (bgr.shape[1] - 1, hr), _CYAN, 1, cv2.LINE_AA)
    _label(bgr, f"{method_tag}   seam={score:.2f}", (6, 18))
    return bgr


def _compose(sample, res, bg, base_rng_seed: int, bg_view) -> dict:
    """Build composites for all 4 variants with the *same* RNG so they
    share paste-site / radiometry / noise. Returns dict method → (pr, score)."""
    out = {}
    for method in ("alpha", "laplacian"):
        for tv in (False, True):
            tag = f"{method}{'+TV' if tv else ''}"
            pr = paste_target(
                sample,
                res.mask,
                bg,
                method=method,
                bg_view=bg_view,
                tv_smooth=tv,
                rng=np.random.default_rng(base_rng_seed),
            )
            # Full-frame mask on composite for seam metric.
            H, W = bg.shape
            fm = np.zeros((H, W), dtype=bool)
            x, y = pr.paste_xy
            ph, pw = pr.mask_patch.shape
            fm[y : y + ph, x : x + pw] = pr.mask_patch
            ring = _boundary_ring(fm, px=1)
            score = seam_gradient(pr.composite, ring)
            out[tag] = (pr, score, ring)
    # Also reference: seam metric on the raw bg at the same ring.
    any_pr, _, any_ring = next(iter(out.values()))
    out["bg_ref"] = (None, seam_gradient(bg, any_ring), any_ring)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-root", default="data/burkeIIA长波")
    ap.add_argument("--bg-root", default="data/background/test_1")
    ap.add_argument("--out", default="outputs/_compare_tv")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("indexing backgrounds ...")
    bg_side, bg_top = _index_bgs(Path(args.bg_root))
    print("indexing targets ...")
    tgt_side, tgt_top = _index_targets(Path(args.targets_root))

    summary = []
    produced = 0
    attempts = 0
    while produced < args.n and attempts < args.n * 6:
        attempts += 1
        kind = "side" if rng.random() < 0.75 else "top"
        if kind == "side" and (not bg_side or not tgt_side):
            kind = "top"
        if kind == "top" and (not bg_top or not tgt_top):
            kind = "side"
        bg_path = (bg_side if kind == "side" else bg_top)[int(rng.integers(len(bg_side if kind == "side" else bg_top)))]
        tgt_stem = (tgt_side if kind == "side" else tgt_top)[int(rng.integers(len(tgt_side if kind == "side" else tgt_top)))]
        try:
            sample = load_sample(tgt_stem)
            res = build_mask(sample)
            if res.n_mask < 40:
                continue
            bg = load_background(bg_path)
            bg_view = classify_background(bg, return_info=True)
            variants = _compose(sample, res, bg, base_rng_seed=args.seed + produced, bg_view=bg_view)
        except Exception as e:
            print(f"  skip: {e}")
            continue

        # Layout: 2×2 grid of the 4 methods.
        order = ["alpha", "alpha+TV", "laplacian", "laplacian+TV"]
        panels = []
        for tag in order:
            pr, score, _ = variants[tag]
            panels.append(_annotate(pr.composite, pr, tag, score))
        h, w = panels[0].shape[:2]
        for i in range(len(panels)):
            if panels[i].shape[:2] != (h, w):
                panels[i] = cv2.resize(panels[i], (w, h))
        top_row = np.concatenate([panels[0], panels[1]], axis=1)
        bot_row = np.concatenate([panels[2], panels[3]], axis=1)
        grid = np.concatenate([top_row, bot_row], axis=0)

        # Title + bg ref score.
        bg_score = variants["bg_ref"][1]
        strip = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
        _label(
            strip,
            f"{kind}: {Path(tgt_stem).name[:26]}  |  bg={bg_path.name}  |  bg-ring={bg_score:.2f}",
            (8, 20),
            (220, 220, 220),
        )
        fig = np.concatenate([strip, grid], axis=0)
        out_name = f"{produced:02d}_{kind}.png"
        cv2.imwrite(str(out / out_name), fig)

        row = dict(
            idx=produced,
            kind=kind,
            bg=bg_path.name,
            target=Path(tgt_stem).name[:28],
            bg_ring=bg_score,
            **{tag: variants[tag][1] for tag in order},
        )
        summary.append(row)
        produced += 1

    # Print summary table.
    print("\n=== Seam gradient (lower = smoother; bg_ring = raw bg reference) ===")
    header = f"{'idx':<4}{'kind':<6}{'bg':<20}{'target':<30}{'bg_ring':>9}" + "".join(
        f"{t:>16}" for t in ["alpha", "alpha+TV", "laplacian", "laplacian+TV"]
    )
    print(header)
    print("-" * len(header))
    for r in summary:
        print(
            f"{r['idx']:<4}{r['kind']:<6}{r['bg'][:18]:<20}{r['target']:<30}"
            f"{r['bg_ring']:>9.2f}"
            f"{r['alpha']:>16.2f}"
            f"{r['alpha+TV']:>16.2f}"
            f"{r['laplacian']:>16.2f}"
            f"{r['laplacian+TV']:>16.2f}"
        )

    # Means.
    if summary:
        import statistics as st
        print("-" * len(header))
        cols = ["bg_ring", "alpha", "alpha+TV", "laplacian", "laplacian+TV"]
        means = {c: st.mean(r[c] for r in summary) for c in cols}
        print(
            f"{'mean':<4}{'':<6}{'':<20}{'':<30}"
            f"{means['bg_ring']:>9.2f}"
            f"{means['alpha']:>16.2f}"
            f"{means['alpha+TV']:>16.2f}"
            f"{means['laplacian']:>16.2f}"
            f"{means['laplacian+TV']:>16.2f}"
        )

    print(f"\nwrote {produced} comparison grids → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
