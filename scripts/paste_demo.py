"""Demo: sample view-matched (target, background) pairs and emit composites.

Usage::

    uv run python scripts/paste_demo.py \
        --targets-root data/burkeIIA长波 \
        --bg-root      data/background/test_1 \
        --out          outputs/_paste \
        --n 30 --seed 1 --method poisson
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample  # noqa: E402
from irpaste.paste import paste_target, load_background, sim_preview_u8  # noqa: E402
from irpaste.viewcls import classify_background, classify_target  # noqa: E402


def _index_bgs(root: Path) -> tuple[list[Path], list[Path]]:
    side, top = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        bg = load_background(p)
        v = classify_background(bg, return_info=True)
        (side if v.kind == "side" else top).append(p)
    return side, top


def _index_targets(root: Path) -> tuple[list[Path], list[Path]]:
    side, top = [], []
    for xml in root.rglob("*.xml"):
        try:
            k = classify_target(xml)
        except Exception:
            continue
        (side if k == "side" else top).append(xml.with_suffix(""))
    return side, top


# --------------------------------------------------------------------------- #
# Visualisation — two panels: (sim + mask + GT bbox)  |  (composite + paste bbox)
# --------------------------------------------------------------------------- #


_YELLOW = (0, 255, 255)   # GT bbox
_RED    = (0,   0, 255)   # mask overlay / contour
_ORANGE = (0, 160, 255)   # paste bbox
_CYAN   = (255, 255, 0)   # horizon line
_GREEN  = (0, 255,   0)   # labels


def _overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=_RED, alpha: float = 0.45) -> None:
    """Blend a translucent colour where mask is True; mutates in place."""
    if mask.sum() == 0:
        return
    layer = np.zeros_like(img_bgr)
    layer[mask] = color
    img_bgr[mask] = cv2.addWeighted(img_bgr, 1 - alpha, layer, alpha, 0)[mask]


def _label(img: np.ndarray, text: str, xy=(6, 18), color=_GREEN) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_horizon(img_bgr: np.ndarray, curve, fallback_row, color=_CYAN) -> None:
    """Draw the horizon: quadratic polyline if a curve is given, else a horizontal line."""
    h, w = img_bgr.shape[:2]
    if curve is not None:
        pts = curve.polyline(n=max(32, w // 8))
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
    elif fallback_row is not None:
        cv2.line(img_bgr, (0, int(fallback_row)), (w - 1, int(fallback_row)), color, 1, cv2.LINE_AA)


def _make_panel(sample, mask, pr, title: str) -> np.ndarray:
    """Two side-by-side panels:
       left  : sim image + GT bbox (yellow) + mask overlay (red) + sim horizon (cyan)
       right : composite + paste bbox (orange) + bg horizon (cyan)
    """
    # ---- Left: simulation image -----------------------------------------
    sim = sim_preview_u8(sample)
    sim_bgr = cv2.cvtColor(sim, cv2.COLOR_GRAY2BGR)
    _overlay_mask(sim_bgr, mask.astype(bool), _RED, 0.45)
    m8 = mask.astype(np.uint8) * 255
    contours, _c = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(sim_bgr, contours, -1, _RED, 1)
    # GT bbox from XML (anchor = bbox ∪ polygon corners, +5 %).
    ann = sample.annotation
    gx0, gy0, gx1, gy1 = ann.anchor_xyxy(expand=0.05, clip=True)
    cv2.rectangle(sim_bgr, (gx0, gy0), (gx1, gy1), _YELLOW, 2)
    _label(sim_bgr, "GT bbox", (gx0, max(14, gy0 - 4)), _YELLOW)
    # Sim horizon curve (only if on horizon).
    if pr.target_on_horizon:
        sim_view = classify_background(sim, return_info=True)
        _draw_horizon(sim_bgr, sim_view.horizon_curve, sim_view.horizon_row, _CYAN)
        _label(sim_bgr, "sim horizon", (6, sim_bgr.shape[0] - 8), _CYAN)
    tag = "ON HORIZON" if pr.target_on_horizon else "IN OCEAN" if pr.bg_view.kind == "side" else "TOP-DOWN"
    _label(sim_bgr, f"sim + mask  [{tag}]", (6, 18))

    # ---- Right: composite ------------------------------------------------
    comp_bgr = cv2.cvtColor(pr.composite, cv2.COLOR_GRAY2BGR)
    # bg horizon curve
    if pr.bg_view.kind == "side":
        _draw_horizon(comp_bgr, pr.bg_view.horizon_curve, pr.bg_view.horizon_row, _CYAN)
        _label(comp_bgr, "bg horizon", (6, comp_bgr.shape[0] - 8), _CYAN)
    # Paste bbox
    x, y = pr.paste_xy
    ph, pw = pr.mask_patch.shape
    cv2.rectangle(comp_bgr, (x, y), (x + pw, y + ph), _ORANGE, 2)
    _label(comp_bgr, f"paste ({pw}x{ph})", (x, max(14, y - 4)), _ORANGE)
    _label(comp_bgr, f"composite  [{pr.method}  bg={pr.bg_view.kind}]", (6, 18))

    # ---- Stitch ----------------------------------------------------------
    h = max(sim_bgr.shape[0], comp_bgr.shape[0])
    def _pad(img: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    panel = np.concatenate([_pad(sim_bgr), _pad(comp_bgr)], axis=1)

    strip = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
    _label(strip, title[:160], (8, 20), (220, 220, 220))
    return np.concatenate([strip, panel], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-root", default="data/burkeIIA长波")
    ap.add_argument("--bg-root", default="data/background/test_1")
    ap.add_argument("--out", default="outputs/_paste")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", default="alpha", choices=["poisson", "alpha", "laplacian"])
    ap.add_argument("--tv", action="store_true", help="apply TV-L1 boundary smoother")
    ap.add_argument("--no-noise", action="store_true", help="disable noise matching")
    ap.add_argument("--all-methods", action="store_true", help="also dump alpha & laplacian variants")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print("indexing backgrounds ...")
    bg_side, bg_top = _index_bgs(Path(args.bg_root))
    print(f"  bg: {len(bg_side)} side, {len(bg_top)} top")

    print("indexing targets ...")
    tgt_side, tgt_top = _index_targets(Path(args.targets_root))
    print(f"  targets: {len(tgt_side)} side, {len(tgt_top)} top")

    produced = 0
    attempts = 0
    while produced < args.n and attempts < args.n * 5:
        attempts += 1
        kind = "side" if rng.random() < 0.8 else "top"
        if kind == "side" and (not bg_side or not tgt_side):
            kind = "top"
        if kind == "top" and (not bg_top or not tgt_top):
            kind = "side"
        if kind == "side":
            bg_path = bg_side[int(rng.integers(len(bg_side)))]
            tgt_stem = tgt_side[int(rng.integers(len(tgt_side)))]
        else:
            bg_path = bg_top[int(rng.integers(len(bg_top)))]
            tgt_stem = tgt_top[int(rng.integers(len(tgt_top)))]

        try:
            sample = load_sample(tgt_stem)
            res = build_mask(sample)
            if res.n_mask < 40:
                continue
            bg = load_background(bg_path)
            methods = [args.method] if not args.all_methods else ["poisson", "alpha", "laplacian"]
            panels = []
            title = f"{kind}: {Path(tgt_stem).name[:28]}  |  bg={bg_path.name}"
            for m in methods:
                pr = paste_target(
                    sample,
                    res.mask,
                    bg,
                    method=m,
                    match_noise=not args.no_noise,
                    tv_smooth=args.tv,
                    rng=np.random.default_rng(args.seed + produced),
                )
                panels.append(_make_panel(sample, res.mask, pr, title + f"  [{m}]"))
            stacked = np.concatenate(panels, axis=0) if len(panels) > 1 else panels[0]
            out_name = f"{produced:03d}_{kind}_{args.method}.png"
            cv2.imwrite(str(out / out_name), stacked)
            produced += 1
        except Exception as e:
            print(f"  skip {tgt_stem}: {e}")
            continue

    print(f"wrote {produced} composites → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
