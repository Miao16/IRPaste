"""CLI: run ``build_mask`` over a folder and report quality metrics.

Usage::

    uv run python scripts/extract_demo.py \
        --folder data/burkeIIA长波/2025_0601_11_晴天 \
        --limit 20 \
        --overlay-dir /home/cvrsg/.copilot/session-state/.../files/overlays
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample  # noqa: E402


def _qa(sample, result) -> dict:
    ann = sample.annotation
    mask = result.mask
    n_mask = int(mask.sum())
    ratio = n_mask / ann.pixel_num if ann.pixel_num else float("nan")

    ys, xs = np.where(mask)
    if xs.size:
        cx = float(xs.mean())
        cy = float(ys.mean())
        cent_err_px = float(np.hypot(cx - ann.center_x, cy - ann.center_y))
        # normalise by bbox diagonal
        diag = float(np.hypot(ann.width, ann.height)) or 1.0
        cent_err_norm = cent_err_px / diag

        # mask extent
        mw = float(xs.max() - xs.min() + 1)
        mh = float(ys.max() - ys.min() + 1)
        ext_w = mw / max(ann.width, 1.0)
        ext_h = mh / max(ann.height, 1.0)

        # containment: fraction of mask pixels inside (unexpanded) xml bbox
        x0, y0, x1, y1 = ann.bbox_xyxy(expand=0.0)
        inside = ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).sum()
        containment = float(inside) / n_mask
    else:
        cent_err_norm = float("nan")
        ext_w = ext_h = 0.0
        containment = float("nan")

    # fragmentation: number of connected components
    if mask.any():
        num, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        n_components = max(0, num - 1)
    else:
        n_components = 0

    return dict(
        n_mask=n_mask,
        pixel_num=ann.pixel_num,
        ratio=ratio,
        centroid_err_norm=cent_err_norm,
        extent_w=ext_w,
        extent_h=ext_h,
        containment=containment,
        n_components=n_components,
    )


def _overlay(sample, result, out_path: Path) -> None:
    """Triptych: [original + GT bbox | mask | overlay]."""
    rad = sample.radiance
    lo, hi = np.percentile(rad, (1, 99))
    disp = np.clip((rad - lo) / max(hi - lo, 1e-6), 0, 1)
    disp = (disp * 255).astype(np.uint8)
    H, W = disp.shape

    # Crop window = context + margin, same for all three panels.
    cx0, cy0, cx1, cy1 = result.context
    pad = 30
    x0 = max(0, cx0 - pad); y0 = max(0, cy0 - pad)
    x1 = min(W, cx1 + pad); y1 = min(H, cy1 + pad)

    # Panel 1: original + GT bbox (unexpanded XML bbox in red,
    # corner polygon in yellow if present, extractor's union anchor
    # in cyan so the viewer can tell when bbox/corners disagree).
    left = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)[y0:y1, x0:x1].copy()
    ann = sample.annotation
    bx0, by0, bx1, by1 = ann.bbox_xyxy(expand=0.0)
    cv2.rectangle(left, (bx0 - x0, by0 - y0), (bx1 - x0, by1 - y0),
                  (0, 0, 255), 1)
    corners = ann.corners_pixel()
    if corners is not None:
        pts = corners.astype(np.int32).copy()
        pts[:, 0] -= x0; pts[:, 1] -= y0
        cv2.polylines(left, [pts], True, (0, 255, 255), 1)
    ax0, ay0, ax1, ay1 = ann.anchor_xyxy(expand=0.0)
    if (ax0, ay0, ax1, ay1) != (bx0, by0, bx1, by1):
        cv2.rectangle(left, (ax0 - x0, ay0 - y0), (ax1 - x0, ay1 - y0),
                      (255, 255, 0), 1)

    # Panel 2: mask alone (white on black).
    mid = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    mid[result.mask[y0:y1, x0:x1]] = (255, 255, 255)

    # Panel 3: green translucent mask on original.
    right = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)[y0:y1, x0:x1].copy()
    m = result.mask[y0:y1, x0:x1]
    if m.any():
        green = right.copy()
        green[m] = (0, 255, 0)
        right = cv2.addWeighted(right, 0.45, green, 0.55, 0)
    # outline the mask so even thin/dark masks are visible.
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if contours:
        cv2.drawContours(right, contours, -1, (0, 255, 0), 1)

    # Stitch with 4-px separators + small labels.
    sep = np.full((y1 - y0, 4, 3), 40, dtype=np.uint8)
    combined = np.hstack([left, sep, mid, sep, right])

    # Add a thin header band with sample info and ratio.
    n_mask = int(result.mask.sum())
    ratio = n_mask / ann.pixel_num if ann.pixel_num else float("nan")
    header_h = 22
    header = np.full((header_h, combined.shape[1], 3), 25, dtype=np.uint8)
    text = f"ratio={ratio:.3f}   mask={n_mask}   gt={ann.pixel_num}"
    cv2.putText(header, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    panel_w = left.shape[1]
    # simple panel captions as a separate footer bar
    footer_h = 18
    footer = np.full((footer_h, combined.shape[1], 3), 25, dtype=np.uint8)
    for idx, label in enumerate(["orig + bbox/poly/anchor", "mask", "overlay"]):
        x = idx * (panel_w + 4) + 6
        cv2.putText(footer, label, (x, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)

    final = np.vstack([header, combined, footer])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True,
                    help="data sub-folder containing <stem>.xml/.dat/.png")
    ap.add_argument("--limit", type=int, default=20,
                    help="max samples (random) to process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay-dir", default=None,
                    help="if set, write diagnostic overlay PNGs there")
    args = ap.parse_args()

    folder = Path(args.folder)
    xmls = sorted(folder.glob("*.xml"))
    if not xmls:
        print(f"No XML files in {folder}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    if len(xmls) > args.limit:
        xmls = rng.sample(xmls, args.limit)

    rows = []
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None

    for i, xml in enumerate(xmls):
        try:
            sample = load_sample(xml)
            result = build_mask(sample)
        except Exception as e:
            print(f"[{i+1}/{len(xmls)}] {xml.name}: ERROR {e}")
            continue
        qa = _qa(sample, result)
        rows.append((xml.name, qa, result.notes))
        print(
            f"[{i+1}/{len(xmls)}] {xml.name}  "
            f"ratio={qa['ratio']:.2f}  mask={qa['n_mask']}  gt={qa['pixel_num']}  "
            f"cent={qa['centroid_err_norm']:.2f}  ext=({qa['extent_w']:.2f},{qa['extent_h']:.2f})  "
            f"contain={qa['containment']:.2f}  comp={qa['n_components']}  "
            f"notes={'; '.join(result.notes)}"
        )
        if overlay_dir is not None:
            _overlay(sample, result, overlay_dir / (xml.stem + ".png"))

    if not rows:
        return 1

    ratios = [r[1]["ratio"] for r in rows if np.isfinite(r[1]["ratio"])]
    cent = [r[1]["centroid_err_norm"] for r in rows if np.isfinite(r[1]["centroid_err_norm"])]
    contain = [r[1]["containment"] for r in rows if np.isfinite(r[1]["containment"])]
    comp = [r[1]["n_components"] for r in rows]
    print("\n=== Aggregate ===")
    print(f"n={len(rows)}")
    if ratios:
        print(f"ratio  mean={mean(ratios):.3f} median={median(ratios):.3f} "
              f"min={min(ratios):.3f} max={max(ratios):.3f}")
        good = sum(1 for r in ratios if 0.7 <= r <= 1.3)
        print(f"ratio in [0.7, 1.3]: {good}/{len(ratios)}")
    if cent:
        print(f"centroid_err_norm  mean={mean(cent):.3f} median={median(cent):.3f}")
    if contain:
        print(f"containment       mean={mean(contain):.3f} median={median(contain):.3f}")
    if comp:
        print(f"components        mean={mean(comp):.2f} median={median(comp):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
