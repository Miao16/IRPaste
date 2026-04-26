"""end-to-end smoke test for the IRPaste pipeline.

Usage
-----
# Run from the project root (IRPaste-main/)
python scripts/test_pipeline.py --sample-dir <path/to/sample_folder>
                                 --bg-dir      <path/to/background_dir>
                                 [--out-dir    outputs/_test]
                                 [--n-pairs    3]
                                 [--ship-scale-min 0.55]
                                 [--ship-scale-max 0.90]
                                 [--augment-bg]
                                 [--align-axis]
                                 [--blend-mode poisson|alpha|lap]

The script picks ``--n-pairs`` random (sample, background) pairs,
runs the full extract → paste pipeline, and writes:
  <out_dir>/composite_<i>.png   — final 512×512 composite
  <out_dir>/mask_<i>.png        — binary ship mask on composite
  <out_dir>/label_<i>.txt       — YOLO HBB label (class cx cy w h)
  <out_dir>/debug_<i>.png       — 4-panel debug visualisation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Allow running as ``python scripts/test_pipeline.py`` from project root.
# --------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from irpaste import (
    PasteResult,
    augment_background,
    build_mask,
    classify_background,
    load_background,
    load_sample,
    paste_target,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _imsave(path: Path, img: np.ndarray) -> None:
    """Save *img* to *path*, handling non-ASCII paths on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ret, buf = cv2.imencode(ext, img)
    if ret:
        buf.tofile(str(path))


def _find_samples(sample_dir: Path) -> list[Path]:
    """Return stem paths for all complete (.xml + .dat) sample pairs."""
    xmls = sorted(sample_dir.glob("*.xml"))
    stems = [x for x in xmls if (x.with_suffix(".dat")).exists()]
    return stems


def _find_backgrounds(bg_dir: Path) -> list[Path]:
    exts = {".png", ".bmp", ".jpg", ".jpeg"}
    return [p for p in sorted(bg_dir.rglob("*")) if p.suffix.lower() in exts]


def _build_debug_panel(
    bg_orig: np.ndarray,
    composite: np.ndarray,
    mask: np.ndarray,
    info_text: list[str],
) -> np.ndarray:
    """Return a 2×2 panel: orig bg | composite | mask | text info."""
    H, W = composite.shape
    vis_bg = cv2.resize(bg_orig, (W, H), interpolation=cv2.INTER_LINEAR)

    def to3(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    vis_bg = to3(vis_bg)
    vis_comp = to3(composite)

    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[:, :, 2] = cv2.resize(
        mask.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST
    )

    text_panel = np.zeros((H, W, 3), dtype=np.uint8)
    y0 = 28
    for i, line in enumerate(info_text[:18]):
        cv2.putText(
            text_panel,
            line,
            (8, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 200),
            1,
        )

    top = np.hstack([vis_bg, vis_comp])
    bot = np.hstack([mask_vis, text_panel])
    return np.vstack([top, bot])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="IRPaste end-to-end smoke test")
    ap.add_argument(
        "--sample-dir", required=True, help="directory with .xml+.dat pairs"
    )
    ap.add_argument("--bg-dir", required=True, help="directory with background images")
    ap.add_argument("--out-dir", default="outputs/_test")
    ap.add_argument(
        "--n-pairs", type=int, default=3, help="number of composites to generate"
    )
    ap.add_argument("--ship-scale-min", type=float, default=0.55)
    ap.add_argument("--ship-scale-max", type=float, default=0.90)
    ap.add_argument(
        "--bg-scale-max",
        type=float,
        default=1.3,
        help="background random zoom factor upper bound",
    )
    ap.add_argument("--augment-bg", action="store_true")
    ap.add_argument("--align-axis", action="store_true")
    ap.add_argument(
        "--blend-mode", default="poisson", choices=["poisson", "alpha", "lap"]
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    bg_dir = Path(args.bg_dir)
    out_dir = Path(args.out_dir)

    samples = _find_samples(sample_dir)
    if not samples:
        sys.exit(f"[ERROR] No complete .xml+.dat samples found in {sample_dir}")
    backgrounds = _find_backgrounds(bg_dir)
    if not backgrounds:
        sys.exit(f"[ERROR] No background images found in {bg_dir}")

    print(f"Found {len(samples)} samples, {len(backgrounds)} backgrounds.")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_pairs, max(len(samples), len(backgrounds)))
    s_idxs = rng.choice(len(samples), size=n, replace=True)
    b_idxs = rng.choice(len(backgrounds), size=n, replace=True)

    passed = 0
    for i, (si, bi) in enumerate(zip(s_idxs.tolist(), b_idxs.tolist())):
        sample_xml = samples[si]
        bg_path = backgrounds[bi]
        print(f"\n[{i+1}/{n}] sample={sample_xml.name}  bg={bg_path.name}")

        try:
            # 1. Load
            sample = load_sample(sample_xml.with_suffix(""))
            bg_raw = load_background(bg_path)

            # 2. Classify background
            bv = classify_background(bg_raw, return_info=True)
            print(
                f"       background: kind={bv.kind}  horizon_row={bv.horizon_row}"
                f"  score={bv.score:.1f}"
            )

            # 3. Extract mask
            ext_res = build_mask(sample)
            print(
                f"       mask: fill={ext_res.fill_ratio:.3f}  "
                f"quality={ext_res.quality:.3f}  notes={ext_res.notes[:3]}"
            )

            # 4. Paste
            bg = bg_raw.copy()
            pr: PasteResult = paste_target(
                sample,
                ext_res.mask,
                bg,
                bg_path=bg_path,
                rng=rng,
                augment_bg=args.augment_bg,
                bg_scale_range=(1.0, args.bg_scale_max),
                align_to_horizon=args.align_axis,
                ship_scale_range=(args.ship_scale_min, args.ship_scale_max),
                blend_mode=args.blend_mode,
            )
            print(
                f"       paste_xy={pr.paste_xy}  "
                f"mask_px={pr.mask_patch.sum()}  notes={pr.notes[:3]}"
            )

            composite = pr.composite  # uint8 grayscale 512×512
            H, W = composite.shape

            # 5. Tight YOLO bbox from actual mask pixels
            mask_ys, mask_xs = np.where(pr.mask_patch)
            if mask_ys.size > 0:
                mk_x0 = int(mask_xs.min())
                mk_x1 = int(mask_xs.max()) + 1
                mk_y0 = int(mask_ys.min())
                mk_y1 = int(mask_ys.max()) + 1
            else:
                mk_x0, mk_y0 = 0, 0
                mk_x1, mk_y1 = pr.mask_patch.shape[1], pr.mask_patch.shape[0]
            px, py = pr.paste_xy
            cx_n = (px + mk_x0 + (mk_x1 - mk_x0) / 2.0) / W
            cy_n = (py + mk_y0 + (mk_y1 - mk_y0) / 2.0) / H
            w_n = (mk_x1 - mk_x0) / W
            h_n = (mk_y1 - mk_y0) / H
            cx_n, cy_n = max(0.0, min(1.0, cx_n)), max(0.0, min(1.0, cy_n))
            w_n, h_n = max(0.0, min(1.0, w_n)), max(0.0, min(1.0, h_n))

            # 6. Save outputs
            _imsave(out_dir / f"composite_{i:02d}.png", composite)
            mask_on_comp = np.zeros((H, W), dtype=np.uint8)
            mask_on_comp[
                py : py + pr.mask_patch.shape[0],
                px : px + pr.mask_patch.shape[1],
            ] = (pr.mask_patch * 255).astype(np.uint8)
            _imsave(out_dir / f"mask_{i:02d}.png", mask_on_comp)

            label_path = out_dir / f"label_{i:02d}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

            debug_panel = _build_debug_panel(
                bg_raw,
                composite,
                pr.mask_patch,
                [
                    f"sample: {sample_xml.name}",
                    f"bg:     {bg_path.name}",
                    f"kind:   {bv.kind}",
                    f"horizon: {bv.horizon_row}",
                    f"step:   {bv.score:.1f}",
                    "---",
                    f"fill:   {ext_res.fill_ratio:.3f}",
                    f"qual:   {ext_res.quality:.3f}",
                    "---",
                    f"paste:  {pr.paste_xy}",
                    f"blend:  {args.blend_mode}",
                    f"scale:  {args.ship_scale_min:.2f}-{args.ship_scale_max:.2f}",
                    f"cx/cy:  {cx_n:.3f} / {cy_n:.3f}",
                    f"w/h:    {w_n:.3f} / {h_n:.3f}",
                ]
                + [f"  {n}" for n in pr.notes[:4]],
            )
            _imsave(out_dir / f"debug_{i:02d}.png", debug_panel)
            print(f"       saved → {out_dir}/composite_{i:02d}.png  [PASS]")
            passed += 1

        except Exception as exc:
            print(f"       [FAIL] {type(exc).__name__}: {exc}")

    print(f"\n{'='*50}")
    print(f"Smoke test done: {passed}/{n} passed")
    if passed < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
