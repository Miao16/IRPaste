"""Interactive horizon calibration tool for background images.

Usage::

    uv run python scripts/calibrate_horizon.py --bg-root bg/

Mouse:
  Left-click near a point (<=8px)    -> select it (white ring)
  Left-click elsewhere               -> add control point (green dot)
  Right-click on a point             -> remove it
  h / j / k / l                      -> nudge selected point left/down/up/right 1px

Keyboard:
  s  -> Save cache, rename file, advance
  d  -> Skip (mark in _skip.json), advance
  t  -> Toggle view kind (side <-> top)
  r  -> Reset to auto-computed curve
  c  -> Clear all manual control points
  p  -> Previous image
  q / Esc -> Quit (saves progress)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste.paste import load_background  # noqa: E402
from irpaste.viewcls import classify_background, HorizonCurve  # noqa: E402
from irpaste.horizon_cache import HorizonCacheData, save_horizon  # noqa: E402


# --- Constants ---

POINT_RADIUS = 6
SELECT_RADIUS = 8
SELECT_RING = 2
CURVE_COLOR = (255, 255, 0)       # cyan (BGR)
AUTO_COLOR = (0, 255, 255)        # yellow (BGR)
POINT_COLOR = (0, 255, 0)         # green (BGR)
SELECT_COLOR = (255, 255, 255)    # white (BGR)
SKY_TINT_COLOR = (200, 150, 50)   # brownish-blue tint (BGR)
SKY_TINT_ALPHA = 0.15


# --- Data loading ---

def _image_stems(bg_root: Path) -> list[Path]:
    """Return sorted list of image paths in bg_root (excluding _tile variants)."""
    exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif"}
    paths = []
    for p in sorted(bg_root.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        if "_tile" in p.stem:
            continue
        paths.append(p)
    return paths


def _load_skip_set(bg_root: Path) -> set[str]:
    skip_path = bg_root / "_skip.json"
    if not skip_path.exists():
        return set()
    with skip_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("skipped", []))


def _save_skip_set(bg_root: Path, skip_set: set[str]) -> None:
    skip_path = bg_root / "_skip.json"
    data = {"version": 1, "skipped": sorted(skip_set), "notes": {}}
    with skip_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_progress(bg_root: Path) -> Optional[str]:
    prog_path = bg_root / "_progress.json"
    if not prog_path.exists():
        return None
    with prog_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("last_stem")


def _save_progress(bg_root: Path, last_stem: str, total: int) -> None:
    prog_path = bg_root / "_progress.json"
    with prog_path.open("w", encoding="utf-8") as f:
        json.dump({"version": 1, "last_stem": last_stem, "total": total}, f, indent=2)


def _filter_uncalibrated(all_paths: list[Path], skip_set: set[str]) -> list[Path]:
    result = []
    for p in all_paths:
        stem = p.stem
        if stem.startswith("side_") or stem.startswith("top_"):
            continue
        if stem in skip_set:
            continue
        result.append(p)
    return result


def _rename_log_path(bg_root: Path) -> Path:
    return bg_root / "_rename_log.csv"


def _log_rename(bg_root: Path, old_name: str, new_name: str) -> None:
    log = _rename_log_path(bg_root)
    is_new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["old_name", "new_name"])
        w.writerow([old_name, new_name])


# --- Quadratic curve fitting ---

def _fit_quadratic(pts: list[tuple[int, int]], W: int) -> Optional[HorizonCurve]:
    """Fit y = a*x**2 + b*x + c through control points (image coords)."""
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    try:
        poly = np.polyfit(xs, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    pred = np.polyval(poly, xs)
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return HorizonCurve(
        a=float(poly[0]),
        b=float(poly[1]),
        c=float(poly[2]),
        rmse=rmse,
        n_inliers=len(pts),
        width=int(W),
    )


def _curve_points(curve: HorizonCurve, W: int) -> np.ndarray:
    """Return (N, 2) int32 polyline for rendering."""
    xs = np.linspace(0, W - 1, max(W // 2, 64))
    ys = curve.y_at(xs)
    pts = np.stack([xs, ys], axis=1)
    return pts.round().astype(np.int32)


# --- Rendering ---

def _render_overlay(
    bg: np.ndarray,
    curve: Optional[HorizonCurve],
    auto_curve: Optional[HorizonCurve],
    ctrl_pts: list[tuple[int, int]],
    selected_idx: int,
    view_kind: str,
) -> np.ndarray:
    """Return a BGR image with horizon overlay."""
    if bg.ndim == 2:
        bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    else:
        bgr = bg.copy() if bg.shape[2] == 3 else cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    H, W = bgr.shape[:2]

    # Sky tint: semi-transparent blue above the active curve.
    active = curve if curve is not None else auto_curve
    if active is not None and view_kind == "side":
        ys = active.y_at(np.arange(W))
        overlay = bgr.copy()
        for x in range(W):
            yh = max(0, min(H, int(round(ys[x]))))
            if yh > 0:
                overlay[:yh, x] = SKY_TINT_COLOR
        bgr = cv2.addWeighted(bgr, 1.0 - SKY_TINT_ALPHA, overlay, SKY_TINT_ALPHA, 0)

    # Auto-computed curve (yellow dashed) - always show for reference.
    if auto_curve is not None and curve is not None:
        pts = _curve_points(auto_curve, W)
        for i in range(len(pts) - 1):
            if i % 3 == 0:  # dash pattern
                cv2.line(bgr, tuple(pts[i]), tuple(pts[i + 1]), AUTO_COLOR, 1, cv2.LINE_AA)

    # Active curve (cyan solid).
    if curve is not None:
        pts = _curve_points(curve, W)
        cv2.polylines(bgr, [pts], False, CURVE_COLOR, 2, cv2.LINE_AA)
    elif auto_curve is not None:
        # Show auto curve as solid cyan when no manual curve.
        pts = _curve_points(auto_curve, W)
        cv2.polylines(bgr, [pts], False, CURVE_COLOR, 2, cv2.LINE_AA)

    # Control points (green dots).
    for i, (px, py) in enumerate(ctrl_pts):
        cx, cy = int(px), int(py)
        cv2.circle(bgr, (cx, cy), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)
        if i == selected_idx:
            cv2.circle(bgr, (cx, cy), POINT_RADIUS + SELECT_RING, SELECT_COLOR, SELECT_RING, cv2.LINE_AA)

    return bgr


# --- File renaming ---

def _next_sequence(bg_root: Path, kind: str) -> int:
    """Find the next available sequence number for side_XXXXXX or top_XXXXXX."""
    max_n = 0
    prefix = f"{kind}_"
    for p in bg_root.iterdir():
        stem = p.stem
        if stem.startswith(prefix):
            try:
                n = int(stem[len(prefix):])
                max_n = max(max_n, n)
            except ValueError:
                pass
    return max_n + 1


def _rename_to_kind(
    bg_root: Path, image_path: Path, kind: str
) -> tuple[Path, str]:
    """Rename image to side_XXXXXX.ext or top_XXXXXX.ext. Returns (new_path, new_stem)."""
    seq = _next_sequence(bg_root, kind)
    new_stem = f"{kind}_{seq:06d}"
    new_path = image_path.with_stem(new_stem)
    if new_path != image_path:
        _log_rename(bg_root, image_path.name, new_path.name)
        image_path.rename(new_path)
    return new_path, new_stem


def _save_current(
    bg_root: Path,
    image_path: Path,
    bg: np.ndarray,
    view_kind: str,
    curve: Optional[HorizonCurve],
    auto_curve: Optional[HorizonCurve],
) -> Path:
    """Save horizon cache and rename file. Returns new image path."""
    # Use manual curve if available, else auto.
    final_curve = curve if curve is not None else auto_curve
    if final_curve is not None:
        horizon_row = int(round(float(final_curve.y_at(bg.shape[1] / 2.0))))
    else:
        horizon_row = None

    data = HorizonCacheData(
        kind=view_kind,
        horizon_curve={
            "a": final_curve.a,
            "b": final_curve.b,
            "c": final_curve.c,
            "rmse": final_curve.rmse,
            "n_inliers": final_curve.n_inliers,
            "width": final_curve.width,
        } if final_curve is not None else None,
        horizon_row=horizon_row,
    )

    # Rename image to side_XXXXXX / top_XXXXXX first.
    new_path, new_stem = _rename_to_kind(bg_root, image_path, view_kind)
    # Save JSON cache next to renamed image.
    save_horizon(new_path, data)
    return new_path


# --- Mouse callback ---

class _State:
    """Mutable state shared between main loop and mouse callback."""
    ctrl_pts: list[tuple[int, int]]
    selected_idx: int
    curve: Optional[HorizonCurve]
    auto_curve: Optional[HorizonCurve]
    view_kind: str

    def __init__(self, auto_curve, view_kind):
        self.ctrl_pts = []
        self.selected_idx = -1
        self.curve = None
        self.auto_curve = auto_curve
        self.view_kind = view_kind


def _find_near_point(pts: list[tuple[int, int]], x: int, y: int, max_dist: int = 8) -> int:
    """Return index of nearest point within max_dist px, or -1."""
    best_i, best_d = -1, max_dist + 1
    for i, (px, py) in enumerate(pts):
        d = np.hypot(px - x, py - y)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _make_mouse_cb(state: _State, W: int):
    """Factory returning a mouse callback that captures state."""

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near an existing point.
            idx = _find_near_point(state.ctrl_pts, x, y, max_dist=8)
            if idx >= 0:
                state.selected_idx = idx
            else:
                state.ctrl_pts.append((x, y))
                state.selected_idx = len(state.ctrl_pts) - 1
            # Re-fit if >=3 points.
            if len(state.ctrl_pts) >= 3:
                state.curve = _fit_quadratic(state.ctrl_pts, W)

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = _find_near_point(state.ctrl_pts, x, y, max_dist=8)
            if idx >= 0:
                del state.ctrl_pts[idx]
                if state.selected_idx >= len(state.ctrl_pts):
                    state.selected_idx = len(state.ctrl_pts) - 1
                if len(state.ctrl_pts) >= 3:
                    state.curve = _fit_quadratic(state.ctrl_pts, W)
                else:
                    state.curve = None

    return on_mouse


# --- Main loop ---

def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive horizon calibration")
    ap.add_argument("--bg-root", default="bg", help="background images directory")
    ap.add_argument("--start", type=int, default=None, help="start at image index")
    args = ap.parse_args()

    bg_root = Path(args.bg_root).resolve()
    if not bg_root.is_dir():
        print(f"ERROR: {bg_root} is not a directory")
        return 1

    all_paths = _image_stems(bg_root)
    if not all_paths:
        print(f"No images found in {bg_root}")
        return 1

    skip_set = _load_skip_set(bg_root)

    # Filter out already-renamed files (already calibrated) and skipped files.
    uncalibrated = _filter_uncalibrated(all_paths, skip_set)

    total = len(uncalibrated)
    if total == 0:
        print("All images are already calibrated or skipped.")
        return 0

    if args.start is not None:
        start_idx = max(0, min(args.start, total - 1))
    else:
        last_stem = _load_progress(bg_root)
        start_idx = 0
        if last_stem is not None:
            for i, p in enumerate(uncalibrated):
                if p.stem == last_stem:
                    start_idx = i
                    break

    print(f"Found {total} uncalibrated images. Starting at #{start_idx + 1}.")
    print("Controls: S=save  D=skip  T=toggle view  R=reset  C=clear  P=prev  Q=quit")
    print("Mouse:   left-click=add/select point  right-click=remove point  hjkl=nudge")

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    idx = start_idx
    while 0 <= idx < total:
        image_path = uncalibrated[idx]
        print(f"\n[{idx + 1}/{total}] {image_path.name}")

        # Load image and classify.
        try:
            bg = load_background(image_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            skip_set.add(image_path.stem)
            _save_skip_set(bg_root, skip_set)
            idx += 1
            _save_progress(bg_root, image_path.stem, total)
            continue

        H, W = bg.shape[:2]

        try:
            bg_view = classify_background(bg, return_info=True)
        except Exception as e:
            print(f"  classify_background error: {e}")
            bg_view = None

        auto_curve = bg_view.horizon_curve if bg_view and bg_view.kind == "side" else None
        view_kind = bg_view.kind if bg_view else "top"

        # State for this image.
        state = _State(auto_curve, view_kind)
        cv2.setMouseCallback("calibrate", _make_mouse_cb(state, W))

        while True:
            display = _render_overlay(
                bg, state.curve, state.auto_curve,
                state.ctrl_pts, state.selected_idx, state.view_kind,
            )
            # Title bar with status.
            n_pts = len(state.ctrl_pts)
            title = (
                f"[{idx + 1}/{total}] {state.view_kind} | "
                f"pts:{n_pts} | S=save D=skip T=toggle R=reset C=clear P=prev Q=quit"
            )
            cv2.setWindowTitle("calibrate", title)
            cv2.imshow("calibrate", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q") or key == 27:  # q or Esc
                _save_progress(bg_root, image_path.stem, total)
                print(f"\nProgress saved at {idx + 1}/{total}. Bye.")
                cv2.destroyAllWindows()
                return 0

            elif key == ord("s"):
                new_path = _save_current(
                    bg_root, image_path, bg, state.view_kind,
                    state.curve, state.auto_curve,
                )
                print(f"  Saved -> {new_path.name}")
                idx += 1
                _save_progress(bg_root, image_path.stem, total)
                break

            elif key == ord("d"):
                skip_set.add(image_path.stem)
                _save_skip_set(bg_root, skip_set)
                print(f"  Skipped -> {image_path.name} (added to _skip.json)")
                idx += 1
                _save_progress(bg_root, image_path.stem, total)
                break

            elif key == ord("t"):
                state.view_kind = "top" if state.view_kind == "side" else "side"
                print(f"  View toggled -> {state.view_kind}")

            elif key == ord("r"):
                state.ctrl_pts = []
                state.curve = None
                state.selected_idx = -1
                print("  Reset to auto-computed curve")

            elif key == ord("c"):
                state.ctrl_pts = []
                state.curve = None
                state.selected_idx = -1
                print("  Cleared all control points")

            elif key == ord("p"):
                if idx > 0:
                    prev_uncalibrated = _filter_uncalibrated(all_paths, skip_set)
                    if prev_uncalibrated:
                        idx = min(idx - 1, len(prev_uncalibrated) - 1)
                        uncalibrated = prev_uncalibrated
                        total = len(uncalibrated)
                        _save_progress(bg_root, uncalibrated[idx].stem, total)
                        print(f"  Going back to #{idx + 1}")
                    break

            # Nudge selected point with hjkl (vim-style, cross-platform).
            elif key in (ord("h"), ord("j"), ord("k"), ord("l")):
                if state.selected_idx >= 0:
                    px, py = state.ctrl_pts[state.selected_idx]
                    if key == ord("h"):   # left
                        px = max(0, px - 1)
                    elif key == ord("l"):  # right
                        px = min(W - 1, px + 1)
                    elif key == ord("k"):  # up
                        py = max(0, py - 1)
                    elif key == ord("j"):  # down
                        py = min(H - 1, py + 1)
                    state.ctrl_pts[state.selected_idx] = (px, py)
                    if len(state.ctrl_pts) >= 3:
                        state.curve = _fit_quadratic(state.ctrl_pts, W)

    cv2.destroyAllWindows()
    print(f"\nAll {total} images processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
