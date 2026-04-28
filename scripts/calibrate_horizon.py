"""Interactive horizon calibration tool for background images.

Usage::

    uv run python scripts/calibrate_horizon.py --bg-root bg/

Modes
-----
Default (no mode flag)
    Interactive calibration of uncalibrated images.

--auto-save-all
    Headless batch: auto-classify every uncalibrated image, save cache, rename.
    No GUI window.  Prints progress and a summary.

--review [--view side|top|both]
    Interactive review of **already-calibrated** images (side_* / top_*).
    Same mouse/keyboard controls; S re-saves the cache JSON.
    If view is toggled with T, file is renamed + old cache deleted on save.
    X marks the image as deleted (skip + delete cache + rename file).

--single STEM
    Interactively calibrate one image by stem, e.g. ``--single IMG_1234``.

--export-report CSV
    Scan all side_*/top_* images under --bg-root and write a CSV with
    stem, kind, curve coefficients, horizon_row, rmse, n_inliers.
    Non-interactive.

Interactive controls
--------------------
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
  x  -> In review mode: mark as deleted (skip + delete cache + rename img)
         In calibrate mode: same as skip
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
from irpaste.viewcls import classify_background, fit_horizon_curve, HorizonCurve  # noqa: E402
from irpaste.horizon_cache import HorizonCacheData, save_horizon, load_horizon, cache_path_for_image  # noqa: E402


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


def _save_skip_set(bg_root: Path, skip_set: set[str], deleted: set[str] | None = None) -> None:
    skip_path = bg_root / "_skip.json"
    data = {"version": 1, "skipped": sorted(skip_set), "notes": {}}
    if deleted:
        data["deleted"] = sorted(deleted)
    with skip_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _mark_deleted(bg_root: Path, image_path: Path) -> None:
    """Mark a calibrated image as deleted: add to _skip.json, remove JSON cache,
    rename image with _deleted_ prefix."""
    skip_path = bg_root / "_skip.json"
    if skip_path.exists():
        with skip_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"version": 1, "skipped": [], "deleted": [], "notes": {}}

    # Update both skipped and deleted lists.
    skip_set = set(data.get("skipped", []))
    deleted_set = set(data.get("deleted", []))
    skip_set.add(image_path.stem)
    deleted_set.add(image_path.stem)
    data["skipped"] = sorted(skip_set)
    data["deleted"] = sorted(deleted_set)
    with skip_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Delete the JSON cache file.
    cache_path = cache_path_for_image(image_path)
    if cache_path.exists():
        cache_path.unlink()
        print(f"  Deleted cache: {cache_path.name}")

    # Rename image with _deleted_ prefix.
    new_name = f"_deleted_{image_path.name}"
    new_path = image_path.with_name(new_name)
    image_path.rename(new_path)
    _log_rename(bg_root, image_path.name, new_name)
    print(f"  Marked deleted -> {new_name}")


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


def _filter_calibrated(all_paths: list[Path], view: str | None = None) -> list[Path]:
    """Return paths that are already renamed (side_* / top_*), optionally filtered by view."""
    result = [p for p in all_paths
              if p.stem.startswith("side_") or p.stem.startswith("top_")]
    if view == "side":
        result = [p for p in result if p.stem.startswith("side_")]
    elif view == "top":
        result = [p for p in result if p.stem.startswith("top_")]
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
    rename: bool = True,
    original_kind: str | None = None,
) -> Path:
    """Save horizon cache and optionally rename file. Returns (possibly new) image path.

    When *original_kind* differs from *view_kind*, forces rename (for review mode
    where the user toggled the view and the file stem must change).
    """
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

    # Determine whether renaming is needed.
    kind_changed = (original_kind is not None) and (view_kind != original_kind)
    should_rename = rename or kind_changed

    if should_rename:
        # If kind changed, delete the old JSON cache first.
        if kind_changed and not rename:
            old_cache = cache_path_for_image(image_path)
            if old_cache.exists():
                old_cache.unlink()
                print(f"  Deleted old cache: {old_cache.name}")
        new_path, _new_stem = _rename_to_kind(bg_root, image_path, view_kind)
        save_horizon(new_path, data)
        return new_path
    else:
        save_horizon(image_path, data)
        return image_path


# --- Mouse callback ---

class _State:
    """Mutable state shared between main loop and mouse callback."""
    ctrl_pts: list[tuple[int, int]]
    selected_idx: int
    curve: Optional[HorizonCurve]
    auto_curve: Optional[HorizonCurve]
    view_kind: str
    original_kind: str

    def __init__(self, auto_curve, view_kind, original_kind=None):
        self.ctrl_pts = []
        self.selected_idx = -1
        self.curve = None
        self.auto_curve = auto_curve
        self.view_kind = view_kind
        self.original_kind = original_kind if original_kind is not None else view_kind


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


# --- Interactive loop (shared by default, --review, --single) ---

def _interactive_loop(
    bg_root: Path,
    image_list: list[Path],
    start_idx: int,
    rename_on_save: bool,
    curve_band: int,
    ransac_iters: int,
    inlier_thresh: float,
    max_slope_deg: float,
) -> int:
    """Run the OpenCV interactive calibration loop over *image_list*.

    Parameters
    ----------
    rename_on_save : bool
        If True, S renames the file to side_*/top_* and writes cache.
        If False (review mode), S just overwrites the existing JSON cache
        (unless view kind was toggled, in which case a rename is forced).
    """
    total = len(image_list)
    if total == 0:
        print("No images to process.")
        return 0

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    idx = start_idx
    while 0 <= idx < total:
        image_path = image_list[idx]
        # Skip entries that were deleted (marked None).
        if image_path is None:
            idx += 1
            continue
        print(f"\n[{idx + 1}/{total}] {image_path.name}")

        # Load image.
        try:
            bg = load_background(image_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            idx += 1
            _save_progress(bg_root, image_path.stem, total)
            continue

        H, W = bg.shape[:2]

        # If in review mode, try to load existing curve from cache.
        cached_curve: Optional[HorizonCurve] = None
        cached_kind: Optional[str] = None
        if not rename_on_save:
            cached = load_horizon(image_path)
            if cached is not None and cached.horizon_curve is not None:
                cached_curve = HorizonCurve(**cached.horizon_curve)
                cached_kind = cached.kind

        # Classify / auto-detect.
        try:
            bg_view = classify_background(bg, return_info=True)
        except Exception as e:
            print(f"  classify_background error: {e}")
            bg_view = None

        auto_curve = bg_view.horizon_curve if bg_view and bg_view.kind == "side" else None
        view_kind = cached_kind or (bg_view.kind if bg_view else "top")
        original_kind = view_kind  # track for view-change detection

        # State for this image.
        state = _State(auto_curve, view_kind, original_kind=original_kind)
        # Pre-load cached curve as the manual curve in review mode.
        if cached_curve is not None:
            state.curve = cached_curve

        cv2.setMouseCallback("calibrate", _make_mouse_cb(state, W))

        while True:
            display = _render_overlay(
                bg, state.curve, state.auto_curve,
                state.ctrl_pts, state.selected_idx, state.view_kind,
            )
            n_pts = len(state.ctrl_pts)
            mode_label = "review" if not rename_on_save else "calibrate"
            del_hint = " X=delete" if not rename_on_save else ""
            title = (
                f"[{idx + 1}/{total}] {state.view_kind} | {mode_label} | "
                f"pts:{n_pts} | S=save D=skip T=toggle R=reset C=clear P=prev{del_hint} Q=quit"
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
                    rename=rename_on_save,
                    original_kind=state.original_kind,
                )
                print(f"  Saved -> {new_path.name}")
                # If kind changed and file was renamed, update the image list entry
                # so that subsequent prev/next navigation works correctly.
                if new_path != image_path:
                    image_list[idx] = new_path
                idx += 1
                _save_progress(bg_root, image_path.stem, total)
                break

            elif key == ord("d"):
                if rename_on_save:
                    skip_set = _load_skip_set(bg_root)
                    skip_set.add(image_path.stem)
                    _save_skip_set(bg_root, skip_set)
                    print(f"  Skipped -> {image_path.name} (added to _skip.json)")
                else:
                    print(f"  Skipped -> {image_path.name}")
                idx += 1
                _save_progress(bg_root, image_path.stem, total)
                break

            elif key == ord("t"):
                state.view_kind = "top" if state.view_kind == "side" else "side"
                print(f"  View toggled -> {state.view_kind} (original: {state.original_kind})")

            elif key == ord("x"):
                if rename_on_save:
                    # In calibration mode, x acts like d (skip).
                    skip_set = _load_skip_set(bg_root)
                    skip_set.add(image_path.stem)
                    _save_skip_set(bg_root, skip_set)
                    print(f"  Skipped -> {image_path.name} (added to _skip.json)")
                else:
                    # In review mode, x marks as deleted.
                    _mark_deleted(bg_root, image_path)
                    # Remove from image_list so prev/next skips it.
                    image_list[idx] = None
                idx += 1
                _save_progress(bg_root, image_path.stem, total)
                break

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
                    idx -= 1
                    _save_progress(bg_root, image_list[idx].stem, total)
                    print(f"  Going back to #{idx + 1}")
                    break

            # Nudge selected point with hjkl.
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


# --- Headless auto-save-all ---

def _auto_save_all(
    bg_root: Path,
    uncalibrated: list[Path],
    curve_band: int,
    ransac_iters: int,
    inlier_thresh: float,
    max_slope_deg: float,
) -> int:
    """Headless: auto-classify every image, save cache, rename. No GUI."""
    total = len(uncalibrated)
    n_side, n_top, n_error = 0, 0, 0
    for i, image_path in enumerate(uncalibrated):
        print(f"[{i + 1}/{total}] {image_path.name} ...", end=" ", flush=True)
        try:
            bg = load_background(image_path)
            bg_view = classify_background(bg, return_info=True)
        except Exception as e:
            print(f"ERROR: {e}")
            n_error += 1
            continue

        kind = bg_view.kind if bg_view else "top"
        curve = bg_view.horizon_curve if bg_view and bg_view.kind == "side" else None
        _save_current(bg_root, image_path, bg, kind, curve, curve, rename=True)
        if kind == "side":
            n_side += 1
        else:
            n_top += 1
        print(f"-> {kind}")

    print(f"\nDone: {n_side} side, {n_top} top, {n_error} errors  (total {total})")
    return 0


# --- Export report ---

def _export_report(bg_root: Path, csv_path: Path) -> int:
    """Write a CSV of all calibrated images' horizon data."""
    calibrated = _filter_calibrated(_image_stems(bg_root))
    if not calibrated:
        print("No calibrated images found.")
        return 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "kind", "a", "b", "c", "horizon_row", "rmse", "n_inliers", "width"])
        for p in sorted(calibrated):
            cached = load_horizon(p)
            if cached is None:
                w.writerow([p.stem, "?", "", "", "", "", "", "", ""])
                continue
            if cached.horizon_curve:
                c = cached.horizon_curve
                w.writerow([
                    p.stem, cached.kind,
                    f"{c['a']:.6f}", f"{c['b']:.6f}", f"{c['c']:.2f}",
                    cached.horizon_row or "",
                    f"{c['rmse']:.3f}", c['n_inliers'], c['width'],
                ])
            else:
                w.writerow([
                    p.stem, cached.kind, "", "", "",
                    cached.horizon_row or "", "", "", "",
                ])

    print(f"Exported {len(calibrated)} entries -> {csv_path}")
    return 0


# --- Main ---

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive horizon calibration tool for IR backgrounds"
    )
    ap.add_argument("--bg-root", default="bg", help="background images directory")
    ap.add_argument("--start", type=int, default=None, help="start at image index")

    # Operation modes.
    ap.add_argument("--auto-save-all", action="store_true",
                    help="headless: auto-classify and save all uncalibrated images")
    ap.add_argument("--review", action="store_true",
                    help="interactively review already-calibrated (renamed) images")
    ap.add_argument("--single", type=str, default=None,
                    help="interactively calibrate a single image by stem (e.g. IMG_1234)")
    ap.add_argument("--export-report", type=str, default=None,
                    help="export horizon data CSV for all calibrated images (provide path)")

    # Filtering.
    ap.add_argument("--view", choices=["side", "top", "both"], default="both",
                    help="filter by view kind in review mode (default: both)")

    # Adjustable fitting parameters.
    ap.add_argument("--curve-band", type=int, default=40,
                    help="search band half-width for curve fitting (default 40)")
    ap.add_argument("--ransac-iters", type=int, default=300,
                    help="RANSAC iterations for curve fitting (default 300)")
    ap.add_argument("--inlier-thresh", type=float, default=2.5,
                    help="pixel distance for inlier classification (default 2.5)")
    ap.add_argument("--max-slope-deg", type=float, default=18.0,
                    help="maximum horizon slope in degrees (default 18.0)")

    args = ap.parse_args()

    bg_root = Path(args.bg_root).resolve()
    if not bg_root.is_dir():
        print(f"ERROR: {bg_root} is not a directory")
        return 1

    all_paths = _image_stems(bg_root)
    if not all_paths:
        print(f"No images found in {bg_root}")
        return 1

    # --- Export report mode (non-interactive) ---
    if args.export_report:
        return _export_report(bg_root, Path(args.export_report))

    # --- Auto-save-all mode (non-interactive) ---
    if args.auto_save_all:
        skip_set = _load_skip_set(bg_root)
        uncalibrated = _filter_uncalibrated(all_paths, skip_set)
        if not uncalibrated:
            print("All images are already calibrated or skipped.")
            return 0
        print(f"Auto-saving {len(uncalibrated)} uncalibrated images ...")
        return _auto_save_all(
            bg_root, uncalibrated,
            args.curve_band, args.ransac_iters, args.inlier_thresh, args.max_slope_deg,
        )

    # --- Single-image mode ---
    if args.single is not None:
        stem = args.single
        matches = [p for p in all_paths if p.stem == stem]
        if not matches:
            print(f"ERROR: no image with stem '{stem}' found in {bg_root}")
            return 1
        # Determine if it's already calibrated.
        image_path = matches[0]
        is_cal = image_path.stem.startswith("side_") or image_path.stem.startswith("top_")
        print(f"Single-image mode: {image_path.name}  (calibrated={is_cal})")
        return _interactive_loop(
            bg_root, [image_path], 0, rename_on_save=not is_cal,
            curve_band=args.curve_band, ransac_iters=args.ransac_iters,
            inlier_thresh=args.inlier_thresh, max_slope_deg=args.max_slope_deg,
        )

    # --- Review mode ---
    if args.review:
        view_filter = args.view if args.view != "both" else None
        calibrated = _filter_calibrated(all_paths, view=view_filter)
        if not calibrated:
            vmsg = f" ({args.view}-view)" if args.view != "both" else ""
            print(f"No calibrated{vmsg} images found. Run default calibration first.")
            return 1

        if args.start is not None:
            start_idx = max(0, min(args.start, len(calibrated) - 1))
        else:
            last_stem = _load_progress(bg_root)
            start_idx = 0
            if last_stem is not None:
                for i, p in enumerate(calibrated):
                    if p.stem == last_stem:
                        start_idx = i
                        break

        print(f"Review mode: {len(calibrated)} calibrated images. Starting at #{start_idx + 1}.")
        print("Controls: S=save  D=skip  T=toggle view  R=reset  C=clear  P=prev  X=delete  Q=quit")
        return _interactive_loop(
            bg_root, calibrated, start_idx, rename_on_save=False,
            curve_band=args.curve_band, ransac_iters=args.ransac_iters,
            inlier_thresh=args.inlier_thresh, max_slope_deg=args.max_slope_deg,
        )

    # --- Default: interactive calibration of uncalibrated images ---
    skip_set = _load_skip_set(bg_root)
    uncalibrated = _filter_uncalibrated(all_paths, skip_set)
    total = len(uncalibrated)
    if total == 0:
        print("All images are already calibrated or skipped. Use --review to inspect.")
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
    print("Controls: S=save  D=skip  T=toggle view  R=reset  C=clear  P=prev  X=skip  Q=quit")
    print("Mouse:   left-click=add/select point  right-click=remove point  hjkl=nudge")
    return _interactive_loop(
        bg_root, uncalibrated, start_idx, rename_on_save=True,
        curve_band=args.curve_band, ransac_iters=args.ransac_iters,
        inlier_thresh=args.inlier_thresh, max_slope_deg=args.max_slope_deg,
    )


if __name__ == "__main__":
    sys.exit(main())
