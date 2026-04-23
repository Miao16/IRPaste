"""View-angle classification for backgrounds and simulation targets.

Two entry points:

* ``classify_target(xml_path)`` — read sensor ``pitch`` from the XML
  ``<imageSensor>`` element. Pitch ``<= -80°`` → ``"top"`` (nadir-ish).
  Anything else → ``"side"``.

* ``classify_background(gray)`` — given an IR background image (uint8 or
  float, single-channel), return ``"side"`` if a clean sea/sky horizon
  line can be detected, else ``"top"``. In side-view mode the horizon
  is modelled as a quadratic curve ``y = a·x² + b·x + c`` fitted via
  RANSAC over per-column Sobel-Y peaks (robust to ships, clouds, and
  dead zones). The straight-line ``horizon_row`` field is retained for
  back-compat and equals the curve evaluated at the image centre.

The background classifier is intentionally self-contained so it can be
reused outside the segmentation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import xml.etree.ElementTree as ET

import cv2
import numpy as np


ViewKind = Literal["side", "top"]


# --------------------------------------------------------------------------- #
# Target view classifier (XML pitch)
# --------------------------------------------------------------------------- #


def classify_target_pitch(pitch_deg: float, nadir_thresh: float = -80.0) -> ViewKind:
    """Pitch-only classification."""
    return "top" if pitch_deg <= nadir_thresh else "side"


def classify_target(xml_path: str | Path, nadir_thresh: float = -80.0) -> ViewKind:
    """Read sensor pitch from an IR simulation XML and classify view."""
    root = ET.parse(Path(xml_path)).getroot()
    sensor = root.find("imageSensor")
    if sensor is None:
        raise ValueError(f"{xml_path}: missing <imageSensor>")
    pitch = float(sensor.get("pitch", "0"))
    return classify_target_pitch(pitch, nadir_thresh=nadir_thresh)


# --------------------------------------------------------------------------- #
# Horizon curve — quadratic ``y = a·x² + b·x + c``
# --------------------------------------------------------------------------- #


@dataclass
class HorizonCurve:
    """Quadratic model of the sea-sky horizon in image coordinates."""

    a: float
    b: float
    c: float
    rmse: float
    n_inliers: int
    width: int      # image width the fit was performed on

    def y_at(self, x: float | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.a * x * x + self.b * x + self.c

    def polyline(self, n: int = 128) -> np.ndarray:
        """Return an ``(n, 2)`` int32 array of polyline points for drawing."""
        xs = np.linspace(0, self.width - 1, n)
        ys = self.y_at(xs)
        pts = np.stack([xs, ys], axis=1)
        return pts.round().astype(np.int32)


def fit_horizon_curve(
    gray: np.ndarray,
    y_hint: Optional[int] = None,
    band: int = 40,
    ransac_iters: int = 200,
    inlier_thresh: float = 2.5,
    rng: Optional[np.random.Generator] = None,
) -> Optional[HorizonCurve]:
    """Fit a quadratic horizon ``y = a·x² + b·x + c`` to an IR background.

    Parameters
    ----------
    gray : (H, W) uint8 or float
        Single-channel background.
    y_hint : int, optional
        Initial guess for the horizon row (from the coarse classifier).
        Columns are searched within ``[y_hint-band, y_hint+band]``. If
        ``None`` the whole image is searched.
    band : int
        Half-width (rows) of the candidate search strip.
    ransac_iters : int
        Number of random triplet trials. 200 is ample for ``W ≈ 512``.
    inlier_thresh : float
        Pixel distance for inlier classification.
    rng : np.random.Generator, optional
        Seeded RNG for reproducibility.

    Returns
    -------
    HorizonCurve or None
        ``None`` if the fit lacks support (``< max(20, W/20)`` inliers).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    g = gray.astype(np.float32)
    H, W = g.shape

    # Sobel-Y with a mild horizontal smooth so per-column peaks are stable.
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sy = cv2.GaussianBlur(sy, (11, 3), 0)
    abs_sy = np.abs(sy)

    if y_hint is None:
        y_lo, y_hi = 2, max(3, int(H * 0.9))
    else:
        y_lo = max(2, int(y_hint) - band)
        y_hi = min(H - 2, int(y_hint) + band)
    if y_hi - y_lo < 3:
        return None

    strip = abs_sy[y_lo:y_hi, :]
    col_peak_offsets = np.argmax(strip, axis=0)
    col_peak_vals = strip.max(axis=0)
    col_ys = col_peak_offsets + y_lo

    # Drop the weakest columns (ships, overexposed patches, dead bands).
    thresh = np.percentile(col_peak_vals, 20)
    keep = col_peak_vals > thresh
    xs_all = np.arange(W)[keep].astype(np.float64)
    ys_all = col_ys[keep].astype(np.float64)
    if xs_all.size < max(30, W // 15):
        return None

    # RANSAC — 3 points determine a parabola exactly.
    min_inliers = max(20, W // 20)
    best_inl: Optional[np.ndarray] = None
    best_n = 0
    n = xs_all.size
    for _ in range(ransac_iters):
        idx = rng.choice(n, 3, replace=False)
        x3, y3 = xs_all[idx], ys_all[idx]
        # Skip degenerate triples.
        if np.ptp(x3) < max(8.0, W * 0.04):
            continue
        try:
            p = np.polyfit(x3, y3, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        resid = np.abs(np.polyval(p, xs_all) - ys_all)
        inl = resid < inlier_thresh
        k = int(inl.sum())
        if k > best_n:
            best_n = k
            best_inl = inl
    if best_inl is None or best_n < min_inliers:
        return None

    # LS refine on inliers.
    p = np.polyfit(xs_all[best_inl], ys_all[best_inl], 2)
    pred = np.polyval(p, xs_all[best_inl])
    rmse = float(np.sqrt(np.mean((ys_all[best_inl] - pred) ** 2)))

    return HorizonCurve(
        a=float(p[0]),
        b=float(p[1]),
        c=float(p[2]),
        rmse=rmse,
        n_inliers=int(best_inl.sum()),
        width=int(W),
    )


# --------------------------------------------------------------------------- #
# Background view classifier (horizon detector)
# --------------------------------------------------------------------------- #


@dataclass
class BackgroundView:
    kind: ViewKind
    horizon_row: Optional[int]
    score: float           # strength of the best horizon candidate (higher = stronger)
    variance_ratio: float  # (above+below) variance / overall variance — lower = more bimodal
    horizon_curve: Optional[HorizonCurve] = None


def _smooth_1d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.astype(np.float32)
    k = k / k.sum()
    return np.convolve(x, k, mode="same")


def classify_background(
    gray: np.ndarray,
    min_step: float = 15.0,
    sky_band: int = 40,
    sea_band: int = 20,
    return_info: bool = False,
    fit_curve: bool = True,
):
    """Classify an IR background as side-view or top-down.

    Algorithm
    ---------
    1. Within the central 60 % of columns, smooth the per-row mean and
       pick the candidate horizon row as the argmax of ``|d/dy row_mean|``
       restricted to the upper 65 % of the image (skipping the prow
       band at the bottom).
    2. Compute the **step**:
       ``step = |median(sky_band_above_candidate) − median(sea_band_below)|``.
       `sky_band` and `sea_band` are thin strips (default 40 rows above,
       20 rows below), so the test is purely a local sky↔sea contrast
       measurement that does not need the rest of the scene.
    3. Side-view iff ``step ≥ min_step``. A secondary sanity check
       rejects the rare case where the sky band itself is extremely
       non-uniform (``sky_std > 2·step``) — that typically signals a
       cloud/terrain boundary rather than a sky/sea horizon.
    4. If side-view and ``fit_curve``, fit a quadratic horizon via
       :func:`fit_horizon_curve` (RANSAC over per-column Sobel-Y peaks).
       ``horizon_row`` is then the curve evaluated at ``W/2``; the full
       curve is exposed on ``BackgroundView.horizon_curve``.

    The ``step`` magnitude on the calibration set (45 IR backgrounds)
    gives a clean split at 15: clear side-view scenes score 17–85
    while overhead/top-down scenes score 0–11.
    """
    g = gray.astype(np.float32)
    if g.ndim != 2:
        raise ValueError("classify_background expects a 2-D array")
    H, W = g.shape

    cw = int(W * 0.6)
    x0 = (W - cw) // 2
    strip = g[:, x0 : x0 + cw]

    row_mean = strip.mean(axis=1)
    row_mean_s = _smooth_1d(row_mean, np.array([1, 2, 4, 2, 1], dtype=np.float32))
    full_std = float(strip.std()) + 1e-3

    y_lo = max(6, int(H * 0.05))
    y_hi = int(H * 0.65)
    grad = np.abs(np.diff(row_mean_s))
    if grad.size == 0 or y_hi <= y_lo:
        return BackgroundView("top", None, 0.0, 1.0) if return_info else "top"
    cand_y = int(y_lo + np.argmax(grad[y_lo:y_hi]))

    sky = strip[max(0, cand_y - sky_band) : cand_y]
    sea = strip[cand_y : cand_y + sea_band]
    if sky.size == 0 or sea.size == 0:
        return BackgroundView("top", None, 0.0, 1.0) if return_info else "top"

    sky_med = float(np.median(sky))
    sea_med = float(np.median(sea))
    sky_std = float(sky.std())
    step = abs(sky_med - sea_med)

    is_side = step >= min_step and sky_std <= 2.0 * step

    curve: Optional[HorizonCurve] = None
    horizon_row: Optional[int] = int(cand_y) if is_side else None
    if is_side and fit_curve:
        curve = fit_horizon_curve(g, y_hint=cand_y, band=max(25, int(H * 0.08)))
        if curve is not None:
            horizon_row = int(round(float(curve.y_at(W / 2.0))))

    return (
        BackgroundView(
            kind="side" if is_side else "top",
            horizon_row=horizon_row,
            score=float(step),
            variance_ratio=float(sky_std / full_std),
            horizon_curve=curve,
        )
        if return_info
        else ("side" if is_side else "top")
    )
