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
    width: int  # image width the fit was performed on

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
    ransac_iters: int = 300,
    inlier_thresh: float = 2.5,
    rng: Optional[np.random.Generator] = None,
    max_slope_deg: float = 18.0,
) -> Optional[HorizonCurve]:
    """Fit a quadratic horizon ``y = a·x² + b·x + c`` to an IR background.

    Upgraded algorithm (v2):
    -------------------------
    1. **Bilateral pre-filter** — denoises while preserving strong
       edges, giving cleaner column peak detection even in rainy / foggy
       images.
    2. **Multi-scale Sobel fusion** — combine fine (ksize=3) and coarse
       (ksize=5) Sobel-Y responses by taking their element-wise maximum.
       This increases sensitivity to both sharp horizon lines and diffuse
       intensity gradients caused by fog or haze.
    3. **Spatial coherence filter** — reject column peaks that deviate
       too far from their neighbours (outlier peaks caused by ships,
       land, or dead pixels).
    4. **Guided column sampling** — during RANSAC triplet selection,
       samples are spread across the full image width (three disjoint
       thirds) instead of being picked uniformly at random.  This avoids
       degenerate near-collinear triplets and raises the success rate for
       curved horizons.
    5. **Iterative refinement (LO-RANSAC step)** — after finding the
       best RANSAC model, a single weighted least-squares refinement
       pass on all inliers further reduces the RMSE without extra
       iterations.
    6. **Confidence guard** — models are rejected when the residuals of
       inlier columns show a bimodal distribution (ship or land boundary
       mistaken for horizon).

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
        Number of random triplet trials. 300 gives a good balance for
        ``W ≈ 512``.
    inlier_thresh : float
        Pixel distance for inlier classification.
    rng : np.random.Generator, optional
        Seeded RNG for reproducibility.
    max_slope_deg : float
        Maximum allowed horizon slope (degrees from horizontal).
        Stops the horizon from tracking steep land or cloud boundaries.
        Default 18.0° matches the Hough angle filter in the classic
        pipeline.

    Returns
    -------
    HorizonCurve or None
        ``None`` if the fit lacks support (``< max(20, W/20)`` inliers).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    H, W = gray.shape

    # 1. Bilateral pre-filter: preserve edge sharpness while removing noise.
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    g_filt = cv2.bilateralFilter(g8, d=7, sigmaColor=20, sigmaSpace=7).astype(
        np.float32
    )

    # 2. Multi-scale Sobel-Y fusion.
    sy3 = cv2.Sobel(g_filt, cv2.CV_32F, 0, 1, ksize=3)
    sy5 = cv2.Sobel(g_filt, cv2.CV_32F, 0, 1, ksize=5)

    # Normalise each scale to [0, 1] before fusing.
    def _norm(a: np.ndarray) -> np.ndarray:
        a = np.abs(a)
        mx = float(a.max()) + 1e-6
        return a / mx

    abs_sy = _norm(sy3) * 0.5 + _norm(sy5) * 0.5
    # Light horizontal blur for stable per-column peaks.
    abs_sy = cv2.GaussianBlur(abs_sy, (11, 3), 0)

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
    thresh = np.percentile(col_peak_vals, 25)
    keep = col_peak_vals > thresh
    xs_all = np.arange(W)[keep].astype(np.float64)
    ys_all = col_ys[keep].astype(np.float64)
    if xs_all.size < max(30, W // 15):
        return None

    # --- v3: spatial coherence filter ---
    # Reject column peaks that deviate too far from their local median
    # (outliers caused by ships, land, dead pixels).  Use a sliding
    # window of ~30 columns and drop peaks > 3 MAD from the window median.
    if xs_all.size >= 60:
        _win = max(15, int(xs_all.size * 0.06))
        _coherent = np.ones(xs_all.size, dtype=bool)
        for i in range(xs_all.size):
            lo = max(0, i - _win)
            hi = min(xs_all.size, i + _win + 1)
            _local_med = float(np.median(ys_all[lo:hi]))
            _local_mad = float(
                np.median(np.abs(ys_all[lo:hi] - _local_med))
            ) + 1e-3
            if abs(ys_all[i] - _local_med) > 3.0 * _local_mad:
                _coherent[i] = False
        xs_all = xs_all[_coherent]
        ys_all = ys_all[_coherent]
        if xs_all.size < max(30, W // 15):
            return None

    # 3. Guided RANSAC: sample one point from each of three equal-width
    #    horizontal thirds so triplets span the full image width.
    min_inliers = max(20, W // 20)
    third = max(1, len(xs_all) // 3)
    best_inl: Optional[np.ndarray] = None
    best_n = 0
    best_poly: Optional[np.ndarray] = None
    n = xs_all.size

    # Bucket column indices into three spatial thirds.
    buckets = [
        np.where(xs_all < W / 3)[0],
        np.where((xs_all >= W / 3) & (xs_all < 2 * W / 3))[0],
        np.where(xs_all >= 2 * W / 3)[0],
    ]
    all_buckets_valid = all(b.size > 0 for b in buckets)

    for _ in range(ransac_iters):
        if all_buckets_valid:
            try:
                idx = np.array(
                    [
                        int(rng.choice(buckets[0])),
                        int(rng.choice(buckets[1])),
                        int(rng.choice(buckets[2])),
                    ]
                )
            except Exception:
                idx = rng.choice(n, 3, replace=False)
        else:
            idx = rng.choice(n, 3, replace=False)

        x3, y3 = xs_all[idx], ys_all[idx]
        if np.ptp(x3) < max(8.0, W * 0.04):
            continue
        try:
            p = np.polyfit(x3, y3, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        # Slope check: derivative dydx = 2*a*x + b at image centre
        # should be within max_slope_deg.
        dydx = 2.0 * p[0] * (W / 2.0) + p[1]
        if abs(dydx) > np.tan(np.radians(max_slope_deg)):
            continue
        resid = np.abs(np.polyval(p, xs_all) - ys_all)
        inl = resid < inlier_thresh
        k = int(inl.sum())
        if k > best_n:
            best_n = k
            best_inl = inl
            best_poly = p

    if best_inl is None or best_n < min_inliers:
        return None

    # 4. LO-RANSAC refinement: weighted LS on inliers (weight ∝ 1/|resid|).
    xi, yi = xs_all[best_inl], ys_all[best_inl]
    pred0 = np.polyval(best_poly, xi)
    resid_inl = np.abs(pred0 - yi) + 1e-4
    weights = 1.0 / resid_inl
    try:
        p_ref = np.polyfit(xi, yi, 2, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        p_ref = best_poly
    resid_ref = np.abs(np.polyval(p_ref, xs_all) - ys_all)
    inl_ref = resid_ref < inlier_thresh
    if int(inl_ref.sum()) >= max(min_inliers, int(best_n * 0.85)):
        best_poly = p_ref
        best_inl = inl_ref

    # 5. Confidence guard: reject if inlier y-residuals are bimodal
    #    (suggests a ship hull or land feature rather than a true horizon).
    xi_f, yi_f = xs_all[best_inl], ys_all[best_inl]
    pred_f = np.polyval(best_poly, xi_f)
    signed_resid = yi_f - pred_f
    sr_mad = float(np.median(np.abs(signed_resid - np.median(signed_resid))))
    sr_range = float(np.ptp(signed_resid))
    if sr_range > 10.0 * (sr_mad + 1e-3):
        return None

    pred_all = np.polyval(best_poly, xi_f)
    rmse = float(np.sqrt(np.mean((yi_f - pred_all) ** 2)))

    return HorizonCurve(
        a=float(best_poly[0]),
        b=float(best_poly[1]),
        c=float(best_poly[2]),
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
    score: float  # strength of the best horizon candidate (higher = stronger)
    variance_ratio: (
        float  # (above+below) variance / overall variance — lower = more bimodal
    )
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
    filename: Optional[str] = None,
):
    """Classify an IR background as side-view or top-down.

    Algorithm (v2)
    --------------
    1. Bilateral-filter the image for robust row-mean estimation in
       rainy / foggy conditions.
    2. Compute per-row mean from the central 60 % of columns on the
       filtered image and pick the **bottom-most** strong gradient peak
       in rows 25–75 % as the candidate horizon.
    3. Compute the **step** (sky–sea intensity jump) and a secondary
       **histogram bimodality** score: the row-mean histogram should
       have two distinct modes for a side-view image.
    4. Side-view iff ``step ≥ min_step`` and the sky strip is not
       excessively noisy.
    5. If side-view and ``fit_curve``, fit the upgraded quadratic horizon
       via the multi-scale RANSAC/LO-RANSAC procedure in
       :func:`fit_horizon_curve`.

    v3 improvements
    ---------------
    6. **Multi-peak verification**: when the bottom-most gradient peak
       fails the step test, try the next-strongest peaks in ascending
       order to handle double horizons (e.g., fog layer above true
       sea-sky boundary).
    7. **Adaptive band sizing**: scale sky/sea bands with image height
       so that small images don't receive oversized bands that cross
       the horizon.
    """
    g = gray.astype(np.float32)
    if g.ndim != 2:
        raise ValueError("classify_background expects a 2-D array")
    H, W = g.shape

    # Bilateral filter for robustness in foggy/rainy scenes.
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    g_filt = cv2.bilateralFilter(g8, d=9, sigmaColor=25, sigmaSpace=9).astype(
        np.float32
    )

    cw = int(W * 0.6)
    x0 = (W - cw) // 2
    strip = g_filt[:, x0 : x0 + cw]

    row_mean = strip.mean(axis=1)
    row_mean_s = _smooth_1d(row_mean, np.array([1, 2, 4, 2, 1], dtype=np.float32))
    full_std = float(strip.std()) + 1e-3

    y_lo = max(6, int(H * 0.22))
    y_hi = int(H * 0.78)
    grad = np.abs(np.diff(row_mean_s))
    if grad.size == 0 or y_hi <= y_lo:
        return BackgroundView("top", None, 0.0, 1.0) if return_info else "top"

    # Collect all gradient peaks sorted by strength, then try in order.
    sub = grad[y_lo:y_hi]
    pct65 = float(np.percentile(sub, 65))
    strong = np.where(sub >= pct65)[0]
    # Sort strong indices by gradient magnitude (descending) so we try
    # the most prominent peaks first.
    if strong.size > 0:
        strong_sorted = strong[np.argsort(sub[strong])[::-1]]
    else:
        strong_sorted = np.array([int(np.argmax(sub))])

    # Adaptive band sizing.
    _sky_band = min(sky_band, int(H * 0.15))
    _sea_band = min(sea_band, int(H * 0.10))

    # Secondary bimodality test on the full column strip's row-mean
    # histogram: a genuine sky/sea transition should give two modes.
    hist, _ = np.histogram(row_mean_s, bins=32)
    hist_f = hist.astype(np.float32)
    hist_f /= hist_f.sum() + 1e-6
    if hist_f.max() > 0:
        peak_idx = np.argmax(hist_f)
        left_min = float(hist_f[:peak_idx].min()) if peak_idx > 0 else float(hist_f[0])
        right_min = (
            float(hist_f[peak_idx + 1 :].min())
            if peak_idx < len(hist_f) - 1
            else float(hist_f[-1])
        )
        valley = min(left_min, right_min)
        bimodal_score = 1.0 - valley / (float(hist_f.max()) + 1e-6)
    else:
        bimodal_score = 0.0

    is_side = False
    best_cand_y: Optional[int] = None

    for cand_offset in strong_sorted:
        cand_y = y_lo + int(cand_offset)

        sky = strip[max(0, cand_y - _sky_band) : cand_y]
        sea = strip[cand_y : cand_y + _sea_band]
        if sky.size < 3 or sea.size < 3:
            continue

        sky_med = float(np.median(sky))
        sea_med = float(np.median(sea))
        sky_std = float(sky.std())
        step = abs(sky_med - sea_med)

        candidate_ok = step >= min_step and sky_std <= 2.0 * step
        if not candidate_ok and bimodal_score > 0.7 and step > min_step * 0.7:
            candidate_ok = True

        if candidate_ok:
            is_side = True
            best_cand_y = cand_y
            break

    # Filename-based override removed; view classification is now driven
    # by horizon_cache.json files (see scripts/calibrate_horizon.py).

    curve: Optional[HorizonCurve] = None
    horizon_row: Optional[int] = int(best_cand_y) if (is_side and best_cand_y is not None) else None
    if is_side and fit_curve and best_cand_y is not None:
        curve = fit_horizon_curve(g, y_hint=best_cand_y, band=max(25, int(H * 0.08)))
        if curve is not None:
            horizon_row = int(round(float(curve.y_at(W / 2.0))))

    return (
        BackgroundView(
            kind="side" if is_side else "top",
            horizon_row=horizon_row,
            score=float(step) if best_cand_y is not None else 0.0,
            variance_ratio=float(sky_std / full_std) if best_cand_y is not None else 1.0,
            horizon_curve=curve,
        )
        if return_info
        else ("side" if is_side else "top")
    )
