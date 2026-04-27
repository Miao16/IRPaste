"""Seamless paste of IR simulation targets onto real IR backgrounds.

Pipeline (see ``scripts/paste_demo.py`` for a CLI):

1. Load a simulation :class:`Sample` and its mask (via :func:`build_mask`).
2. Crop the target & mask to a tight bounding box.
3. Select a paste site on the background, respecting view type
   (side-view → near the horizon; top-down → low-texture water-like area).
4. Radiometrically match the target patch to the local background.
5. Blend using one of three methods:

   - ``poisson``   : :func:`cv2.seamlessClone` (NORMAL_CLONE), default.
   - ``alpha``     : feathered soft-alpha blend.
   - ``laplacian`` : 3-level Laplacian-pyramid multi-band blend.

6. Optionally inject Gaussian noise to match background sensor noise.
7. Optionally run a TV-L1 Chambolle smoother on a narrow band around the
   mask contour to erase any residual seam.

All images are handled as ``uint8`` grayscale end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from .io_utils import Sample
from .viewcls import BackgroundView, classify_background


BlendMethod = Literal["poisson", "alpha", "laplacian"]


# --------------------------------------------------------------------------- #
# Paste result
# --------------------------------------------------------------------------- #


@dataclass
class PasteResult:
    composite: np.ndarray  # uint8 (H, W) final composite
    bg: np.ndarray  # uint8 (H, W) background
    target_patch: np.ndarray  # uint8 tight target crop (radiometrically matched)
    mask_patch: np.ndarray  # bool tight mask
    paste_xy: Tuple[int, int]  # top-left corner of target crop on bg
    method: BlendMethod
    bg_view: BackgroundView
    target_on_horizon: bool = False  # target straddles sim-image horizon
    sim_horizon_row: Optional[int] = None  # detected horizon row in sim image (if any)
    radiometric: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip((img - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(
            np.uint8
        )
    return img


def load_background(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return _to_gray_u8(img)


def target_patch_from_sample(
    sample: Sample, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop a tight patch around the mask; return (patch_u8, mask_bool, bbox)."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("empty mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # Prefer the 8-bit preview for color-accurate pasting; fall back to
    # a normalised radiance crop if no preview is available.
    if sample.preview is not None:
        patch = sample.preview[y0:y1, x0:x1].copy()
    else:
        rad = sample.radiance[y0:y1, x0:x1]
        lo, hi = float(rad.min()), float(rad.max())
        patch = np.clip((rad - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(
            np.uint8
        )
    m = mask[y0:y1, x0:x1].astype(bool)
    return patch, m, (x0, y0, x1, y1)


def sim_preview_u8(sample: Sample) -> np.ndarray:
    """Return an 8-bit grayscale view of the simulation image (for
    visualisation and horizon detection)."""
    if sample.preview is not None:
        return sample.preview
    rad = sample.radiance
    lo, hi = np.percentile(rad, (1, 99))
    return np.clip((rad - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)


def detect_target_on_horizon(
    sample: Sample,
    mask: np.ndarray,
    min_step: int = 12,
) -> tuple[bool, Optional[int]]:
    """Does the target straddle the sea-sky horizon in the simulation image?

    Returns ``(on_horizon, sim_horizon_row)``. ``on_horizon`` is True when
    the simulation image has a clear horizon *and* the target mask
    bounding box straddles it (top above, bottom at/below) or is within
    a small band around it. When a quadratic ``HorizonCurve`` is fitted
    on the sim image, it is evaluated at the mask's horizontal centre.
    """
    sim = sim_preview_u8(sample)
    view = classify_background(sim, min_step=min_step, return_info=True)
    if view.kind != "side" or view.horizon_row is None:
        return False, None
    ys, xs = np.where(mask)
    if ys.size == 0:
        return False, view.horizon_row
    y_top = int(ys.min())
    y_bot = int(ys.max())
    x_mid = float((xs.min() + xs.max()) / 2.0)
    mh = max(1, y_bot - y_top)
    if view.horizon_curve is not None:
        hr = int(round(float(view.horizon_curve.y_at(x_mid))))
    else:
        hr = int(view.horizon_row)
    straddles = y_top <= hr <= y_bot + max(2, mh // 4)
    near = abs(((y_top + y_bot) // 2) - hr) <= max(4, mh)
    return bool(straddles or near), int(hr)


# --------------------------------------------------------------------------- #
# Paste-site selection
# --------------------------------------------------------------------------- #


def _variance_map(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    g = gray.astype(np.float32)
    mu = cv2.boxFilter(g, ddepth=-1, ksize=(ksize, ksize))
    mu2 = cv2.boxFilter(g * g, ddepth=-1, ksize=(ksize, ksize))
    return np.clip(mu2 - mu * mu, 0, None)


def _horizon_at(bg_view: BackgroundView, x: float, default: int) -> int:
    """Evaluate the bg horizon at column ``x`` (using curve when available)."""
    if bg_view.horizon_curve is not None:
        return int(round(float(bg_view.horizon_curve.y_at(float(x)))))
    return int(default)


def _center_biased_int(
    rng: np.random.Generator, lo: int, hi: int, bias: float = 2.0
) -> int:
    """Return a random integer in [lo, hi] biased toward the center.

    Uses a Beta(``bias``, ``bias``) distribution symmetric about 0.5 so the
    expected value is the centre of the interval.  ``bias=1`` = uniform;
    ``bias=2`` = mild centre bias; ``bias=3`` = stronger centre bias.
    """
    if hi <= lo:
        return lo
    t = float(rng.beta(bias, bias))
    return int(round(lo + t * (hi - lo)))


def choose_paste_site(
    bg: np.ndarray,
    bg_view: BackgroundView,
    patch_wh: tuple[int, int],
    rng: np.random.Generator,
    margin: int = 8,
    target_on_horizon: bool = False,
    occupied_mask: Optional[np.ndarray] = None,
    max_retry: int = 40,
) -> tuple[int, int]:
    """Return the top-left (x, y) of the target patch on ``bg``.

    Placement rules
    ---------------
    * **Top-down bg** — anywhere inside the frame with centre bias.
    * **Side-view bg with ``target_on_horizon=True``** — the ship's
      *bottom* lands on the bg horizon at the patch's horizontal centre,
      i.e. ``y_top = horizon(x_center) − ph``. When a quadratic
      ``HorizonCurve`` is available the horizon is evaluated per column;
      otherwise the straight-line ``horizon_row`` is used.
    * **Side-view bg with ``target_on_horizon=False``** — the ship is
      placed entirely below the horizon (in the ocean).

    A prow-exclusion column filter avoids saturated/dead-zone columns on
    side-view backgrounds.

    When ``occupied_mask`` is provided, placements that overlap already-
    occupied pixels (ship/hull areas) are penalised and skipped in favour
    of non-overlapping alternatives.

    Ships are biased toward the center of the frame (away from image
    borders) using a dynamic margin of 1/12 of the smaller dimension.
    """
    H, W = bg.shape
    pw, ph = patch_wh

    # Dynamic margin: keep ships away from boundaries.
    # At least 1/12 of the smaller dimension, clamped to [30, 80] px,
    # but never less than the caller-requested margin.
    dyn_margin = max(margin, min(80, max(30, min(H, W) // 12)))
    x_lo, x_hi = dyn_margin, W - pw - dyn_margin
    y_lo, y_hi = dyn_margin, H - ph - dyn_margin
    if x_hi <= x_lo or y_hi <= y_lo:
        return max(0, (W - pw) // 2), max(0, (H - ph) // 2)

    # Overlap penalty function.
    def _overlap_ok(xx: int, yy: int) -> bool:
        if occupied_mask is None or not occupied_mask.any():
            return True
        region = occupied_mask[yy : yy + ph, xx : xx + pw]
        if region.size == 0:
            return False
        return float(region.mean()) < 0.08  # < 8 % overlap is OK

    # -- Side-view bg --------------------------------------------------
    if bg_view.kind == "side" and bg_view.horizon_row is not None:
        horizon_mid = int(bg_view.horizon_row)

        # Column filter: use the highest point of the horizon curve as
        # the top of the "water band" so we don't lose columns where the
        # curve dips upwards.
        if bg_view.horizon_curve is not None:
            ys = bg_view.horizon_curve.y_at(np.arange(W))
            top_of_water = int(np.clip(np.floor(ys.min()), 0, H - 1))
        else:
            top_of_water = max(horizon_mid, 0)
        if top_of_water < H:
            var = _variance_map(bg[top_of_water:, :], 9).mean(axis=0)
        else:
            var = np.ones(W, dtype=np.float32)
        col_ok = var > np.percentile(var, 15)
        valid_x = np.arange(x_lo, x_hi + 1)
        valid_x = valid_x[col_ok[valid_x]] if col_ok.size == W else valid_x
        if valid_x.size == 0:
            valid_x = np.arange(x_lo, x_hi + 1)
        valid_x_sorted = np.sort(valid_x)

        for _ in range(max_retry):
            # Center-biased column selection.
            if len(valid_x_sorted) > 1:
                idx = _center_biased_int(rng, 0, len(valid_x_sorted) - 1, bias=2.0)
                x = int(valid_x_sorted[idx])
            else:
                x = int(valid_x_sorted[0])
            hr_local = _horizon_at(bg_view, x + pw / 2.0, horizon_mid)

            if target_on_horizon:
                # Ensure at least 1/3 of the ship patch sits below the horizon
                # so the hull appears in the sea rather than the whole patch
                # floating in the sky.
                min_sea_rows = max(4, ph // 3)
                jitter = int(rng.integers(-2, 3))
                y_ideal = hr_local - (ph - min_sea_rows) + jitter
                if y_ideal >= y_lo and y_ideal + ph <= H:
                    y_candidate = int(np.clip(y_ideal, y_lo, y_hi))
                    if _overlap_ok(x, y_candidate):
                        return x, y_candidate
                # Fall through to ocean placement.
            buffer = max(2, ph // 8)
            y_sea_lo = min(y_hi, max(y_lo, hr_local + buffer))
            if y_sea_lo < y_hi:
                y_candidate = _center_biased_int(rng, y_sea_lo, y_hi, bias=2.0)
                if _overlap_ok(x, y_candidate):
                    return x, y_candidate

        # No overlap-free site found after retries — accept best-effort
        # but still honour horizon constraints.
        if len(valid_x_sorted) > 1:
            idx = _center_biased_int(rng, 0, len(valid_x_sorted) - 1, bias=2.0)
            x = int(valid_x_sorted[idx])
        else:
            x = int(valid_x_sorted[0])
        hr_local = _horizon_at(bg_view, x + pw / 2.0, horizon_mid)
        if target_on_horizon:
            min_sea_rows = max(4, ph // 3)
            jitter = int(rng.integers(-2, 3))
            y_ideal = hr_local - (ph - min_sea_rows) + jitter
            y = int(np.clip(y_ideal, y_lo, y_hi))
            return x, y
        buffer = max(2, ph // 8)
        y_sea_lo = min(y_hi, max(y_lo, hr_local + buffer))
        if y_sea_lo < y_hi:
            y = _center_biased_int(rng, y_sea_lo, y_hi, bias=2.0)
        else:
            y = y_sea_lo
        return x, max(y_lo, y)

    # -- Top-down bg — center-biased, try to avoid overlap, then accept best-effort
    for _ in range(max_retry):
        x = _center_biased_int(rng, x_lo, x_hi, bias=2.0)
        y = _center_biased_int(rng, y_lo, y_hi, bias=2.0)
        if _overlap_ok(x, y):
            return x, y
    # Fallback — still prefer no overlap, but accept if unavoidable.
    for _ in range(max_retry):
        x = _center_biased_int(rng, x_lo, x_hi, bias=2.0)
        y = _center_biased_int(rng, y_lo, y_hi, bias=2.0)
        if occupied_mask is None or not occupied_mask.any():
            return x, y
        region = occupied_mask[y : y + ph, x : x + pw]
        if region.size == 0:
            return x, y
        if float(region.mean()) < 0.25:  # <= 25% overlap as last resort
            return x, y
    x = _center_biased_int(rng, x_lo, x_hi, bias=2.0)
    y = _center_biased_int(rng, y_lo, y_hi, bias=2.0)
    return x, y


# --------------------------------------------------------------------------- #
# Radiometric match
# --------------------------------------------------------------------------- #


def _bg_ring_stats(
    bg: np.ndarray, x: int, y: int, pw: int, ph: int, ring: int = 10
) -> tuple[float, float]:
    H, W = bg.shape
    x0 = max(0, x - ring)
    y0 = max(0, y - ring)
    x1 = min(W, x + pw + ring)
    y1 = min(H, y + ph + ring)
    outer = bg[y0:y1, x0:x1]
    # Inner region: the paste bbox, excluded from stats.
    mask = np.ones_like(outer, dtype=bool)
    mask[
        max(0, y - y0) : min(outer.shape[0], y - y0 + ph),
        max(0, x - x0) : min(outer.shape[1], x - x0 + pw),
    ] = False
    pix = outer[mask]
    if pix.size < 20:
        pix = bg.reshape(-1)
    return float(np.median(pix)), float(np.std(pix) + 1e-3)


def radiometric_match(
    patch: np.ndarray,
    mask: np.ndarray,
    bg_med: float,
    bg_std: float,
    preserve_contrast: bool = True,
) -> tuple[np.ndarray, dict]:
    """Shift (and optionally scale) target radiance to fit the local bg.

    * Target's masked mean is moved to ``bg_med + Δ``, where Δ is the
      target-minus-bg mean offset *before* shift (preserves thermal
      polarity — a hot ship stays hotter than water).
    * Un-masked (surrounding) pixels in the patch are moved by the same
      global shift so the patch remains self-consistent.
    * Optional contrast clamp: if the target's intra-mask std exceeds
      ``3 × bg_std``, rescale to ``2 × bg_std`` around the mask mean.
    """
    p = patch.astype(np.float32)
    tgt_vals = p[mask]
    if tgt_vals.size == 0:
        return patch.copy(), {}
    tgt_mean = float(tgt_vals.mean())
    tgt_std = float(tgt_vals.std() + 1e-3)

    # Determine polarity: if target mean > local surround mean, keep it hotter.
    surround = p[~mask] if (~mask).any() else p.reshape(-1)
    sur_mean = float(np.median(surround))
    delta = tgt_mean - sur_mean  # signed preserved contrast
    new_mean = bg_med + (delta if preserve_contrast else 0.0)

    shift = new_mean - tgt_mean
    p = p + shift

    # Contrast clamp: only rescale if target is extremely over-contrasted
    # relative to the local background.  The old threshold (3×) fired too
    # readily on bright MWIR ships against calm-sea backgrounds, flattening
    # the target’s internal structure into a featureless blob.  The new
    # threshold (8×) reserves the clamp for obvious simulation artefacts.
    if tgt_std > 8.0 * bg_std:
        scale = (4.0 * bg_std) / tgt_std
        p_masked = p.copy()
        p_masked[mask] = new_mean + (p[mask] - new_mean) * scale
        p = p_masked

    p = np.clip(p, 0, 255).astype(np.uint8)
    info = {
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "sur_mean": sur_mean,
        "bg_med": bg_med,
        "bg_std": bg_std,
        "delta": delta,
        "shift": shift,
    }
    return p, info


# --------------------------------------------------------------------------- #
# Blending primitives
# --------------------------------------------------------------------------- #


def _feather_alpha(mask: np.ndarray, dilate: int = 1, sigma: float = 1.2) -> np.ndarray:
    """Soft alpha: tiny dilate so mast pixels survive, then Gaussian blur.

    Default sigma is small (~1.2) to keep a crisp silhouette while still
    killing aliasing at the boundary. Callers that want heavier feathering
    pass a larger ``sigma``.
    """
    m = mask.astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
        )
        m = cv2.dilate(m, k)
    ksz = max(3, int(sigma * 6) | 1)
    a = cv2.GaussianBlur(m, (ksz, ksz), sigma)
    return a.astype(np.float32) / 255.0


def _composite_fg(
    fg_patch: np.ndarray, bg_patch: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return a composite where only mask pixels come from *fg_patch*;
    non-mask pixels are taken from *bg_patch*.

    This prevents the simulation-source background texture (the area
    around the ship in the synthetic image) from bleeding into the
    Laplacian pyramid or the alpha weighting, which is the root cause
    of the visible halo / shadow around pasted ships.
    """
    out = bg_patch.astype(np.float32).copy()
    out[mask] = fg_patch.astype(np.float32)[mask]
    return out


def _blend_alpha(
    bg_patch: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    # Zero out simulation background outside the mask before blending so
    # synthetic-background halo / shadow cannot appear.
    fg_clean = _composite_fg(fg_patch, bg_patch, mask)
    a = _feather_alpha(mask, dilate=0, sigma=0.6)
    out = a * fg_clean + (1 - a) * bg_patch.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_poisson(
    bg: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray, paste_xy: tuple[int, int]
) -> np.ndarray:
    """Poisson (``seamlessClone``, NORMAL_CLONE) on grayscale via 3-ch wrap."""
    x, y = paste_xy
    ph, pw = fg_patch.shape
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    fg_bgr = cv2.cvtColor(fg_patch, cv2.COLOR_GRAY2BGR)
    m = mask.astype(np.uint8) * 255
    # seamlessClone requires the clone center AND the mask not to touch
    # the image border; clamp the center to avoid OpenCV assertion.
    H, W = bg.shape
    cx = int(np.clip(x + pw // 2, pw // 2 + 2, W - pw // 2 - 2))
    cy = int(np.clip(y + ph // 2, ph // 2 + 2, H - ph // 2 - 2))
    # If the mask touches the patch border, erode it once; seamlessClone
    # treats white-to-border as invalid.
    if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        if m.sum() == 0:
            # Fall back to alpha blend if erosion wiped the mask.
            patch_bg = bg[y : y + ph, x : x + pw].copy()
            blended = _blend_alpha(patch_bg, fg_patch, mask)
            out = bg.copy()
            out[y : y + ph, x : x + pw] = blended
            return out
    out = cv2.seamlessClone(fg_bgr, bg_bgr, m, (cx, cy), cv2.NORMAL_CLONE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def _gaussian_pyramid(img: np.ndarray, n: int) -> list[np.ndarray]:
    pyr = [img.astype(np.float32)]
    for _ in range(n):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplacian_pyramid(img: np.ndarray, n: int) -> list[np.ndarray]:
    gp = _gaussian_pyramid(img, n)
    lp = []
    for i in range(n):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def _blend_laplacian(
    bg_patch: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray, levels: int = 3
) -> np.ndarray:
    # Replace simulation background with real bg before building the pyramid.
    # Without this, non-mask pixels from the synthetic image (different texture)
    # bleed across the mask boundary through Gaussian down-sampling, creating
    # a visible halo around the ship silhouette.
    fg_clean = _composite_fg(fg_patch, bg_patch, mask)
    a = _feather_alpha(mask, dilate=0, sigma=0.6)
    lp_a = _gaussian_pyramid(a, levels)
    lp_f = _laplacian_pyramid(fg_clean, levels)
    lp_b = _laplacian_pyramid(bg_patch, levels)
    blended = []
    for la, lf, lb in zip(lp_a, lp_f, lp_b):
        if la.shape != lf.shape:
            la = cv2.resize(la, (lf.shape[1], lf.shape[0]))
        blended.append(la * lf + (1 - la) * lb)
    out = blended[-1]
    for i in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        out = out + blended[i]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Post-paste adaptive boundary blur
# --------------------------------------------------------------------------- #


def _adaptive_boundary_blur(
    composite: np.ndarray,
    full_mask: np.ndarray,
) -> np.ndarray:
    """Smooth the mask boundary ring after pasting to eliminate seam artifacts.

    Only the dilated boundary ring (not the ship interior) is blurred, so
    internal ship details are preserved.  The blur sigma scales with the
    square-root of the ship area so tiny targets are left untouched while
    large ships get a wider transition.

    ``sigma`` schedule:
    * ship_size = √(mask_area)
    * sigma     = clip(ship_size / 80, 0.5, 3.0)
    * skip entirely when ship_size < 15 px (very small target)
    """
    mask_area = int(full_mask.sum())
    ship_size = float(np.sqrt(max(mask_area, 1)))
    if ship_size < 15.0:
        return composite

    sigma = float(np.clip(ship_size / 80.0, 0.5, 3.0))
    ring_w = max(2, int(sigma * 2.0))

    m_u8 = full_mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_w + 1, 2 * ring_w + 1))
    # Build a SMOOTH radial weight so the blend has no hard rectangular
    # boundary.  A binary cv2.dilate followed by a Gaussian blur creates
    # a continuous falloff from 1.0 at the ship silhouette to 0.0 beyond
    # the dilation ring, preventing the visible "frame" artifact that
    # appeared when blurred and non-blurred regions met at a hard edge.
    dilated = cv2.dilate(m_u8, k)
    blur_k = max(3, ring_w * 4 + 1) | 1  # odd kernel; wider than ring_w
    ring_f = cv2.GaussianBlur(
        dilated.astype(np.float32) / 255.0,
        (blur_k, blur_k),
        max(1.0, ring_w * 1.5),
    )

    ksz = max(3, int(sigma * 4) | 1)
    img_f = composite.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (ksz, ksz), sigma)
    # Blend: boundary zone uses blurred; away from mask uses original
    result = img_f * (1.0 - ring_f) + blurred * ring_f
    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Noise match
# --------------------------------------------------------------------------- #


def _noise_sigma(patch: np.ndarray) -> float:
    """Estimate per-pixel noise σ from MAD of Laplacian (Immerkær)."""
    lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F, ksize=3)
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return mad * 1.4826 / np.sqrt(6.0)


def inject_matching_noise(
    img: np.ndarray,
    mask: np.ndarray,
    sigma_bg: float,
    sigma_tgt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    extra = max(0.0, (sigma_bg * sigma_bg - sigma_tgt * sigma_tgt))
    if extra <= 1e-6:
        return img
    s = float(np.sqrt(extra))
    noise = rng.normal(0, s, img.shape).astype(np.float32)
    out = img.astype(np.float32)
    out[mask] += noise[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Boundary TV-L1 smoother (Chambolle projected-gradient)
# --------------------------------------------------------------------------- #


def tv_boundary_smooth(
    img: np.ndarray,
    mask: np.ndarray,
    band_px: int = 2,
    weight: float = 0.08,
    n_iter: int = 50,  # kept for API compat; unused (skimage picks its own)
) -> np.ndarray:
    """TV-L1 denoise restricted to the **outer** ring around the mask.

    We only modify pixels in ``dilate(mask, band_px) & ~mask``. The
    target interior is never touched (previous versions used a
    ``dilate − erode`` ring which ate small IR targets). The actual
    TV solver is ``skimage.restoration.denoise_tv_chambolle`` on a
    tight crop containing the ring, for a fair, stable, boundary-safe
    comparison.
    """
    from skimage.restoration import denoise_tv_chambolle  # local import

    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return img
    r = max(1, int(band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    outer = (cv2.dilate(m, kernel).astype(bool)) & (~m.astype(bool))
    if not outer.any():
        return img

    H, W = img.shape
    ys, xs = np.where(outer)
    pad = 6
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)

    crop = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    smooth = denoise_tv_chambolle(crop, weight=float(weight), max_num_iter=200)

    out = img.copy()
    patch_outer = outer[y0:y1, x0:x1]
    out_crop = out[y0:y1, x0:x1].astype(np.float32)
    out_crop[patch_outer] = np.clip(smooth[patch_outer] * 255.0, 0, 255)
    out[y0:y1, x0:x1] = out_crop.astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Background zoom-in augmentation
# --------------------------------------------------------------------------- #


def augment_background(
    bg: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (1.0, 1.4),
    n_candidates: int = 10,
) -> np.ndarray:
    """Randomly zoom into *bg* by a factor drawn from *scale_range*, then
    crop back to the original spatial dimensions using a **smart crop
    selection** strategy.

    Instead of a purely random crop offset, we sample *n_candidates*
    candidate windows and score each by:

    1. **Border cleanliness** — penalise crops whose border strip is very
       different from the interior (black-edge / blown-out artefacts that
       appear when the zoom window reaches the physical image boundary).
    2. **Texture richness** — prefer windows with healthy variance so the
       composited ship has a realistic clutter background.
    3. **Horizon stability** (side-view) — prefer windows whose horizon
       row stays close to the original horizon row (avoids cropping to
       pure sky or pure sea).

    A scale of 1.0 returns a copy unchanged.  Values above ~1.5 risk
    cutting out the sea-sky horizon on side-view images, so the
    recommended upper bound is 1.4.
    """
    H, W = bg.shape
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    if scale <= 1.0 + 1e-4:
        return bg.copy()
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    scaled = cv2.resize(bg, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    max_y_off = max(0, new_H - H)
    max_x_off = max(0, new_W - W)

    # Pre-compute the global texture level to normalise scores.
    global_std = float(np.std(bg)) + 1e-3

    # Estimate original horizon row (cheap: look at vertical gradient).
    orig_horizon: Optional[int] = None
    row_mean = bg.mean(axis=1)
    grad1d = np.abs(np.diff(row_mean.astype(np.float32)))
    y_lo_h = max(4, int(H * 0.25))
    y_hi_h = int(H * 0.75)
    if y_hi_h > y_lo_h + 4:
        sub_g = grad1d[y_lo_h:y_hi_h]
        thr = float(np.percentile(sub_g, 55))
        strong_idx = np.where(sub_g >= thr)[0]
        if strong_idx.size:
            orig_horizon = y_lo_h + int(strong_idx[-1])

    cand_y = rng.integers(0, max_y_off + 1, size=n_candidates).tolist()
    cand_x = rng.integers(0, max_x_off + 1, size=n_candidates).tolist()
    # Always include centre crop as a safe fallback.
    cand_y[0] = max_y_off // 2
    cand_x[0] = max_x_off // 2

    best_score = -np.inf
    best_y, best_x = int(cand_y[0]), int(cand_x[0])

    for y_off, x_off in zip(cand_y, cand_x):
        y_off, x_off = int(y_off), int(x_off)
        crop = scaled[y_off : y_off + H, x_off : x_off + W]

        # 1. Texture richness score (0–1).
        crop_std = float(np.std(crop))
        texture_score = crop_std / (crop_std + global_std)

        # 2. Border cleanliness: compare border strip median vs interior.
        border = 8
        if border * 2 < H and border * 2 < W:
            border_px = np.concatenate(
                [
                    crop[:border, :].ravel(),
                    crop[-border:, :].ravel(),
                    crop[:, :border].ravel(),
                    crop[:, -border:].ravel(),
                ]
            )
            interior = crop[border:-border, border:-border].ravel()
            border_med = float(np.median(border_px))
            int_med = float(np.median(interior))
            int_range = float(np.ptp(interior)) + 1e-3
            border_penalty = abs(border_med - int_med) / int_range
        else:
            border_penalty = 0.0

        # 3. Horizon stability: for side-view images prefer crops where
        #    the horizon row stays near its original position.
        if orig_horizon is not None:
            # After zooming and cropping, the original row `orig_horizon`
            # maps to row `(orig_horizon * scale) - y_off` in the crop.
            mapped_hr = orig_horizon * scale - y_off
            horizon_dist = abs(mapped_hr - orig_horizon) / max(H, 1)
            horizon_penalty = float(np.clip(horizon_dist, 0.0, 1.0))
        else:
            horizon_penalty = 0.0

        score = texture_score - 0.4 * border_penalty - 0.3 * horizon_penalty
        if score > best_score:
            best_score = score
            best_y, best_x = y_off, x_off

    return scaled[best_y : best_y + H, best_x : best_x + W].copy()


# --------------------------------------------------------------------------- #
# Ship principal-axis / horizon alignment
# --------------------------------------------------------------------------- #


def get_mask_principal_angle(mask: np.ndarray) -> float:
    """Return the angle (degrees) of the **principal (long) axis** of *mask*
    via PCA on pixel coordinates.

    Convention: 0° = horizontal; positive values tilt clockwise.
    Output range: ``[-90, 90)``.
    """
    ys, xs = np.where(mask)
    if xs.size < 5:
        return 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = (pts_c.T @ pts_c) / max(len(pts) - 1, 1)
    # eigh returns eigenvalues in ascending order; last eigenvector = principal
    _, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, -1]
    angle = float(np.degrees(np.arctan2(vy, vx)))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def get_horizon_tangent_angle(bg_view: BackgroundView, x_center: float) -> float:
    """Tangent angle (degrees) of the horizon at column *x_center*.

    For a quadratic horizon ``y = a·x² + b·x + c`` the tangent slope is
    ``dy/dx = 2a·x + b``.  A straight (or absent) horizon returns 0°.
    """
    if bg_view.kind != "side" or bg_view.horizon_curve is None:
        return 0.0
    c = bg_view.horizon_curve
    dydx = 2.0 * c.a * x_center + c.b
    return float(np.degrees(np.arctan(dydx)))


def rotate_patch_to_angle(
    patch: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate *patch* and *mask* by *angle_deg* degrees (positive = clockwise),
    expanding the canvas so no content is clipped.

    Returns ``(rotated_patch, rotated_mask_bool)``.
    """
    if abs(angle_deg) < 0.3:
        return patch.copy(), mask.astype(bool).copy()
    h, w = patch.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # cv2 convention: positive angle = counter-clockwise → negate for CW
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a) + 2
    new_h = int(h * cos_a + w * sin_a) + 2
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    rot_patch = cv2.warpAffine(
        patch, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0
    )
    rot_mask_f = cv2.warpAffine(
        mask.astype(np.float32),
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    rot_mask = rot_mask_f > 0.5
    return rot_patch, rot_mask


# --------------------------------------------------------------------------- #
# Size-dependent ship downscale
# --------------------------------------------------------------------------- #


def _ship_scale_from_area(
    mask_area: int,
    scale_range: tuple[float, float],
    ref_area: float = 3000.0,
    sensitivity: float = 1.2,
    rng: "np.random.Generator | None" = None,
) -> float:
    """Compute a size-dependent ship downscale factor.

    Large ships (area > *ref_area*) receive a scale closer to
    ``scale_range[0]`` (more downscaling).  Small ships (area < *ref_area*)
    receive a scale closer to ``scale_range[1]`` (less downscaling, keeping
    them visible).

    The mapping uses a sigmoid over the log-area ratio so it behaves
    smoothly across orders of magnitude.  A small random jitter (±4 % of
    the range) preserves natural variation.
    """
    lo, hi = scale_range
    if lo >= hi:
        return float(lo)

    area = max(mask_area, 1)
    log_ratio = np.log(area / ref_area)
    # t → 0 for large ships, t → 1 for small ships
    t = 1.0 / (1.0 + np.exp(sensitivity * log_ratio))
    base = lo + t * (hi - lo)

    # Small random jitter (±4 % of range) for natural variation.
    if rng is not None:
        jitter = float(rng.uniform(-0.04, 0.04)) * (hi - lo)
        base = float(np.clip(base + jitter, lo, hi))

    return float(base)


# --------------------------------------------------------------------------- #
# paste_patch — low-level entry point for pre-tight-cropped targets
# --------------------------------------------------------------------------- #


def paste_patch(
    patch: np.ndarray,
    mask: np.ndarray,
    bg: np.ndarray,
    *,
    method: BlendMethod = "laplacian",
    bg_view: BackgroundView,
    target_on_horizon: bool = False,
    match_noise: bool = True,
    tv_smooth: bool = False,
    rng: Optional[np.random.Generator] = None,
    align_to_horizon: bool = False,
    ship_scale_range: tuple[float, float] = (0.55, 0.90),
    max_bbox_px: Optional[int] = None,
    occupied_mask: Optional[np.ndarray] = None,
) -> PasteResult:
    """Paste a pre-cropped (patch, mask) pair onto a background.

    Unlike :func:`paste_target`, this function does NOT load samples,
    extract masks, augment backgrounds, or classify views.  It receives
    an already-tight-cropped target patch + mask and a fully-prepared
    background, and handles the rest: scaling, rotation, placement,
    radiometric matching, blending, and post-processing.

    Parameters
    ----------
    patch : np.ndarray
        uint8 tight-cropped target patch.
    mask : np.ndarray
        bool tight mask (same shape as patch).
    bg : np.ndarray
        uint8 background (already augmented if desired).
    bg_view : BackgroundView
        Pre-computed background view classification.
    target_on_horizon : bool
        Whether the target straddles the sim horizon.
    Other parameters : same as :func:`paste_target`.
    """
    if rng is None:
        rng = np.random.default_rng()

    notes: list[str] = []
    H, W = bg.shape
    ph, pw = patch.shape

    # --- Downscale patch if it cannot fit on bg ---
    if ph >= H - 4 or pw >= W - 4:
        scale = min((W - 8) / pw, (H - 8) / ph)
        new_w = max(4, int(pw * scale))
        new_h = max(4, int(ph * scale))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        ph, pw = patch.shape

    # --- Optional principal-axis / horizon alignment ---
    if align_to_horizon and bg_view.kind == "side":
        principal_angle = get_mask_principal_angle(mask)
        horizon_angle = get_horizon_tangent_angle(bg_view, W / 2.0)
        rotation = horizon_angle - principal_angle
        if abs(rotation) >= 0.3:
            patch, mask = rotate_patch_to_angle(patch, mask, rotation)
            ph, pw = patch.shape
            notes.append(
                f"axis-align: principal={principal_angle:.1f}deg "
                f"horizon={horizon_angle:.1f}deg rot={rotation:.1f}deg"
            )

    # --- Optional ship downscale (size-dependent) ---
    # Large ships → smaller multiplier (more downscaling).
    # Small ships → larger multiplier (less downscaling, stay visible).
    ship_scale = _ship_scale_from_area(
        int(mask.sum()), ship_scale_range, rng=rng,
    )
    if ship_scale < 0.99:
        new_pw = max(4, int(round(pw * ship_scale)))
        new_ph = max(4, int(round(ph * ship_scale)))
        patch = cv2.resize(patch, (new_pw, new_ph), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask.astype(np.uint8), (new_pw, new_ph), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        ph, pw = patch.shape
        notes.append(f"ship scale={ship_scale:.2f} -> {pw}x{ph}")

    # --- Re-tight-crop after all transforms ---
    _ys, _xs = np.where(mask)
    if _ys.size > 0:
        _y0c = int(_ys.min())
        _y1c = int(_ys.max()) + 1
        _x0c = int(_xs.min())
        _x1c = int(_xs.max()) + 1
        patch = patch[_y0c:_y1c, _x0c:_x1c]
        mask = mask[_y0c:_y1c, _x0c:_x1c]
        ph, pw = patch.shape

    # --- Optional bbox max-side clamp ---
    # Ensure the longest side of the tight bounding box does not exceed
    # *max_bbox_px* pixels (e.g. 125).  Applied after tight-crop so the
    # measurement reflects the actual ship silhouette.
    if max_bbox_px is not None and max(pw, ph) > max_bbox_px:
        clamp_scale = max_bbox_px / max(pw, ph)
        new_pw = max(4, int(round(pw * clamp_scale)))
        new_ph = max(4, int(round(ph * clamp_scale)))
        patch = cv2.resize(patch, (new_pw, new_ph), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask.astype(np.uint8), (new_pw, new_ph), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        ph, pw = patch.shape
        notes.append(f"bbox clamp max={max_bbox_px}px -> {pw}x{ph}")

    x, y = choose_paste_site(
        bg, bg_view, (pw, ph), rng, target_on_horizon=target_on_horizon,
        occupied_mask=occupied_mask,
    )

    # --- Erode mask by 1 px to shed bright anti-aliasing ring pixels ---
    if int(mask.sum()) >= 30:
        _k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _m_core = cv2.erode(mask.astype(np.uint8), _k_erode).astype(bool)
        if int(_m_core.sum()) >= 10:
            mask = _m_core

    bg_med, bg_std = _bg_ring_stats(bg, x, y, pw, ph, ring=max(5, min(pw, ph) // 2))
    matched, radi_info = radiometric_match(
        patch, mask, bg_med, bg_std, preserve_contrast=True
    )

    if method == "poisson":
        composite = _blend_poisson(bg, matched, mask, (x, y))
    elif method == "alpha":
        bg_patch = bg[y : y + ph, x : x + pw].copy()
        blended = _blend_alpha(bg_patch, matched, mask)
        composite = bg.copy()
        composite[y : y + ph, x : x + pw] = blended
    elif method == "laplacian":
        bg_patch = bg[y : y + ph, x : x + pw].copy()
        blended = _blend_laplacian(bg_patch, matched, mask, levels=3)
        composite = bg.copy()
        composite[y : y + ph, x : x + pw] = blended
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Full-frame mask for downstream ops.
    full_mask = np.zeros(bg.shape, dtype=bool)
    full_mask[y : y + ph, x : x + pw] = mask

    composite = _adaptive_boundary_blur(composite, full_mask)

    if match_noise:
        sigma_bg = _noise_sigma(bg)
        sigma_tgt = _noise_sigma(matched)
        composite = inject_matching_noise(
            composite, full_mask, sigma_bg, sigma_tgt, rng
        )
        radi_info["sigma_bg"] = sigma_bg
        radi_info["sigma_tgt"] = sigma_tgt

    if tv_smooth:
        composite = tv_boundary_smooth(composite, full_mask, band_px=2, weight=0.08)

    return PasteResult(
        composite=composite,
        bg=bg,
        target_patch=matched,
        mask_patch=mask,
        paste_xy=(x, y),
        method=method,
        bg_view=bg_view,
        target_on_horizon=target_on_horizon,
        sim_horizon_row=None,
        radiometric=radi_info,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# paste_target — high-level entry point (sample → mask → bg)
# --------------------------------------------------------------------------- #


def paste_target(
    sample: Sample,
    mask: np.ndarray,
    bg: np.ndarray,
    *,
    method: BlendMethod = "laplacian",
    bg_view: Optional[BackgroundView] = None,
    bg_path: Optional["str | Path"] = None,
    target_on_horizon: Optional[bool] = None,
    match_noise: bool = True,
    tv_smooth: bool = False,
    rng: Optional[np.random.Generator] = None,
    # --- augmentation / alignment ---
    augment_bg: bool = False,
    bg_scale_range: tuple[float, float] = (1.0, 1.4),
    align_to_horizon: bool = False,
    # --- ship size control ---
    ship_scale_range: tuple[float, float] = (0.55, 0.90),
    max_bbox_px: Optional[int] = None,
    # --- multi-ship overlap avoidance ---
    occupied_mask: Optional[np.ndarray] = None,
) -> PasteResult:
    """High-level: extract target, match radiometry, blend into bg.

    Default is ``method="laplacian"``. For the lowest seam gradient,
    enable ``tv_smooth=True`` (≈ 23 % below plain alpha) while preserving
    IR radiometric contrast. Use ``method="alpha"`` for the fastest path,
    or ``method="poisson"`` only when the source target has strong
    internal texture you want to preserve (Poisson's mean-retargeting
    will otherwise wash out low-texture IR ships).

    Placement is view-aware:

    * side-view bg + target on sim horizon → bottom flush with bg horizon.
    * side-view bg + target below sim horizon → entirely in the ocean.
    * top-down bg → anywhere inside the frame.

    ``target_on_horizon`` is auto-detected from the sim image when None.

    Additional augmentation parameters
    ------------------------------------
    augment_bg
        If ``True``, the background is randomly zoomed-in by a factor
        drawn uniformly from *bg_scale_range* and then smart-cropped
        back to its original resolution (prefers clean, horizon-stable
        crop windows).  ``bg_view`` is re-computed on the augmented
        frame.  Simulates different sensor zoom levels.
    bg_scale_range
        ``(lo, hi)`` for the zoom factor; recommended ``(1.0, 1.4)``.
    align_to_horizon
        If ``True`` **and** the background is side-view, the target
        patch is rotated so its principal (long) axis becomes parallel
        to the local background horizon.  This enforces the physical
        constraint that a ship floating on water has its keel horizontal
        with respect to the sea-sky line.  Has no effect on top-down
        backgrounds (ship heading is arbitrary in nadir view).
    ship_scale_range
        ``(lo, hi)`` for the **size-dependent** ship downscale factor.
        *Large* ships are scaled closer to ``lo`` (more downscaling);
        *small* ships are scaled closer to ``hi`` (kept closer to
        original size so they remain visible).  Default ``(0.55, 0.90)``.
        Set to ``(1.0, 1.0)`` to disable.  A small random jitter (±4 %
        of range) is added for natural variation.  Ships on the horizon
        are placed with their bottom edge aligned to the horizon after
        scaling.
    max_bbox_px
        If set, the longest side of the ship's tight bounding box is
        clamped to this pixel count (e.g. ``125``).  Applied after the
        size-dependent scaling and tight-crop, so small ships already
        below the limit are unaffected.
    """
    if rng is None:
        rng = np.random.default_rng()

    notes: list[str] = []

    # --- Optional background augmentation ---
    if augment_bg:
        bg = augment_background(bg, rng, scale_range=bg_scale_range)
        bg_view = None
        notes.append(f"bg augmented scale in {bg_scale_range}")

    if bg_view is None:
        _fname = Path(bg_path).name if bg_path is not None else None
        bg_view = classify_background(bg, return_info=True, filename=_fname)

    if target_on_horizon is None:
        on_horizon, sim_hr = detect_target_on_horizon(sample, mask)
    else:
        on_horizon = bool(target_on_horizon)
        _, sim_hr = detect_target_on_horizon(sample, mask)

    patch, m, _ = target_patch_from_sample(sample, mask)

    pr = paste_patch(
        patch,
        m,
        bg,
        method=method,
        bg_view=bg_view,
        target_on_horizon=on_horizon,
        match_noise=match_noise,
        tv_smooth=tv_smooth,
        rng=rng,
        align_to_horizon=align_to_horizon,
        ship_scale_range=ship_scale_range,
        max_bbox_px=max_bbox_px,
        occupied_mask=occupied_mask,
    )
    pr.sim_horizon_row = sim_hr
    pr.notes = notes + pr.notes
    return pr
