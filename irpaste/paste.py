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
    composite: np.ndarray                 # uint8 (H, W) final composite
    bg: np.ndarray                        # uint8 (H, W) background
    target_patch: np.ndarray              # uint8 tight target crop (radiometrically matched)
    mask_patch: np.ndarray                # bool tight mask
    paste_xy: Tuple[int, int]             # top-left corner of target crop on bg
    method: BlendMethod
    bg_view: BackgroundView
    target_on_horizon: bool = False       # target straddles sim-image horizon
    sim_horizon_row: Optional[int] = None # detected horizon row in sim image (if any)
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
        img = np.clip((img - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    return img


def load_background(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return _to_gray_u8(img)


def target_patch_from_sample(sample: Sample, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
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
        patch = np.clip((rad - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
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


def choose_paste_site(
    bg: np.ndarray,
    bg_view: BackgroundView,
    patch_wh: tuple[int, int],
    rng: np.random.Generator,
    margin: int = 8,
    target_on_horizon: bool = False,
) -> tuple[int, int]:
    """Return the top-left (x, y) of the target patch on ``bg``.

    Placement rules
    ---------------
    * **Top-down bg** — anywhere inside the frame (random uniform with
      ``margin`` px border).
    * **Side-view bg with ``target_on_horizon=True``** — the ship's
      *bottom* lands on the bg horizon at the patch's horizontal centre,
      i.e. ``y_top = horizon(x_center) − ph``. When a quadratic
      ``HorizonCurve`` is available the horizon is evaluated per column;
      otherwise the straight-line ``horizon_row`` is used.
    * **Side-view bg with ``target_on_horizon=False``** — the ship is
      placed entirely below the horizon (in the ocean).

    A prow-exclusion column filter avoids saturated/dead-zone columns on
    side-view backgrounds.
    """
    H, W = bg.shape
    pw, ph = patch_wh
    x_lo, x_hi = margin, W - pw - margin
    y_lo, y_hi = margin, H - ph - margin
    if x_hi <= x_lo or y_hi <= y_lo:
        return max(0, (W - pw) // 2), max(0, (H - ph) // 2)

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
        x = int(rng.choice(valid_x))

        # Horizon at the patch's horizontal centre.
        hr_local = _horizon_at(bg_view, x + pw / 2.0, horizon_mid)

        if target_on_horizon:
            jitter = int(rng.integers(-2, 3))
            y = int(np.clip(hr_local - ph + jitter, y_lo, y_hi))
            return x, y

        # Ship entirely in ocean: strictly below horizon at that column.
        buffer = max(2, ph // 8)
        y_sea_lo = min(y_hi, max(y_lo, hr_local + buffer))
        if y_sea_lo >= y_hi:
            return x, int(np.clip(hr_local - ph, y_lo, y_hi))
        y = int(rng.integers(y_sea_lo, y_hi + 1))
        return x, y

    # -- Top-down bg — anywhere ---------------------------------------
    x = int(rng.integers(x_lo, x_hi + 1))
    y = int(rng.integers(y_lo, y_hi + 1))
    return x, y


# --------------------------------------------------------------------------- #
# Radiometric match
# --------------------------------------------------------------------------- #


def _bg_ring_stats(bg: np.ndarray, x: int, y: int, pw: int, ph: int, ring: int = 10) -> tuple[float, float]:
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

    # Contrast clamp (only rescale the target pixels, softly).
    if tgt_std > 3.0 * bg_std:
        scale = (2.0 * bg_std) / tgt_std
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
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    ksz = max(3, int(sigma * 6) | 1)
    a = cv2.GaussianBlur(m, (ksz, ksz), sigma)
    return a.astype(np.float32) / 255.0


def _blend_alpha(bg_patch: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = _feather_alpha(mask, dilate=1, sigma=1.2)
    out = a * fg_patch.astype(np.float32) + (1 - a) * bg_patch.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_poisson(bg: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray, paste_xy: tuple[int, int]) -> np.ndarray:
    """Poisson (``seamlessClone``, NORMAL_CLONE) on grayscale via 3-ch wrap."""
    x, y = paste_xy
    ph, pw = fg_patch.shape
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    fg_bgr = cv2.cvtColor(fg_patch, cv2.COLOR_GRAY2BGR)
    m = (mask.astype(np.uint8) * 255)
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


def _blend_laplacian(bg_patch: np.ndarray, fg_patch: np.ndarray, mask: np.ndarray, levels: int = 3) -> np.ndarray:
    # Work on matched-size patches only.
    a = _feather_alpha(mask, dilate=1, sigma=1.0)
    lp_a = _gaussian_pyramid(a, levels)
    lp_f = _laplacian_pyramid(fg_patch, levels)
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
# Top-level paste API
# --------------------------------------------------------------------------- #


def paste_target(
    sample: Sample,
    mask: np.ndarray,
    bg: np.ndarray,
    *,
    method: BlendMethod = "laplacian",
    bg_view: Optional[BackgroundView] = None,
    target_on_horizon: Optional[bool] = None,
    match_noise: bool = True,
    tv_smooth: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> PasteResult:
    """High-level: extract target, match radiometry, blend into bg.

    Default is ``method="laplacian"`` with ``tv_smooth=True`` — this
    combination had the lowest seam-gradient in the TV comparison
    (≈ 23 % below plain alpha) while preserving IR radiometric
    contrast. Use ``method="alpha"`` for the fastest path, or
    ``method="poisson"`` only when the source target has strong
    internal texture you want to preserve (Poisson's mean-retargeting
    will otherwise wash out low-texture IR ships).

    Placement is view-aware:

    * side-view bg + target on sim horizon → bottom flush with bg horizon.
    * side-view bg + target below sim horizon → entirely in the ocean.
    * top-down bg → anywhere inside the frame.

    ``target_on_horizon`` is auto-detected from the sim image when None.
    """
    if rng is None:
        rng = np.random.default_rng()
    if bg_view is None:
        bg_view = classify_background(bg, return_info=True)

    if target_on_horizon is None:
        on_horizon, sim_hr = detect_target_on_horizon(sample, mask)
    else:
        on_horizon = bool(target_on_horizon)
        _, sim_hr = detect_target_on_horizon(sample, mask)

    patch, m, _ = target_patch_from_sample(sample, mask)
    ph, pw = patch.shape

    # Downscale patch if it cannot fit on bg.
    H, W = bg.shape
    if ph >= H - 4 or pw >= W - 4:
        scale = min((W - 8) / pw, (H - 8) / ph)
        new_w = max(4, int(pw * scale))
        new_h = max(4, int(ph * scale))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ph, pw = patch.shape

    x, y = choose_paste_site(bg, bg_view, (pw, ph), rng, target_on_horizon=on_horizon)

    bg_med, bg_std = _bg_ring_stats(bg, x, y, pw, ph, ring=max(5, min(pw, ph) // 2))
    matched, radi_info = radiometric_match(patch, m, bg_med, bg_std, preserve_contrast=True)

    if method == "poisson":
        composite = _blend_poisson(bg, matched, m, (x, y))
    elif method == "alpha":
        bg_patch = bg[y : y + ph, x : x + pw].copy()
        blended = _blend_alpha(bg_patch, matched, m)
        composite = bg.copy()
        composite[y : y + ph, x : x + pw] = blended
    elif method == "laplacian":
        bg_patch = bg[y : y + ph, x : x + pw].copy()
        blended = _blend_laplacian(bg_patch, matched, m, levels=3)
        composite = bg.copy()
        composite[y : y + ph, x : x + pw] = blended
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Full-frame mask for downstream ops.
    full_mask = np.zeros(bg.shape, dtype=bool)
    full_mask[y : y + ph, x : x + pw] = m

    if match_noise:
        sigma_bg = _noise_sigma(bg)
        sigma_tgt = _noise_sigma(matched)
        composite = inject_matching_noise(composite, full_mask, sigma_bg, sigma_tgt, rng)
        radi_info["sigma_bg"] = sigma_bg
        radi_info["sigma_tgt"] = sigma_tgt

    if tv_smooth:
        composite = tv_boundary_smooth(composite, full_mask, band_px=2, weight=0.08)

    return PasteResult(
        composite=composite,
        bg=bg,
        target_patch=matched,
        mask_patch=m,
        paste_xy=(x, y),
        method=method,
        bg_view=bg_view,
        target_on_horizon=on_horizon,
        sim_horizon_row=sim_hr,
        radiometric=radi_info,
    )
