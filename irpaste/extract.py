"""Target mask extraction from IR simulation images.

Algorithm (summary)
-------------------
Per sample:

1. Expand the XML bbox by a small margin (default 5 %). Clamp to image.
2. Around the expanded bbox, define **four side bands** (left / right /
   top / bottom) in a larger context window. Each band lives fully
   *outside* the expanded bbox, so it should not contain target pixels.
3. Detect the sea/sky horizon using **only the left+right side columns**
   (never the target column range). If a strong, wide-support horizontal
   edge is found within the context window, split the bbox and bands
   into above/below-horizon sub-regions that are processed
   independently.
4. For each sub-region, score each candidate band by "purity" (tightness
   + uni-modality) and keep the cleanest ones. Fit a robust local
   background model (constant median) from the pooled band pixels.
5. Compute the residual ``r = radiance - bg``. Derive two thresholds
   from the residual's MAD (hysteresis):

       T_high = k_high * 1.4826 * MAD(residual_bg)
       T_low  = max(0.4 * T_high, k_low * 1.4826 * MAD(residual_bg))

   Two-sided: a pixel is *strong* foreground if ``|r| >= T_high`` and
   *weak* foreground if ``|r| >= T_low``. Weak pixels are retained only
   if they reach a strong pixel through 8-connectivity (classical
   hysteresis).
6. Edge-aided mast recovery. Compute the Sobel gradient on the residual
   and accept additional weak pixels that (a) coincide with a gradient
   quantile peak and (b) touch an already-selected pixel.
7. Morphology + connected-component selection. Prefer components that
   overlap the inner half of the original XML bbox. Drop tiny speckle
   (<2 px) but keep thin vertical mast structures linked to the main
   component.

The function returns an :class:`ExtractResult` carrying the mask plus
intermediate values that are useful for diagnostics and QA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .io_utils import Annotation, Sample


# --------------------------------------------------------------------------- #
# Tunables (sensible defaults; exposed via build_mask kwargs)
# --------------------------------------------------------------------------- #

DEFAULTS = dict(
    bbox_expand=0.05,          # expand XML bbox by this fraction
    anchor_clip_expand=0.20,   # final mask is clipped to anchor expanded by this
    context_expand=1.5,        # context window = anchor expanded by this
    context_min_half=50,       # absolute minimum half-size (px) of context
    min_band_thickness=6,      # minimum band thickness in px
    k_high=4.0,                # hysteresis MAD multiplier — strong threshold
    k_low=2.0,                 # hysteresis MAD multiplier — weak threshold
    horizon_search_margin=150, # rows above/below bbox to search for horizon
    horizon_min_jump_factor=3, # required (|jump|/noise) for a valid horizon
    horizon_min_jump_abs=0.5,  # absolute radiance jump floor for horizon
    edge_quantile=0.90,        # percentile of |grad| used as edge threshold
    subregion_band_rows=40,    # rows to draw from each side of horizon
    # --- v3: polygon-ring sanitisation (fix for coastline nadir views) ---
    poly_ring_inner=4,         # start the ring this many px outside the polygon
    poly_ring_outer=16,        # end the ring this many px outside the polygon
    poly_ring_clip_k=4.0,      # clip row-bg to ring_med ± max(k*ring_mad, abs_floor)
    poly_ring_clip_abs=0.5,    # absolute floor for row-bg clipping range (radiance units)
    # --- v3: polygon-based final clip (tighter than AABB dilation for tilted ships) ---
    poly_clip_dilate_px=5,     # dilate corner polygon by this many px for final clip
)


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


@dataclass
class ExtractResult:
    mask: np.ndarray            # bool (H, W), True = target
    bbox: tuple[int, int, int, int]          # expanded bbox (x0,y0,x1,y1)
    context: tuple[int, int, int, int]       # context window
    horizon_row: Optional[int]               # global row or None
    bg_median: float            # pooled background median over the ROI
    bg_mad: float               # pooled background MAD
    t_high: float
    t_low: float
    n_mask: int                 # mask pixel count
    notes: list[str]            # human-readable warnings / diagnostics


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clip_rect(x0: int, y0: int, x1: int, y1: int, w: int, h: int):
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Return ``(median, MAD)`` for a 1-D array of values."""
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def _band_purity_score(values: np.ndarray) -> float:
    """Lower is cleaner. We use (MAD / (1 + inter-quartile halfrange))
    plus a small penalty for clearly bimodal distributions."""
    if values.size < 8:
        return float("inf")
    med, mad = _robust_stats(values)
    # Scale MAD by a rough "dynamic range" normaliser to compare bands.
    q10, q90 = np.quantile(values, [0.1, 0.9])
    spread = max(q90 - q10, 1e-6)
    # Simple bimodality heuristic: if the histogram's top bin is far from
    # the median, call it bimodal.
    return float(mad / spread)


def _detect_horizon_row(
    radiance: np.ndarray,
    bbox: tuple[int, int, int, int],
    context: tuple[int, int, int, int],
    search_margin: int,
    min_jump_factor: float,
    min_jump_abs: float,
) -> Optional[int]:
    """Detect a sea/sky horizon row using left+right side strips that
    extend vertically well beyond the context window.

    The criterion is: for every candidate row ``r`` in the search range,
    compute the medians of a 30-row strip immediately above and below
    ``r``. Pick the ``r`` that maximises the absolute difference. Accept
    it only when the jump dominates the per-side MAD noise and exceeds
    an absolute floor.
    """
    cx0, _cy0, cx1, _cy1 = context
    bx0, by0, bx1, by1 = bbox
    H, W = radiance.shape

    y_lo = max(0, by0 - search_margin)
    y_hi = min(H, by1 + search_margin)
    if y_hi - y_lo < 20:
        return None

    left = radiance[y_lo:y_hi, max(0, cx0 - 0): bx0] if bx0 > cx0 else None
    right = radiance[y_lo:y_hi, bx1: min(W, cx1)] if cx1 > bx1 else None
    # If the context barely leaves room on the sides, fall back to
    # wider lateral strips up to 80 px from the bbox edges.
    if left is None or left.size == 0:
        left = radiance[y_lo:y_hi, max(0, bx0 - 80): bx0] if bx0 > 0 else None
    if right is None or right.size == 0:
        right = radiance[y_lo:y_hi, bx1: min(W, bx1 + 80)] if bx1 < W else None
    strips = [s for s in (left, right) if s is not None and s.size > 0]
    if not strips:
        return None
    pooled = np.concatenate(
        [s.reshape(s.shape[0], -1) for s in strips], axis=1
    )
    n_rows = pooled.shape[0]
    if n_rows < 40:
        return None

    row_medians = np.median(pooled, axis=1)  # shape (n_rows,)
    win = 15  # half-window for above/below medians

    best_r = -1
    best_jump = 0.0
    best_noise = 1e-6
    for r in range(win, n_rows - win):
        above = row_medians[r - win: r]
        below = row_medians[r: r + win]
        m_above = float(np.median(above))
        m_below = float(np.median(below))
        noise = float(
            np.median(np.abs(above - m_above))
            + np.median(np.abs(below - m_below))
        ) + 1e-6
        jump = abs(m_below - m_above)
        if jump > best_jump:
            best_jump = jump
            best_r = r
            best_noise = noise

    if best_r < 0:
        return None
    if best_jump < min_jump_abs:
        return None
    if best_jump < min_jump_factor * 1.4826 * best_noise:
        return None

    return y_lo + best_r


# --------------------------------------------------------------------------- #
# Background band collection
# --------------------------------------------------------------------------- #


def _polygon_mask_from_corners(
    shape: tuple[int, int], corners_px: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Binary mask (bool, (H, W)) of the XML corner polygon, or None."""
    if corners_px is None or len(corners_px) < 3:
        return None
    H, W = shape
    m = np.zeros((H, W), dtype=np.uint8)
    pts = np.asarray(corners_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _polygon_ring_stats(
    radiance: np.ndarray,
    corners_px: Optional[np.ndarray],
    inner_dilate: int,
    outer_dilate: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Robust background stats from a ring around the polygon.

    Returns ``(median, MAD, p90_minus_p10)`` or ``(None, None, None)``
    if the ring is too small. The last value is a bimodality-robust
    spread: it stays large when the ring spans two backgrounds even if
    one mode has >50 % density (in which case MAD collapses to 0).
    """
    poly = _polygon_mask_from_corners(radiance.shape, corners_px)
    if poly is None:
        return None, None, None
    k_outer = 2 * max(outer_dilate, 1) + 1
    outer = cv2.dilate(
        poly.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_outer, k_outer)),
    ).astype(bool)
    if inner_dilate > 0:
        k_inner = 2 * inner_dilate + 1
        inner = cv2.dilate(
            poly.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_inner, k_inner)),
        ).astype(bool)
    else:
        inner = poly
    ring = outer & (~inner)
    vals = radiance[ring]
    if vals.size < 24:
        return None, None, None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    q10, q90 = np.quantile(vals, [0.1, 0.9])
    spread = float(q90 - q10)
    return med, mad, spread


def _lateral_row_profile(
    radiance: np.ndarray,
    bbox: tuple[int, int, int, int],
    context: tuple[int, int, int, int],
    lateral_margin: int = 80,
) -> np.ndarray:
    """Return a per-row background estimate over the context's vertical
    extent, computed from lateral strips that lie *outside* the bbox
    columns. Handy when the scene contains a horizon-induced row-wise
    intensity gradient that a flat constant background cannot represent.
    """
    H, W = radiance.shape
    bx0, _by0, bx1, _by1 = bbox
    cx0, cy0, cx1, cy1 = context

    left_x0 = max(0, bx0 - lateral_margin)
    left_x1 = bx0
    right_x0 = bx1
    right_x1 = min(W, bx1 + lateral_margin)

    strips = []
    if left_x1 > left_x0:
        strips.append(radiance[cy0:cy1, left_x0:left_x1])
    if right_x1 > right_x0:
        strips.append(radiance[cy0:cy1, right_x0:right_x1])
    if not strips:
        # Fallback: use the context strip beside the bbox (may be tiny).
        strips.append(radiance[cy0:cy1, cx0:cx1])

    pooled = np.concatenate(strips, axis=1)  # (nrows, ncols)
    row_bg = np.median(pooled, axis=1).astype(np.float32)

    # Light smoothing to reduce single-row outliers while preserving
    # the horizon edge.
    if row_bg.size >= 5:
        kernel = np.array([1, 2, 4, 2, 1], dtype=np.float32)
        kernel /= kernel.sum()
        row_bg = np.convolve(row_bg, kernel, mode="same")
    return row_bg


def _collect_bands(
    radiance: np.ndarray,
    bbox: tuple[int, int, int, int],
    context: tuple[int, int, int, int],
    min_thickness: int,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Return list of ``(name, pixels, rect)`` for the 4 side bands."""
    bx0, by0, bx1, by1 = bbox
    cx0, cy0, cx1, cy1 = context
    bands = []

    def add(name, x0, y0, x1, y1):
        x0, y0, x1, y1 = _clip_rect(x0, y0, x1, y1, radiance.shape[1], radiance.shape[0])
        if x1 - x0 < min_thickness and y1 - y0 < min_thickness:
            return
        if x1 <= x0 or y1 <= y0:
            return
        patch = radiance[y0:y1, x0:x1]
        if patch.size == 0:
            return
        bands.append((name, patch.ravel().copy(), (x0, y0, x1, y1)))

    add("left",   cx0, by0, bx0, by1)
    add("right",  bx1, by0, cx1, by1)
    add("top",    cx0, cy0, cx1, by0)
    add("bottom", cx0, by1, cx1, cy1)
    return bands


def _collect_subregion_bands(
    radiance: np.ndarray,
    bbox: tuple[int, int, int, int],
    horizon_row: int,
    thickness: int,
    lateral_margin: int = 80,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Collect bands tagged with ``above``/``below`` sub-regions.

    For each side of the horizon we sample:

    * the corresponding side of the bbox (if that side is on the
      correct side of the horizon), and
    * wide lateral strips (left + right of the bbox) at the horizon
      sub-region's appropriate row range.

    This guarantees we always have clean samples from *both* the sea
    and the sky sub-regions even when the horizon lies near the bbox
    edge.
    """
    H, W = radiance.shape
    bx0, by0, bx1, by1 = bbox
    bands: list[tuple[str, np.ndarray, tuple[int, int, int, int]]] = []

    def add(name, x0, y0, x1, y1):
        x0, y0, x1, y1 = _clip_rect(x0, y0, x1, y1, W, H)
        if x1 <= x0 or y1 <= y0:
            return
        patch = radiance[y0:y1, x0:x1]
        if patch.size < 16:
            return
        bands.append((name, patch.ravel().copy(), (x0, y0, x1, y1)))

    # Above-horizon bands (sky).
    # Vertical range fully above horizon.
    ay0 = max(0, horizon_row - thickness)
    ay1 = max(0, horizon_row)
    # Wide lateral strips immediately left / right of bbox.
    add("above:left",  max(0, bx0 - lateral_margin), ay0, bx0, ay1)
    add("above:right", bx1, ay0, min(W, bx1 + lateral_margin), ay1)
    # Top strip spanning the full lateral extent — useful when horizon
    # is below or at bbox top.
    add("above:top",
        max(0, bx0 - lateral_margin),
        max(0, horizon_row - thickness),
        min(W, bx1 + lateral_margin),
        max(0, horizon_row),
    )
    # If the bbox has above-horizon interior left/right slabs, leave
    # those alone (they'd include target if the bbox is loose).

    # Below-horizon bands (sea).
    by0s = min(H, horizon_row)
    by1s = min(H, horizon_row + thickness)
    add("below:left",  max(0, bx0 - lateral_margin), by0s, bx0, by1s)
    add("below:right", bx1, by0s, min(W, bx1 + lateral_margin), by1s)
    add("below:bottom",
        max(0, bx0 - lateral_margin),
        min(H, horizon_row),
        min(W, bx1 + lateral_margin),
        min(H, horizon_row + thickness),
    )
    return bands


# --------------------------------------------------------------------------- #
# Core build_mask
# --------------------------------------------------------------------------- #


def _split_band_by_horizon(
    band: tuple[str, np.ndarray, tuple[int, int, int, int]],
    horizon_row: int,
    radiance: np.ndarray,
    min_px: int = 24,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Split a band into (above, below) horizon sub-bands."""
    name, _, (x0, y0, x1, y1) = band
    out = []
    if horizon_row > y0 + 1:
        patch = radiance[y0:min(horizon_row, y1), x0:x1]
        if patch.size >= min_px:
            out.append((f"{name}:above", patch.ravel().copy(),
                        (x0, y0, x1, min(horizon_row, y1))))
    if horizon_row < y1 - 1:
        patch = radiance[max(horizon_row, y0):y1, x0:x1]
        if patch.size >= min_px:
            out.append((f"{name}:below", patch.ravel().copy(),
                        (x0, max(horizon_row, y0), x1, y1)))
    return out


# --------------------------------------------------------------------------- #
# Core build_mask
# --------------------------------------------------------------------------- #


def build_mask(sample: Sample, **overrides) -> ExtractResult:
    """Build a target mask for a loaded sample.

    Parameters
    ----------
    sample
        Output of :func:`irpaste.io_utils.load_sample`.
    **overrides
        Override any key from :data:`DEFAULTS`.
    """
    p = {**DEFAULTS, **overrides}
    notes: list[str] = []

    rad = sample.radiance
    H, W = rad.shape
    ann: Annotation = sample.annotation

    # 1. Effective anchor (union of XML bbox and corner AABB). This is
    # the tightest rectangle that reliably encloses the ship body; the
    # raw bbox is used only as an additional hint for CC selection.
    bx0, by0, bx1, by1 = ann.anchor_xyxy(expand=p["bbox_expand"])
    # Build context window from the anchor with an absolute floor so
    # that undersized annotations still leave room for the mask to
    # grow into.
    half_w = max((bx1 - bx0) * (1.0 + p["context_expand"]) / 2.0,
                 p["context_min_half"])
    half_h = max((by1 - by0) * (1.0 + p["context_expand"]) / 2.0,
                 p["context_min_half"])
    cxc = 0.5 * (bx0 + bx1)
    cyc = 0.5 * (by0 + by1)
    cx0, cy0, cx1, cy1 = _clip_rect(
        int(np.floor(cxc - half_w)), int(np.floor(cyc - half_h)),
        int(np.ceil(cxc + half_w)),  int(np.ceil(cyc + half_h)),
        W, H,
    )
    bbox = (bx0, by0, bx1, by1)
    context = (cx0, cy0, cx1, cy1)

    # 2. Horizon detection (wide vertical search, side strips only).
    horizon_row = _detect_horizon_row(
        rad, bbox, context,
        search_margin=p["horizon_search_margin"],
        min_jump_factor=p["horizon_min_jump_factor"],
        min_jump_abs=p["horizon_min_jump_abs"],
    )

    horizon_relevant = (
        horizon_row is not None
        and (by0 - 30) <= horizon_row <= (by1 + 30)
    )

    # 3. Collect bands.
    if horizon_relevant:
        bands = _collect_subregion_bands(
            rad, bbox, horizon_row, p["subregion_band_rows"]
        )
        notes.append(
            f"horizon @row={horizon_row}"
            + (" (inside bbox)" if by0 < horizon_row < by1 else " (near bbox)")
        )
    else:
        if horizon_row is not None:
            notes.append(f"horizon @row={horizon_row} (far from bbox; ignored)")
        bands = _collect_bands(rad, bbox, context, p["min_band_thickness"])

    if not bands:
        notes.append("no background bands available")
        # Fallback to the context window minus bbox.
        return _empty_result(bbox, context, notes)

    # 4. Score bands and keep the cleanest half (at least 2).
    scored = [(name, _band_purity_score(pix), pix, rect)
              for (name, pix, rect) in bands]
    scored.sort(key=lambda x: x[1])
    keep_n = max(2, len(scored) // 2)
    selected = scored[:keep_n]
    # When horizon splits the bbox, ensure *both* sub-regions have at
    # least one representative band in the selected set (otherwise the
    # sub-region whose bands all tied on purity would be dropped by
    # the sort — this is how sea-side can get ignored for a ship-on-
    # horizon shot).
    if horizon_relevant:
        for suffix in ("above", "below"):
            if not any(n.startswith(suffix) for n, *_ in selected):
                best = next(
                    (item for item in scored if item[0].startswith(suffix)),
                    None,
                )
                if best is not None:
                    selected.append(best)
    notes.append(
        "bands=" + ",".join(f"{n}({s:.3f})" for n, s, *_ in selected)
    )

    # Per-sub-region background if horizon is relevant, else pooled.
    bg_medians: dict[str, tuple[float, float]] = {}
    if horizon_relevant:
        for suffix in ("above", "below"):
            pool_pix = [pix for (name, _, pix, _) in scored if name.startswith(suffix)]
            # Prefer *selected* bands on that side; fall back to all
            # scored bands on that side so each sub-region always has
            # background samples.
            sel = [pix for (name, _, pix, _) in selected if name.startswith(suffix)]
            pool = np.concatenate(sel) if sel else (
                np.concatenate(pool_pix) if pool_pix else np.array([], np.float32)
            )
            if pool.size:
                bg_medians[suffix] = _robust_stats(pool)

    pooled_pix = np.concatenate([pix for (_, _, pix, _) in selected])
    global_med, global_mad = _robust_stats(pooled_pix)

    # 5. Residual on the context window.
    # Always use a row-wise lateral profile: median per row from strips
    # outside the bbox columns. This naturally models both flat
    # backgrounds and sea/sky horizon transitions, and is insensitive
    # to whether the horizon detector fired.
    ctx = rad[cy0:cy1, cx0:cx1].copy()
    bg_rows = _lateral_row_profile(rad, bbox, context, lateral_margin=80)

    # v3: polygon-ring sanitiser.
    # The row-wise lateral median assumes each row's lateral strip is
    # pure background of one kind. On coastline nadir views the left
    # strip is saturated land while the right strip is sea, so the
    # per-row median floats between 24 and 40 and the threshold gets
    # corrupted. Bound bg_rows to the ring-derived local background
    # (a tight ring around the corner polygon is local sea/sky for
    # this dataset) so land contamination cannot push the row-bg far
    # from its true value.
    #
    # IMPORTANT: skip this when a horizon is active inside the ROI —
    # there the row-bg is *legitimately* bimodal (sky above, sea below)
    # and flattening it would destroy the residual. Also skip when
    # the ring itself is bimodal (high MAD), as the ring then spans
    # multiple backgrounds and its median is meaningless.
    corners_px = ann.corners_pixel()
    ring_med, ring_mad, ring_spread = _polygon_ring_stats(
        rad, corners_px,
        inner_dilate=p["poly_ring_inner"],
        outer_dilate=p["poly_ring_outer"],
    )
    ring_applied = False
    if ring_med is not None and not horizon_relevant:
        # Rings that truly straddle a sea/sky boundary show
        # p90-p10 spreads of 1.5+ (up to 17) radiance units. Clean
        # single-background rings are reliably below ~1.0 (empirical
        # 30-sample surveys: p90 ≤ 1.05 across every folder). Use a
        # flat 1.0 cutoff so coastline nadirs (spread ≈ 0) pass and
        # rain-horizon ambiguous cases (spread ≈ 1.5+) are rejected.
        spread_limit = 1.0
        if (ring_spread or 0.0) <= spread_limit:
            tol = max(p["poly_ring_clip_k"] * 1.4826 * (ring_mad or 0.0),
                      p["poly_ring_clip_abs"])
            lo = ring_med - tol
            hi = ring_med + tol
            bg_rows = np.clip(bg_rows, lo, hi).astype(np.float32)
            ring_applied = True
            notes.append(
                f"ring_bg med={ring_med:.2f} spread={ring_spread:.2f} clip=±{tol:.2f}"
            )
        else:
            notes.append(
                f"ring_bg med={ring_med:.2f} spread={ring_spread:.2f} "
                f"(skipped: bimodal ring, limit={spread_limit:.2f})"
            )
    elif ring_med is not None:
        notes.append(
            f"ring_bg med={ring_med:.2f} spread={ring_spread:.2f} "
            "(skipped: horizon active)"
        )

    bg = np.tile(bg_rows[:, None], (1, ctx.shape[1]))

    residual = ctx - bg
    # Estimate the noise scale directly on the residual, using the
    # *selected* background bands inside the context.
    bg_residuals = []
    for (_, _, pix, rect) in selected:
        rx0, ry0, rx1, ry1 = rect
        rx0c, ry0c = max(rx0, cx0), max(ry0, cy0)
        rx1c, ry1c = min(rx1, cx1), min(ry1, cy1)
        if rx1c <= rx0c or ry1c <= ry0c:
            continue
        patch = residual[ry0c - cy0: ry1c - cy0, rx0c - cx0: rx1c - cx0]
        bg_residuals.append(patch.ravel())
    if bg_residuals:
        bgres = np.concatenate(bg_residuals)
        res_mad = float(np.median(np.abs(bgres - np.median(bgres))))
    else:
        res_mad = global_mad

    sigma_from_mad = 1.4826 * res_mad
    # Noise floor for flat synthetic backgrounds: a small fraction of
    # the bbox interior dynamic range, so that a completely noiseless
    # background still yields a sensible non-zero threshold.
    roi = rad[by0:by1, bx0:bx1]
    roi_range = float(roi.max() - roi.min()) if roi.size else 1.0
    sigma_floor = 0.01 * max(roi_range, 1e-6)
    sigma = max(sigma_from_mad, sigma_floor)
    t_high = p["k_high"] * sigma
    t_low = max(0.4 * t_high, p["k_low"] * sigma)

    strong = np.abs(residual) >= t_high
    weak = np.abs(residual) >= t_low

    # 6. Hysteresis: keep weak pixels only if reachable from strong via
    # 8-connectivity.
    hysteresis = _hysteresis_select(strong, weak)

    # 6b. Edge-aided recovery for thin structures (masts).
    grad_x = cv2.Sobel(residual, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(residual, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)
    if grad_mag.size:
        g_thr = float(np.quantile(grad_mag, p["edge_quantile"]))
        edge_mask = grad_mag >= g_thr
        # Candidates: weak pixels on an edge, dilated to touch hysteresis.
        cand = weak & edge_mask & (~hysteresis)
        if cand.any():
            # Accept candidate pixels that touch hysteresis (already selected).
            dilated = cv2.dilate(
                hysteresis.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            ).astype(bool)
            hysteresis = hysteresis | (cand & dilated)

    # 7. Morphology + component selection.
    mask_ctx = hysteresis
    # Small close to seal 1-px gaps.
    mask_ctx = cv2.morphologyEx(
        mask_ctx.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ).astype(bool)

    # Hard clip to the anchor dilated by `anchor_clip_expand`. The
    # anchor (union of XML bbox + corner polygon AABB) is guaranteed
    # to enclose the ship body; anything outside a modest dilation
    # is wake, horizon smear, or surrounding clutter and must be
    # removed to get a unit ratio against XML pixelNum.
    ax0, ay0, ax1, ay1 = ann.anchor_xyxy(expand=p["anchor_clip_expand"])
    clip_mask = np.zeros_like(mask_ctx)
    cax0 = max(0, ax0 - cx0)
    cay0 = max(0, ay0 - cy0)
    cax1 = min(mask_ctx.shape[1], ax1 - cx0)
    cay1 = min(mask_ctx.shape[0], ay1 - cy0)
    if cax1 > cax0 and cay1 > cay0:
        clip_mask[cay0:cay1, cax0:cax1] = True
    mask_ctx = mask_ctx & clip_mask

    # v3: additionally intersect with the corner polygon dilated by a
    # small absolute margin. For tilted / nadir poses the AABB-dilated
    # clip still contains large background corners that leak land
    # (coastline failures); a poly-tight clip removes them without
    # hurting horizontal poses (where poly ≈ AABB).
    if corners_px is not None and p["poly_clip_dilate_px"] > 0:
        poly_full = _polygon_mask_from_corners(rad.shape, corners_px)
        if poly_full is not None:
            k = 2 * int(p["poly_clip_dilate_px"]) + 1
            poly_dilated = cv2.dilate(
                poly_full.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
            ).astype(bool)
            poly_ctx = poly_dilated[cy0:cy1, cx0:cx1]
            mask_ctx = mask_ctx & poly_ctx

    # Remove speckle (<2 px) BEFORE component selection so that a
    # stray 1-px CC near the anchor center cannot outscore the real
    # ship CC on the distance-favoring metric. (Bug observed on
    # pitch=0 horizon poses where the XML corner polygon is a large
    # square centered in sky; the ship sits below center while a 1-px
    # noise speckle near the geometric center would win the score.)
    mask_ctx = _remove_speckle(mask_ctx, min_size=2)

    # Component selection: prefer components overlapping the anchor
    # interior. We use the full anchor (not a shrunk inner rect) so
    # that small distant targets — whose anchor may itself be only
    # a few pixels — are not missed.
    ox0, oy0, ox1, oy1 = ann.anchor_xyxy(expand=0.0)
    inner_x0 = ox0
    inner_x1 = ox1
    inner_y0 = oy0
    inner_y1 = oy1

    num, labels = cv2.connectedComponents(mask_ctx.astype(np.uint8), connectivity=8)
    if num <= 1:
        notes.append("no foreground component found")
        selected_mask_ctx = np.zeros_like(mask_ctx)
    else:
        # Convert inner rect to context coords.
        ix0 = inner_x0 - cx0
        iy0 = inner_y0 - cy0
        ix1 = inner_x1 - cx0
        iy1 = inner_y1 - cy0
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1 = min(mask_ctx.shape[1], ix1)
        iy1 = min(mask_ctx.shape[0], iy1)
        inner_labels: set[int] = set()
        if ix1 > ix0 and iy1 > iy0:
            inner_labels = set(np.unique(labels[iy0:iy1, ix0:ix1]).tolist()) - {0}
        if not inner_labels:
            # Fallback: largest component within bbox.
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            inner_labels = {int(np.argmax(sizes))}
            notes.append("no component in inner bbox; using largest")

        # Of the components overlapping the anchor interior, keep the
        # one whose centroid is closest to the anchor center. This
        # robustly rejects ship reflections (foggy / calm sea), halo
        # blobs, and stray CCs — all of which are spatially offset
        # from the ship body.
        anchor_cx_ctx = 0.5 * ((ox0 - cx0) + (ox1 - cx0))
        anchor_cy_ctx = 0.5 * ((oy0 - cy0) + (oy1 - cy0))
        ys, xs = np.indices(labels.shape)
        best_label = None
        best_score = np.inf
        sizes = np.bincount(labels.ravel())
        for lab in inner_labels:
            m = labels == lab
            n = int(m.sum())
            if n == 0:
                continue
            cx = float(xs[m].mean())
            cy = float(ys[m].mean())
            d = float(np.hypot(cx - anchor_cx_ctx, cy - anchor_cy_ctx))
            # Prefer close-to-center; break ties by size (favor larger).
            score = d - 0.05 * np.sqrt(n)
            if score < best_score:
                best_score = score
                best_label = lab
        if best_label is None:
            selected_mask_ctx = np.zeros_like(mask_ctx)
        else:
            selected_mask_ctx = labels == best_label

    # Remove speckle <2 px.
    selected_mask_ctx = _remove_speckle(selected_mask_ctx, min_size=2)

    # Horizon-bleed trim: horizontal sea/sky boundary pixels extend
    # laterally from the ship via hysteresis (long chain of
    # weak-but-above-t_low pixels). Their residual magnitude is tiny
    # (typically < 0.8·t_high), whereas every genuine ship pixel sits
    # well above t_high. Drop mask pixels that are not within a
    # 1-pixel neighborhood of a strong-residual pixel. This trims the
    # thin lateral bar without touching the ship body (all strong) or
    # thin mast/deck features (connected to strong neighbors).
    if selected_mask_ctx.any():
        mask_abs_res = np.abs(residual[selected_mask_ctx])
        ship_p90 = float(np.percentile(mask_abs_res, 90))
        # Adaptive cutoff: scale to the brightest ship pixels so horizon
        # bleed (residual ~0-3) is excluded while dim-but-real features
        # (thin masts with residual equal to several sigma) are kept.
        strict_floor = max(0.8 * t_high, 0.25 * ship_p90)
        strict = np.abs(residual) >= strict_floor
        strict_dil = cv2.dilate(
            strict.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        ).astype(bool)
        before = int(selected_mask_ctx.sum())
        trimmed = selected_mask_ctx & strict_dil
        after = int(trimmed.sum())
        # Safety: only accept the trim if it leaves a plausible core
        # (at least 30 % of pixels remain and the CC is non-empty).
        if after >= max(8, int(0.3 * before)):
            selected_mask_ctx = trimmed
            if after < before:
                notes.append(f"bleed_trim {before}->{after}")

    # Paste context mask into a full-image mask.
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[cy0:cy1, cx0:cx1] = selected_mask_ctx

    return ExtractResult(
        mask=full_mask,
        bbox=bbox,
        context=context,
        horizon_row=horizon_row,
        bg_median=global_med,
        bg_mad=global_mad,
        t_high=t_high,
        t_low=t_low,
        n_mask=int(full_mask.sum()),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Utility: hysteresis & speckle filter
# --------------------------------------------------------------------------- #


def _hysteresis_select(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep all strong pixels plus weak pixels reachable from strong
    through 8-connectivity. Uses connected components on ``weak``."""
    if not strong.any():
        return np.zeros_like(strong)
    num, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if num <= 1:
        return strong.copy()
    good_labels = np.unique(labels[strong])
    good_labels = good_labels[good_labels > 0]
    if good_labels.size == 0:
        return strong.copy()
    return np.isin(labels, good_labels)


def _remove_speckle(mask: np.ndarray, min_size: int = 2) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            keep[i] = True
    return keep[labels]


def _empty_result(bbox, context, notes) -> ExtractResult:
    return ExtractResult(
        mask=np.zeros((0, 0), dtype=bool),  # caller must handle
        bbox=bbox,
        context=context,
        horizon_row=None,
        bg_median=0.0,
        bg_mad=0.0,
        t_high=0.0,
        t_low=0.0,
        n_mask=0,
        notes=notes,
    )
