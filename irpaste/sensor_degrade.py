"""Sensor degradation effects to bridge the synthetic-to-real IR domain gap.

Real IR sensors introduce characteristic artefacts that are absent from
simulation data:

* MTF (modulation-transfer-function) blur from optics
* Row/column fixed-pattern noise from detector non-uniformity
* Low-frequency spatial noise from imperfect NUC (non-uniformity correction)
* Bad pixels (dead / stuck / blinking)
* 1/f row-correlated streaks

Usage::

    from irpaste.sensor_degrade import degrade_composite
    degraded = degrade_composite(composite, rng)
"""

from __future__ import annotations

import numpy as np
import cv2


def degrade_composite(
    img: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    p_mtf: float = 0.5,
    p_rowcol: float = 0.25,
    p_nuc: float = 0.4,
    p_bad_pixels: float = 0.20,
    p_streaks: float = 0.20,
) -> np.ndarray:
    """Apply randomised sensor-degradation effects to an IR composite.

    Each effect has an independent probability of being applied so the
    result is not a fixed pipeline — some images get blur, others get
    NUC noise, most get a subset, creating the visual variety seen in
    real sensor captures.

    Parameters
    ----------
    img : np.ndarray
        uint8 grayscale composite image.
    rng : np.random.Generator or None
        Random generator.
    p_mtf : float
        Probability of applying MTF blur.
    p_rowcol : float
        Probability of applying row/column FPN.
    p_nuc : float
        Probability of low-frequency NUC residual noise.
    p_bad_pixels : float
        Probability of bad-pixel injection.
    p_streaks : float
        Probability of 1/f row-streak noise.

    Returns
    -------
    np.ndarray
        uint8 degraded image.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = img.astype(np.float32)

    if rng.random() < p_mtf:
        result = _apply_mtf_blur(result, rng)

    if rng.random() < p_rowcol:
        result = _apply_rowcol_fpn(result, rng)

    if rng.random() < p_nuc:
        result = _apply_nuc_noise(result, rng)

    if rng.random() < p_bad_pixels:
        result = _apply_bad_pixels(result, rng)

    if rng.random() < p_streaks:
        result = _apply_streaks(result, rng)

    np.clip(result, 0.0, 255.0, out=result)
    return result.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Individual effects
# --------------------------------------------------------------------------- #


def _apply_mtf_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian blur simulating optical MTF / diffraction limit.

    Sigma is drawn uniformly from [0.3, 1.5] so the effect ranges from
    barely noticeable (sharp optics, low altitude) to soft (long range,
    atmospheric haze, lower-quality sensor).
    """
    sigma = float(rng.uniform(0.3, 1.5))
    ksz = max(3, (int(sigma * 6) | 1))
    return cv2.GaussianBlur(img, (ksz, ksz), sigma)


def _apply_rowcol_fpn(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject row/column fixed-pattern noise.

    A small fraction of rows and columns get a constant additive offset,
    mimicking uncorrected detector non-uniformity or amplifier drift.
    """
    H, W = img.shape

    # Row FPN — subtle strip offsets
    frac_rows = float(rng.uniform(0.02, 0.06))
    n_rows = max(1, int(H * frac_rows))
    row_idx = rng.choice(H, n_rows, replace=False)
    row_offsets = rng.uniform(-4.0, 4.0, size=n_rows)
    result = img.copy()
    result[row_idx, :] += row_offsets[:, np.newaxis]

    # Column FPN
    frac_cols = float(rng.uniform(0.02, 0.06))
    n_cols = max(1, int(W * frac_cols))
    col_idx = rng.choice(W, n_cols, replace=False)
    col_offsets = rng.uniform(-4.0, 4.0, size=n_cols)
    result[:, col_idx] += col_offsets[np.newaxis, :]

    return result


def _apply_nuc_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency spatial noise simulating imperfect NUC correction.

    A large-kernel Gaussian noise field models the slowly varying
    spatial non-uniformity that remains after a single-point or
    two-point NUC calibration, especially in uncooled microbolometer
    sensors.
    """
    H, W = img.shape
    # Downsample → add noise → upsample for a smooth low-frequency field.
    scale = 0.25
    Hs, Ws = max(8, int(H * scale)), max(8, int(W * scale))
    small = rng.normal(0.0, 1.0, size=(Hs, Ws)).astype(np.float32)

    sigma = float(rng.uniform(2.0, 6.0))
    ksz = max(3, (int(sigma * 4) | 1))
    small = cv2.GaussianBlur(small, (ksz, ksz), sigma)

    field = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    amp = float(rng.uniform(1.5, 5.0))
    field = field / (field.std() + 1e-6) * amp

    return img + field


def _apply_bad_pixels(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace a tiny fraction of pixels with anomalous (not dead/stuck) values.

    Instead of pure 0 / 255, the injected pixels are pushed moderately
    away from the local mean so they read as sensor outliers without
    looking like synthetic artefacts.

    Fraction is drawn from [0.02%, 0.15%] of total pixels.
    """
    H, W = img.shape
    img_mean = float(img.mean())
    frac = float(rng.uniform(0.0002, 0.0015))
    n = max(1, int(H * W * frac))
    ys = rng.integers(0, H, size=n)
    xs = rng.integers(0, W, size=n)

    result = img.copy()
    for i in range(n):
        kind = rng.random()
        # Scale the offset relative to image std so noise adapts to content.
        local = float(result[ys[i], xs[i]])
        if kind < 0.40:
            # Cool outlier — push below local value
            offset = float(rng.uniform(25, 70))
            result[ys[i], xs[i]] = max(0.0, local - offset)
        elif kind < 0.80:
            # Hot outlier — push above local value
            offset = float(rng.uniform(25, 70))
            result[ys[i], xs[i]] = min(255.0, local + offset)
        else:
            # Random flicker — anywhere in [0, 255]
            result[ys[i], xs[i]] = float(rng.uniform(0.0, 255.0))

    return result


def _apply_streaks(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1/f-like row-correlated streak noise.

    Each row receives a low-frequency random offset, producing the
    horizontal banding characteristic of rolling-shutter readout
    circuits in uncooled microbolometers.
    """
    H, _ = img.shape
    # Build correlated row offsets via random walk with drift pull-back.
    steps = rng.normal(0.0, 1.0, size=H).astype(np.float32)
    # Low-pass filter in 1D (row direction)
    sigma = float(rng.uniform(8.0, 25.0))
    ksz = max(3, min(H | 1, (int(sigma * 6) | 1)))
    streaks_1d = cv2.GaussianBlur(steps.reshape(1, -1), (1, ksz), sigma).reshape(-1)

    amp = float(rng.uniform(0.5, 2.5))
    streaks_1d = streaks_1d / (np.std(streaks_1d) + 1e-6) * amp

    return img + streaks_1d[:, np.newaxis]
