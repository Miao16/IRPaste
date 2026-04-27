"""Horizon cache: persist horizon-curve data alongside background images.

Each background image gets a sibling ``.json`` file (e.g. ``side_000001.json``)
so that classified/adjusted horizons survive across sessions and can be read
by the paste pipeline without re-running the classifier.

One-time usage
--------------
    python -c "from irpaste.horizon_cache import *; ..."

Module API
----------
    :func:`save_horizon`       — write cache for one background
    :func:`load_horizon`       — read cache (returns ``None`` if absent)
    :func:`compute_and_save`   — classify + write cache
    :func:`load_or_compute`    — read cache, compute+write on cache miss
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .viewcls import BackgroundView, HorizonCurve, classify_background

HORIZON_CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class HorizonCacheData:
    """Serialisable horizon data for one background image."""

    kind: str  # "side" | "top"
    horizon_curve: Optional[dict] = None  # {"a":a,"b":b,"c":c,"rmse":r, ...}
    horizon_row: Optional[int] = None
    version: int = HORIZON_CACHE_VERSION

    # ------------------------------------------------------------------ #
    # Convert to/from BackgroundView
    # ------------------------------------------------------------------ #

    @classmethod
    def from_background_view(
        cls, bg_view: BackgroundView
    ) -> "HorizonCacheData":
        curve = None
        if bg_view.horizon_curve is not None:
            curve = asdict(bg_view.horizon_curve)
        return cls(
            kind=bg_view.kind,
            horizon_curve=curve,
            horizon_row=bg_view.horizon_row,
        )

    def to_background_view(self) -> BackgroundView:
        curve: Optional[HorizonCurve] = None
        if self.horizon_curve is not None:
            curve = HorizonCurve(**self.horizon_curve)
        return BackgroundView(
            kind=self.kind,
            horizon_row=self.horizon_row,
            score=0.0,
            variance_ratio=0.0,
            horizon_curve=curve,
        )


# --------------------------------------------------------------------------- #
# File path helpers
# --------------------------------------------------------------------------- #


def cache_path_for_image(image_path: str | Path) -> Path:
    """Return the ``.json`` cache path for a given image file."""
    return Path(image_path).with_suffix(".json")


# --------------------------------------------------------------------------- #
# Read / Write
# --------------------------------------------------------------------------- #


def save_horizon(image_path: str | Path, data: HorizonCacheData) -> None:
    """Write horizon cache ``.json`` next to the image."""
    path = cache_path_for_image(image_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(data), f, indent=2, ensure_ascii=False)


def load_horizon(image_path: str | Path) -> Optional[HorizonCacheData]:
    """Read horizon cache ``.json`` next to the image.

    Returns ``None`` when no cache file exists.
    """
    path = cache_path_for_image(image_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return HorizonCacheData(**raw)


# --------------------------------------------------------------------------- #
# Compute + cache
# --------------------------------------------------------------------------- #


def compute_and_save(
    image_path: str | Path, gray: np.ndarray
) -> HorizonCacheData:
    """Run :func:`classify_background` on *gray*, cache the result, return it."""
    bg_view = classify_background(gray, return_info=True)
    data = HorizonCacheData.from_background_view(bg_view)
    save_horizon(image_path, data)
    return data


def load_or_compute(
    image_path: str | Path, gray: np.ndarray
) -> HorizonCacheData:
    """Return cached horizon data if available, otherwise compute + cache."""
    cached = load_horizon(image_path)
    if cached is not None:
        return cached
    return compute_and_save(image_path, gray)
