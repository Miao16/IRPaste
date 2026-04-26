"""I/O helpers for IR simulation samples.

Each sample has four sibling files sharing a stem:

* ``.png`` / ``.bmp`` — 8-bit rendered preview.
* ``.dat`` — raw radiance. Layout::

      uint32 width
      uint32 height
      float32 pixel[H * W]   # row-major

* ``.xml`` — annotation. We read the first ``<target>``.

The radiance array is the preferred signal for segmentation because it
preserves the physical dynamic range that 8-bit rendering compresses
away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Annotation:
    """Single-target annotation extracted from an XML file."""

    center_x: float
    center_y: float
    width: float
    height: float
    # Normalised polygon corners (x1,y1 .. x4,y4) if present.
    corners_norm: Optional[np.ndarray]  # shape (4, 2) or None
    pixel_num: int
    avg_rad_brightness: Optional[float]
    snr: Optional[float]
    contrast: Optional[float]
    image_width: int
    image_height: int

    def bbox_xyxy(
        self, expand: float = 0.0, clip: bool = True
    ) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` bbox, optionally expanded by
        ``expand`` (fraction, e.g. ``0.05`` for +5 %)."""
        w = self.width * (1.0 + expand)
        h = self.height * (1.0 + expand)
        x0 = self.center_x - w / 2.0
        y0 = self.center_y - h / 2.0
        x1 = self.center_x + w / 2.0
        y1 = self.center_y + h / 2.0
        if clip:
            x0 = max(0, int(np.floor(x0)))
            y0 = max(0, int(np.floor(y0)))
            x1 = min(self.image_width, int(np.ceil(x1)))
            y1 = min(self.image_height, int(np.ceil(y1)))
        else:
            x0, y0, x1, y1 = (
                int(np.floor(x0)),
                int(np.floor(y0)),
                int(np.ceil(x1)),
                int(np.ceil(y1)),
            )
        return x0, y0, x1, y1

    def corners_pixel(self) -> Optional[np.ndarray]:
        """Return the XML polygon corners in pixel coords, or ``None``."""
        if self.corners_norm is None:
            return None
        out = self.corners_norm.copy()
        out[:, 0] *= self.image_width
        out[:, 1] *= self.image_height
        return out

    def anchor_xyxy(
        self, expand: float = 0.0, clip: bool = True
    ) -> tuple[int, int, int, int]:
        """Effective target anchor rectangle.

        Union of the XML bbox and the axis-aligned bounding box of the
        XML polygon corners, optionally expanded by ``expand``.

        The bbox and the corners disagree for some oblique/nadir poses
        (for those the bbox encodes a horizontal arm only while the
        corners describe the vertical ship body). Their union is
        guaranteed to enclose the ship for every sample we have
        inspected, so it is a much tighter and more reliable prior than
        the bbox alone.
        """
        bx0, by0, bx1, by1 = self.bbox_xyxy(expand=0.0, clip=False)
        corners = self.corners_pixel()
        if corners is not None and len(corners) >= 3:
            cx0 = float(np.min(corners[:, 0]))
            cy0 = float(np.min(corners[:, 1]))
            cx1 = float(np.max(corners[:, 0]))
            cy1 = float(np.max(corners[:, 1]))
            x0 = min(bx0, cx0)
            y0 = min(by0, cy0)
            x1 = max(bx1, cx1)
            y1 = max(by1, cy1)
        else:
            x0, y0, x1, y1 = bx0, by0, bx1, by1
        if expand:
            w = x1 - x0
            h = y1 - y0
            x0 -= w * expand / 2.0
            x1 += w * expand / 2.0
            y0 -= h * expand / 2.0
            y1 += h * expand / 2.0
        x0i, y0i = int(np.floor(x0)), int(np.floor(y0))
        x1i, y1i = int(np.ceil(x1)), int(np.ceil(y1))
        if clip:
            x0i = max(0, x0i)
            y0i = max(0, y0i)
            x1i = min(self.image_width, x1i)
            y1i = min(self.image_height, y1i)
        return x0i, y0i, x1i, y1i


@dataclass
class Sample:
    """A loaded sample: radiance + (optional) preview + annotation."""

    stem: Path
    radiance: np.ndarray  # float32 (H, W)
    preview: Optional[np.ndarray]  # uint8 (H, W) grayscale, may be None
    annotation: Annotation


# --------------------------------------------------------------------------- #
# XML parser
# --------------------------------------------------------------------------- #


def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_xml(xml_path: str | Path) -> Annotation:
    """Parse the first ``<target>`` from a simulation XML file."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensor = root.find("imageSensor")
    if sensor is None:
        raise ValueError(f"{xml_path}: missing <imageSensor>")
    img_w = int(float(sensor.get("width")))
    img_h = int(float(sensor.get("height")))

    target = root.find("./targets/target")
    if target is None:
        raise ValueError(f"{xml_path}: no <target> element")

    # Corners: 4 normalised (x_i, y_i) points if present.
    corners = []
    for i in range(1, 5):
        xi = target.get(f"x{i}")
        yi = target.get(f"y{i}")
        if xi is None or yi is None:
            corners = None
            break
        corners.append((float(xi), float(yi)))
    corners_arr: Optional[np.ndarray] = (
        np.asarray(corners, dtype=np.float32) if corners else None
    )

    return Annotation(
        center_x=float(target.get("centerX")),
        center_y=float(target.get("centerY")),
        width=float(target.get("width")),
        height=float(target.get("height")),
        corners_norm=corners_arr,
        pixel_num=int(float(target.get("pixelNum", "0"))),
        avg_rad_brightness=_float_or_none(target.get("avgRadBrightness")),
        snr=_float_or_none(target.get("SNR")),
        contrast=_float_or_none(target.get("contrast")),
        image_width=img_w,
        image_height=img_h,
    )


# --------------------------------------------------------------------------- #
# .dat radiance reader
# --------------------------------------------------------------------------- #


def read_dat(dat_path: str | Path) -> np.ndarray:
    """Read a radiance ``.dat`` file.

    Format: 8-byte header ``uint32 width, uint32 height`` followed by
    ``H * W`` little-endian ``float32`` values in row-major order.
    """
    dat_path = Path(dat_path)
    raw = dat_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{dat_path}: file too small for header")
    header = np.frombuffer(raw[:8], dtype=np.uint32)
    w, h = int(header[0]), int(header[1])
    expected = 8 + w * h * 4
    if len(raw) != expected:
        raise ValueError(
            f"{dat_path}: size {len(raw)} does not match header {w}x{h} "
            f"float32 (expected {expected})"
        )
    pixels = np.frombuffer(raw[8:], dtype=np.float32).reshape(h, w)
    # Copy so callers can safely write.
    return pixels.astype(np.float32, copy=True)


# --------------------------------------------------------------------------- #
# Sample loader
# --------------------------------------------------------------------------- #


def _read_preview(stem: Path) -> Optional[np.ndarray]:
    """Best-effort load of the 8-bit preview. Returns a grayscale uint8
    image of shape ``(H, W)`` or ``None``."""
    try:
        import cv2  # local import so io_utils stays importable without cv2
    except ImportError:  # pragma: no cover
        return None

    for ext in (".png", ".bmp"):
        p = stem.with_name(stem.name + ext)
        if not p.exists():
            continue
        buf = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            return img
        # RGBA or BGR → convert to gray using BGR channels only.
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None


def load_sample(stem: str | Path) -> Sample:
    """Load a sample by stem (path *without* extension)."""
    stem = Path(stem)
    # Only strip a known media suffix — filenames in this dataset contain
    # dots in the base name (e.g. "半径5.00"), so ``with_suffix("")``
    # would chop off the real filename. Use an explicit whitelist instead.
    if stem.suffix.lower() in {".xml", ".dat", ".png", ".bmp"}:
        stem = stem.with_name(stem.name[: -len(stem.suffix)])
    xml_path = stem.with_name(stem.name + ".xml")
    dat_path = stem.with_name(stem.name + ".dat")
    annotation = parse_xml(xml_path)
    radiance = read_dat(dat_path)
    if radiance.shape != (annotation.image_height, annotation.image_width):
        raise ValueError(
            f"{stem}: .dat shape {radiance.shape} != XML "
            f"({annotation.image_height}, {annotation.image_width})"
        )
    preview = _read_preview(stem)
    return Sample(stem=stem, radiance=radiance, preview=preview, annotation=annotation)
