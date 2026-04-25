"""IRPaste — extract IR ship targets from simulation images and paste them
onto real backgrounds.

Public API
----------
Mask extraction:
    ``build_mask(sample, **kwargs)`` → ExtractResult

Paste pipeline:
    ``paste_target(sample, mask, bg, ...)`` → PasteResult

Loading helpers:
    ``load_sample(stem)`` → Sample
    ``load_background(path)`` → np.ndarray (uint8 grayscale)

View classification:
    ``classify_background(gray, ...)`` → BackgroundView | str
    ``classify_target(xml_path)``      → 'side' | 'top'
"""

from .io_utils import Annotation, Sample, load_sample, parse_xml, read_dat
from .extract import ExtractResult, build_mask
from .paste import (
    PasteResult,
    augment_background,
    load_background,
    paste_patch,
    paste_target,
    radiometric_match,
)
from .viewcls import BackgroundView, HorizonCurve, classify_background, classify_target

__all__ = [
    # data classes
    "Annotation",
    "ExtractResult",
    "PasteResult",
    "Sample",
    "BackgroundView",
    "HorizonCurve",
    # core API
    "build_mask",
    "load_sample",
    "load_background",
    "paste_target",
    # utilities
    "augment_background",
    "classify_background",
    "classify_target",
    "parse_xml",
    "radiometric_match",
    "read_dat",
]
