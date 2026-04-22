"""IRPaste — extract IR ship targets from simulation images and paste them
onto real backgrounds.

This package currently focuses on the first stage: building a precise
per-pixel target mask from each simulation sample (radiance ``.dat`` +
XML annotation).
"""

from .io_utils import Annotation, Sample, load_sample, parse_xml, read_dat
from .extract import ExtractResult, build_mask

__all__ = [
    "Annotation",
    "ExtractResult",
    "Sample",
    "build_mask",
    "load_sample",
    "parse_xml",
    "read_dat",
]
