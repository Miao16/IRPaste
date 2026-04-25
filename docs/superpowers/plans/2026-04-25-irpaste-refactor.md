# IRPaste 批量张贴管线重构 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two-phase pipeline — pre-extract all target masks to disk cache, then refactor bulk paste to use cached data with shared background augmentation per batch, fix overlap/horizon placement bugs, add tqdm.

**Architecture:** New `scripts/pre_extract.py` handles phase 1 (mask extraction → .npz cache + manifest.csv). Refactored `scripts/paste_bulk.py` handles phase 2 (load cache → paste batches). New `paste_patch()` function in `irpaste/paste.py` accepts pre-cropped patch+mask directly, bypassing `load_sample`/`build_mask`/`target_patch_from_sample`. `paste_target` is refactored to call `paste_patch` internally.

**Tech Stack:** Python 3.12+, numpy, opencv-python, tqdm

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `irpaste/paste.py` | Modify | Add `paste_patch()`, refactor `paste_target`, fix `choose_paste_site` fallback |
| `irpaste/__init__.py` | Modify | Export `paste_patch` |
| `scripts/pre_extract.py` | Create | Phase 1: batch mask extraction → .npz cache |
| `scripts/paste_bulk.py` | Modify | Phase 2: load cache, shared bg augment, tqdm, composite fixes |

---

### Task 1: Add `paste_patch()` and refactor `paste_target`

**Files:**
- Modify: `irpaste/paste.py`
- Modify: `irpaste/__init__.py`

`paste_patch` is the lower-level entry point that takes a pre-cropped (patch, mask) pair and does everything from scaling through final composite. `paste_target` becomes a thin wrapper that handles augment_bg, classify_background, detect_target_on_horizon, and target_patch_from_sample, then delegates to `paste_patch`.

- [ ] **Step 1: Extract `paste_patch` function**

Insert after `rotate_patch_to_angle` (before the existing `paste_target`). The function body is the bottom half of `paste_target` (from the align_to_horizon section onward), with `patch` and `mask` as direct parameters instead of derived from `sample`.

In `irpaste/paste.py`, add this function before `paste_target` (around line 843):

```python
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
                f"axis-align: principal={principal_angle:.1f}° "
                f"horizon={horizon_angle:.1f}° rot={rotation:.1f}°"
            )

    # --- Optional ship downscale ---
    s_lo, s_hi = ship_scale_range
    if s_lo < s_hi:
        ship_scale = float(rng.uniform(s_lo, s_hi))
    else:
        ship_scale = float(s_lo)
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
```

- [ ] **Step 2: Rewrite `paste_target` to delegate to `paste_patch`**

Replace the body of `paste_target` from line 911 onward. Keep the signature, docstring, and steps 1-4 (augment, classify, detect_horizon, target_patch_from_sample), then call `paste_patch`:

```python
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
        occupied_mask=occupied_mask,
    )
    pr.sim_horizon_row = sim_hr
    pr.notes = notes + pr.notes
    return pr
```

- [ ] **Step 3: Update `irpaste/__init__.py` to export `paste_patch`**

Add `paste_patch` to the import from `.paste` and to `__all__`:

```python
from .paste import (
    PasteResult,
    augment_background,
    load_background,
    paste_patch,
    paste_target,
    radiometric_match,
)

__all__ = [
    # ...
    "paste_patch",
    "paste_target",
    # ...
]
```

- [ ] **Step 4: Verify**

```bash
uv run python -c "from irpaste import paste_patch; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add irpaste/paste.py irpaste/__init__.py
git commit -m "refactor: extract paste_patch from paste_target for pre-cropped targets"
```

---

### Task 2: Fix `choose_paste_site` fallback logic

**Files:**
- Modify: `irpaste/paste.py` (lines 289-312)

Two bugs:
1. Side-view fallback when retries exhausted: `target_on_horizon=True` returns `(x, y_hi)` placing the ship at the bottom of frame instead of near the horizon.
2. Top-down fallback doesn't check overlap at all.

- [ ] **Step 1: Fix side-view fallback**

In `irpaste/paste.py`, replace lines 289-301 (the "No overlap-free site found" block):

```python
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
```

- [ ] **Step 2: Fix top-down fallback to check overlap**

Replace lines 309-312 (the top-down fallback):

```python
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
        if float(region.mean()) < 0.25:  # ≤ 25 % overlap as last resort
            return x, y
    x = _center_biased_int(rng, x_lo, x_hi, bias=2.0)
    y = _center_biased_int(rng, y_lo, y_hi, bias=2.0)
    return x, y
```

- [ ] **Step 3: Commit**

```bash
git add irpaste/paste.py
git commit -m "fix: choose_paste_site fallback respects horizon and overlap constraints"
```

---

### Task 3: Create `scripts/pre_extract.py`

**Files:**
- Create: `scripts/pre_extract.py`

Phase 1 script: scan all targets, run `build_mask` once per target, save tight-cropped patch + mask + metadata to `{cache_dir}/{stem_hash}.npz`, write `manifest.csv`. Uses tqdm. Supports `--resume`.

- [ ] **Step 1: Write the script**

```python
"""Pre-extract all target masks and save to disk cache.

Usage::

    uv run python scripts/pre_extract.py \\
      --targets-root data/burkeIIA长波 \\
      --cache-dir outputs/_cache \\
      --resume

Produces:
    {cache_dir}/*.npz          — one per target (patch + mask + metadata)
    {cache_dir}/manifest.csv   — index: stem, view, on_horizon, cache_file
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste import build_mask, load_sample, classify_target  # noqa: E402
from irpaste.paste import detect_target_on_horizon, target_patch_from_sample, sim_preview_u8  # noqa: E402


def _stem_hash(stem: Path) -> str:
    """Short deterministic hash of the stem path for cache filenames."""
    return hashlib.sha256(str(stem).encode()).hexdigest()[:16]


def _extract_one(stem: Path) -> dict | None:
    """Extract mask + patch for one target. Returns dict or None on failure."""
    try:
        sample = load_sample(stem)
        res = build_mask(sample)
        if res.n_mask < 40:
            return None
        patch, mask, _ = target_patch_from_sample(sample, res.mask)
        if mask.sum() < 40:
            return None
        on_horizon, sim_hr = detect_target_on_horizon(sample, res.mask)
        return dict(
            patch=patch,
            mask=mask,
            on_horizon=on_horizon,
            sim_horizon_row=sim_hr if sim_hr is not None else -1,
        )
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-root", default="data/burkeIIA长波")
    ap.add_argument("--cache-dir", default="outputs/_cache")
    ap.add_argument("--resume", action="store_true", help="skip already-cached targets")
    args = ap.parse_args()

    root = Path(args.targets_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all target stems.
    stems: list[tuple[Path, str]] = []  # (stem, view)
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            view = classify_target(xml_path)
        except Exception:
            continue
        stem = xml_path.with_suffix("")
        stems.append((stem, view))

    if not stems:
        print("No targets found.")
        return 1

    manifest_path = cache_dir / "manifest.csv"
    existing = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open("r") as fh:
            for row in csv.reader(fh):
                if row:
                    existing.add(row[0])

    t0 = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0

    with manifest_path.open("a" if args.resume else "w", newline="") as mf:
        writer = csv.writer(mf)
        if not args.resume:
            writer.writerow(["stem", "view", "on_horizon", "sim_horizon_row", "cache_file"])

        for stem, view in tqdm(stems, desc="Extracting masks", unit="target"):
            stem_str = str(stem)
            if args.resume and stem_str in existing:
                n_skip += 1
                continue

            data = _extract_one(stem)
            if data is None:
                n_fail += 1
                continue

            fname = f"{_stem_hash(stem)}.npz"
            npz_path = cache_dir / fname
            np.savez_compressed(
                npz_path,
                patch=data["patch"],
                mask=data["mask"],
                view=view,
                on_horizon=data["on_horizon"],
                sim_horizon_row=np.float32(data["sim_horizon_row"]),
                stem=stem_str,
            )
            writer.writerow([
                stem_str, view,
                int(data["on_horizon"]),
                data["sim_horizon_row"],
                fname,
            ])
            mf.flush()
            n_ok += 1

    dt = time.time() - t0
    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed in {dt:.1f}s")
    print(f"Cache: {cache_dir}/  ({n_ok} .npz files)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/pre_extract.py
git commit -m "feat: add pre_extract.py for batch mask extraction to disk cache"
```

---

### Task 4: Refactor `scripts/paste_bulk.py`

**Files:**
- Modify: `scripts/paste_bulk.py`

Full rewrite of `main()` and the batch loop. Key changes:
- Load manifest.csv + index backgrounds (both now have view pre-computed)
- Each batch: augment bg once, classify once, all ships share
- Load targets from .npz cache, call `paste_patch` instead of `paste_target`
- Fix occupied_mask: dilate 2px before recording
- Fix multi-ship compositing: use feathered alpha on new ship region only
- tqdm progress bar
- Keep: _Shuffler, _annotate_multi, _compute_label, _write_multi_label, _build_contact_sheet, _safe_stem (no changes needed)

- [ ] **Step 1: Rewrite `paste_bulk.py`**

```python
"""Bulk paste: generate N view-matched composites with diverse targets.

Two-phase workflow:
  1. Pre-extract masks:  uv run python scripts/pre_extract.py --targets-root ... --cache-dir ...
  2. Bulk paste:         uv run python scripts/paste_bulk.py --cache-dir ... --bg-root ... --n 512 --seed 7

Usage::

    uv run python scripts/paste_bulk.py --n 512 --seed 7 --cache-dir outputs/_cache

Outputs:
    outputs/_bulk/{idx:06d}_{view}_{bg}_{target}_n{N}.png  — composite
    outputs/_bulk/clean/                                    — clean composite
    outputs/_bulk/vis/                                      — annotated with bbox + horizon
    outputs/_bulk/labels/                                   — YOLO HBB labels
    outputs/_bulk/_contact_{k:02d}.png                      — contact sheets
    outputs/_bulk/_manifest.csv                             — row per composite
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irpaste.paste import (  # noqa: E402
    paste_patch,
    load_background,
    augment_background,
    _feather_alpha,
)
from irpaste.viewcls import classify_background  # noqa: E402


_ORANGE = (0, 160, 255)
_CYAN = (255, 255, 0)


def _index_bgs(root: Path):
    """Index backgrounds by view type, caching bg_view to avoid re-classification."""
    side_paths, top_paths = [], []
    side_views, top_views = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        try:
            bg = load_background(p)
            v = classify_background(bg, return_info=True, filename=p.name)
        except Exception:
            continue
        if v.kind == "side":
            side_paths.append(p)
            side_views.append(v)
        else:
            top_paths.append(p)
            top_views.append(v)
    return (side_paths, side_views), (top_paths, top_views)


def _load_manifest(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load manifest.csv, return (side_entries, top_entries)."""
    side, top = [], []
    manifest = cache_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run pre_extract.py first")
    with manifest.open("r") as fh:
        for row in csv.DictReader(fh):
            row["on_horizon"] = row["on_horizon"] == "1"
            row["sim_horizon_row"] = float(row["sim_horizon_row"])
            entry = dict(row)
            (side if row["view"] == "side" else top).append(entry)
    return side, top


class _Shuffler:
    """Draw items without replacement; reshuffle when pool is exhausted."""

    def __init__(self, items, rng: np.random.Generator):
        self.items = list(items)
        self.rng = rng
        self._order: list[int] = []
        self._reshuffle()

    def _reshuffle(self):
        self._order = list(range(len(self.items)))
        self.rng.shuffle(self._order)

    def next(self):
        if not self.items:
            return None
        if not self._order:
            self._reshuffle()
        return self.items[self._order.pop()]


def _safe_stem(name: str, n: int = 16) -> str:
    s = "".join(c for c in name if c.isascii() and (c.isalnum() or c in "._-"))
    return (s or "x")[:n]


# ---------------------------------------------------------------------------
# Annotation helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _annotate_multi(
    comp_or_bgr: np.ndarray,
    results: list,
    tags: list[str],
    crop_x: int = 0,
    crop_y: int = 0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    if comp_or_bgr.ndim == 2:
        bgr = cv2.cvtColor(comp_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = comp_or_bgr.copy()
    for pr, tag in zip(results, tags):
        ys, xs = np.where(pr.mask_patch)
        if ys.size > 0:
            x = pr.paste_xy[0] + int(xs.min())
            y = pr.paste_xy[1] + int(ys.min())
            pw = int(xs.max()) - int(xs.min())
            ph = int(ys.max()) - int(ys.min())
        else:
            x, y = pr.paste_xy
            ph, pw = pr.mask_patch.shape
        x = int(round((x - crop_x) * scale_x))
        y = int(round((y - crop_y) * scale_y))
        pw = max(1, int(round(pw * scale_x)))
        ph = max(1, int(round(ph * scale_y)))
        cv2.rectangle(bgr, (x, y), (x + pw, y + ph), _ORANGE, 1)
        cv2.putText(
            bgr, tag, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            bgr, tag, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 0), 1, cv2.LINE_AA,
        )
    if results and results[0].bg_view.kind == "side":
        bv = results[0].bg_view
        if bv.horizon_curve is not None:
            pts = bv.horizon_curve.polyline(n=max(32, bgr.shape[1] // 8))
            adj = pts.astype(np.float64)
            adj[:, 0] = (adj[:, 0] - crop_x) * scale_x
            adj[:, 1] = (adj[:, 1] - crop_y) * scale_y
            cv2.polylines(bgr, [adj.round().astype(np.int32)], False, _CYAN, 1, cv2.LINE_AA)
        elif bv.horizon_row is not None:
            hr = int(round((bv.horizon_row - crop_y) * scale_y))
            cv2.line(bgr, (0, hr), (bgr.shape[1] - 1, hr), _CYAN, 1, cv2.LINE_AA)
    view_label = f"{'Side' if results[0].bg_view.kind == 'side' else 'Top'} view  n={len(results)}" if results else f"n={len(results)}"
    cv2.putText(
        bgr, view_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        bgr, view_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return bgr


def _compute_label(
    pr, crop_x: int, crop_y: int, out_w: int, out_h: int
) -> tuple[float, float, float, float]:
    ys, xs = np.where(pr.mask_patch)
    if ys.size == 0:
        mk_x0 = 0
        mk_y0 = 0
        pw = pr.mask_patch.shape[1]
        ph = pr.mask_patch.shape[0]
    else:
        mk_x0 = int(xs.min())
        mk_y0 = int(ys.min())
        pw = int(xs.max()) - mk_x0 + 1
        ph = int(ys.max()) - mk_y0 + 1
    px = pr.paste_xy[0] + mk_x0
    py = pr.paste_xy[1] + mk_y0
    ox0 = px - crop_x
    oy0 = py - crop_y
    ox1 = ox0 + pw
    oy1 = oy0 + ph
    ox0_c = max(0, ox0)
    oy0_c = max(0, oy0)
    ox1_c = min(out_w, ox1)
    oy1_c = min(out_h, oy1)
    if ox1_c <= ox0_c or oy1_c <= oy0_c:
        return 0.5, 0.5, 0.0, 0.0
    pw_c = ox1_c - ox0_c
    ph_c = oy1_c - oy0_c
    cx_n = (ox0_c + ox1_c) / 2.0 / out_w
    cy_n = (oy0_c + oy1_c) / 2.0 / out_h
    w_n = pw_c / out_w
    h_n = ph_c / out_h
    return (
        float(np.clip(cx_n, 0.0, 1.0)),
        float(np.clip(cy_n, 0.0, 1.0)),
        float(np.clip(w_n, 0.0, 1.0)),
        float(np.clip(h_n, 0.0, 1.0)),
    )


def _write_multi_label(
    label_path: Path, labels: list[tuple[float, float, float, float]]
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cx, cy, w, h in labels:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _build_contact_sheet(
    paths: list[Path], cols: int = 8, tile: int = 160
) -> np.ndarray:
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
        oy, ox = (tile - im.shape[0]) // 2, (tile - im.shape[1]) // 2
        canvas[oy : oy + im.shape[0], ox : ox + im.shape[1]] = im
        tiles.append(canvas)
    if not tiles:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i : i + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="outputs/_cache",
                    help="directory with .npz cache + manifest.csv from pre_extract.py")
    ap.add_argument("--bg-root", default="data/background/test_1")
    ap.add_argument("--out", default="outputs/_bulk")
    ap.add_argument("--n", type=int, default=None,
                    help="composites to generate (omit to process every cached target once)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--side-frac", type=float, default=0.80,
                    help="fraction of composites that should be side-view")
    ap.add_argument("--contact-rows", type=int, default=8)
    ap.add_argument("--contact-cols", type=int, default=8)
    ap.add_argument("--augment-bg", action="store_true",
                    help="randomly zoom-in and smart-crop background before pasting")
    ap.add_argument("--bg-scale-max", type=float, default=1.3,
                    help="upper bound of background zoom-in factor")
    ap.add_argument("--align-axis", action="store_true",
                    help="rotate ship principal axis parallel to bg horizon")
    ap.add_argument("--ship-scale-min", type=float, default=0.55)
    ap.add_argument("--ship-scale-max", type=float, default=0.90)
    ap.add_argument("--max-ships-per-bg", type=int, default=1,
                    help="max ships per background (1-5)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_clean = out / "clean"
    out_vis = out / "vis"
    out_labels = out / "labels"
    out_clean.mkdir(exist_ok=True)
    out_vis.mkdir(exist_ok=True)
    out_labels.mkdir(exist_ok=True)

    seed = (
        args.seed
        if args.seed is not None
        else int(np.random.default_rng().integers(0, 2**31))
    )
    print(f"seed: {seed}  (pass --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed)

    t0 = time.time()

    # --- Load manifest ---
    print("loading manifest ...")
    tgt_side, tgt_top = _load_manifest(Path(args.cache_dir))
    print(f"  targets: {len(tgt_side)} side, {len(tgt_top)} top")

    # --- Index backgrounds (with view pre-computed) ---
    print("indexing backgrounds ...")
    (bg_side_paths, bg_side_views), (bg_top_paths, bg_top_views) = _index_bgs(Path(args.bg_root))
    print(f"  bg: {len(bg_side_paths)} side, {len(bg_top_paths)} top")
    print(f"  index took {time.time() - t0:.1f}s")

    # --- Build work queue ---
    if args.n is None:
        _all = tgt_side + tgt_top
        _arr = np.arange(len(_all))
        rng.shuffle(_arr)
        _work_queue = [_all[int(i)] for i in _arr]
        n_total = len(_work_queue)
        s_side = s_top = None
        print(f"  exhaustive mode: {n_total} targets queued")
    else:
        _work_queue = None
        n_total = args.n
        s_side = _Shuffler(tgt_side, np.random.default_rng(seed + 1))
        s_top = _Shuffler(tgt_top, np.random.default_rng(seed + 2))

    cache_dir = Path(args.cache_dir)
    manifest_path = out / "_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["idx", "view", "bg", "target", "on_horizon", "paste_x", "paste_y", "out"]
        )

        produced = 0
        attempts = 0
        written: list[Path] = []
        per_tile = max(args.contact_rows * args.contact_cols, 16)
        _q_iter = iter(_work_queue) if _work_queue is not None else None
        _TARGET = 512
        max_per_bg = max(1, min(args.max_ships_per_bg, 5))

        pbar = tqdm(total=n_total, desc="Pasting", unit="comp")
        while produced < n_total:
            # --- Build batch: 1 to max_per_bg ships sharing one bg ---
            n_ships_this_batch = int(rng.integers(1, max_per_bg + 1))
            batch_entries: list[dict] = []
            batch_kind = None
            batch_bg_idx = None

            for _ in range(n_ships_this_batch):
                if _q_iter is not None:
                    try:
                        entry = next(_q_iter)
                    except StopIteration:
                        break
                    kind = entry["view"]
                    bg_pool_paths = bg_side_paths if kind == "side" else bg_top_paths
                    bg_pool_views = bg_side_views if kind == "side" else bg_top_views
                    if not bg_pool_paths:
                        continue
                    bg_idx = int(rng.integers(len(bg_pool_paths)))
                else:
                    attempts += 1
                    if attempts > n_total * 4:
                        break
                    is_side = rng.random() < args.side_frac
                    if is_side and (not bg_side_paths or not tgt_side):
                        is_side = False
                    if (not is_side) and (not bg_top_paths or not tgt_top):
                        is_side = True
                    kind = "side" if is_side else "top"
                    bg_pool_paths = bg_side_paths if is_side else bg_top_paths
                    bg_pool_views = bg_side_views if is_side else bg_top_views
                    entry = (s_side if is_side else s_top).next()
                    if entry is None:
                        continue
                    bg_idx = int(rng.integers(len(bg_pool_paths)))

                if batch_bg_idx is None:
                    batch_bg_idx = bg_idx
                    batch_kind = kind
                batch_entries.append(entry)

                if _q_iter is None and attempts > n_total * 4:
                    break

            if not batch_entries:
                break

            # --- Load bg ONCE, augment ONCE, classify ONCE per batch ---
            bg_path = (bg_side_paths if batch_kind == "side" else bg_top_paths)[batch_bg_idx]
            bg_view_orig = (bg_side_views if batch_kind == "side" else bg_top_views)[batch_bg_idx]
            try:
                bg = load_background(bg_path)
            except Exception as e:
                print(f"  skip batch (bg) {bg_path.name}: {e}")
                continue

            if args.augment_bg:
                bg = augment_background(bg, rng, scale_range=(1.0, args.bg_scale_max))
                bg_view = classify_background(bg, return_info=True, filename=bg_path.name)
            else:
                bg_view = bg_view_orig

            # --- Paste ships sequentially on the shared bg ---
            results: list = []
            occupied_mask = np.zeros(bg.shape, dtype=bool)

            for i, entry in enumerate(batch_entries):
                # Load cached patch + mask.
                npz_path = cache_dir / entry["cache_file"]
                try:
                    data = np.load(npz_path)
                    patch = data["patch"]
                    mask = data["mask"]
                    on_horizon = bool(data["on_horizon"])
                except Exception as e:
                    print(f"  skip {entry['stem']}: {e}")
                    continue

                try:
                    pr = paste_patch(
                        patch,
                        mask,
                        bg,
                        bg_view=bg_view,
                        target_on_horizon=on_horizon,
                        rng=np.random.default_rng(seed + 1000 + produced + i),
                        align_to_horizon=args.align_axis,
                        ship_scale_range=(args.ship_scale_min, args.ship_scale_max),
                        occupied_mask=occupied_mask if i > 0 else None,
                    )
                except Exception as e:
                    print(f"  skip {entry['stem']}: {e}")
                    continue

                # --- Update composite ---
                if i == 0:
                    composite_full = pr.composite.copy()
                else:
                    x, y = pr.paste_xy
                    ph, pw = pr.mask_patch.shape
                    H_c, W_c = composite_full.shape
                    x1 = min(x + pw, W_c)
                    y1 = min(y + ph, H_c)
                    ry, rx = y1 - y, x1 - x
                    if ry > 0 and rx > 0:
                        m_roi = pr.mask_patch[:ry, :rx]
                        # Feathered alpha from mask only — blend new ship pixels,
                        # leave background and previous ships untouched.
                        a = _feather_alpha(m_roi, dilate=0, sigma=0.8) * 0.9
                        roi_new = pr.composite[y:y1, x:x1].astype(np.float32)
                        roi_old = composite_full[y:y1, x:x1].astype(np.float32)
                        blended = roi_new * a + roi_old * (1.0 - a)
                        composite_full[y:y1, x:x1] = np.clip(blended, 0, 255).astype(np.uint8)

                # --- Update occupied_mask (dilate 2px for breathing room) ---
                x, y = pr.paste_xy
                ph, pw = pr.mask_patch.shape
                H_c, W_c = bg.shape
                x1 = min(x + pw, W_c)
                y1 = min(y + ph, H_c)
                ry, rx = y1 - y, x1 - x
                if ry > 0 and rx > 0:
                    m_occ = pr.mask_patch[:ry, :rx].copy()
                    # Dilate the mask used for occupancy to keep ships apart.
                    m_occ_u8 = m_occ.astype(np.uint8)
                    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    m_occ_u8 = cv2.dilate(m_occ_u8, k_dil)
                    fm = np.zeros(bg.shape, dtype=bool)
                    fm[y:y1, x:x1] = m_occ_u8.astype(bool)
                    occupied_mask = occupied_mask | fm

                results.append(pr)

            if not results:
                continue

            # --- Output naming ---
            batch_idx = produced
            first_stem = batch_entries[0]["stem"]
            out_name = (
                f"{batch_idx:06d}_{batch_kind}"
                f"_{_safe_stem(bg_path.stem)}"
                f"_{_safe_stem(Path(first_stem).name)}"
                f"_n{len(results)}.png"
            )

            # --- Crop/resize to 512x512 ---
            H_full, W_full = composite_full.shape
            clean_full = cv2.cvtColor(composite_full, cv2.COLOR_GRAY2BGR)
            tags = [
                f"ship{j+1}{' H' if pr.target_on_horizon else ''}"
                for j, pr in enumerate(results)
            ]
            if H_full >= _TARGET and W_full >= _TARGET:
                _y0 = (H_full - _TARGET) // 2
                _x0 = (W_full - _TARGET) // 2
                clean_out = clean_full[_y0 : _y0 + _TARGET, _x0 : _x0 + _TARGET]
                vis_out = _annotate_multi(clean_out, results, tags, crop_x=_x0, crop_y=_y0)
            else:
                _y0, _x0 = 0, 0
                scale_x = _TARGET / max(W_full, 1)
                scale_y = _TARGET / max(H_full, 1)
                clean_out = cv2.resize(
                    clean_full, (_TARGET, _TARGET), interpolation=cv2.INTER_LINEAR
                )
                vis_out = _annotate_multi(clean_out, results, tags, scale_x=scale_x, scale_y=scale_y)

            cv2.imwrite(str(out_clean / out_name), clean_out)
            out_vis_path = out_vis / out_name
            cv2.imwrite(str(out_vis_path), vis_out)
            written.append(out_vis_path)

            # --- YOLO HBB labels ---
            out_w = clean_out.shape[1]
            out_h = clean_out.shape[0]
            label_lines: list[tuple[float, float, float, float]] = []
            for pr in results:
                cx_n, cy_n, w_n, h_n = _compute_label(pr, _x0, _y0, out_w, out_h)
                label_lines.append((cx_n, cy_n, w_n, h_n))
            lbl_path = out_labels / Path(out_name).with_suffix(".txt").name
            _write_multi_label(lbl_path, label_lines)

            # --- Manifest row ---
            target_names = ";".join(Path(e["stem"]).name for e in batch_entries)
            w.writerow([
                batch_idx,
                batch_kind,
                bg_path.name,
                target_names,
                int(results[0].target_on_horizon),
                results[0].paste_xy[0],
                results[0].paste_xy[1],
                f"clean/{out_name}",
            ])

            produced += 1
            pbar.update(1)
            pbar.set_postfix(rate=f"{produced / max(time.time() - t0, 1e-3):.1f}/s")

        pbar.close()

    # --- Contact sheets ---
    print("building contact sheets ...")
    for k, start in enumerate(range(0, len(written), per_tile)):
        sheet = _build_contact_sheet(
            written[start : start + per_tile],
            cols=args.contact_cols,
            tile=160,
        )
        cv2.imwrite(str(out / f"_contact_{k:02d}.png"), sheet)

    print(
        f"wrote {produced} composites -> {out_clean} (clean)  {out_vis} (vis)   "
        f"(total {time.time() - t0:.1f}s,   "
        f"manifest: {manifest_path.name})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify import**

```bash
cd D:/projects/teamwork/hw_data_process/IRPaste-main && uv run python -c "from irpaste.paste import paste_patch, _feather_alpha; print('OK')"
```

Expected: `OK`

Note: `_feather_alpha` is a private function in `paste.py`. We need it for the compositing fix in `paste_bulk.py`. Since it's private, we should either make it public or import it carefully. Check that it's accessible:

```bash
cd D:/projects/teamwork/hw_data_process/IRPaste-main && uv run python -c "from irpaste.paste import _feather_alpha; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/paste_bulk.py
git commit -m "refactor: paste_bulk uses pre-extracted cache, shared bg augment, tqdm, fix overlap/composite"
```

---

### Task 5: End-to-end smoke test

**Files:** none (test only)

- [ ] **Step 1: Run pre_extract on a small subset**

First check what targets are available:

```bash
ls D:/projects/teamwork/hw_data_process/IRPaste-main/data/
```

Then run pre_extract on a subset (if available):

```bash
cd D:/projects/teamwork/hw_data_process/IRPaste-main && uv run python scripts/pre_extract.py --targets-root data/burkeIIA长波 --cache-dir outputs/_cache_test
```

- [ ] **Step 2: Run paste_bulk with 5 composites**

```bash
cd D:/projects/teamwork/hw_data_process/IRPaste-main && uv run python scripts/paste_bulk.py --cache-dir outputs/_cache_test --n 5 --seed 42 --max-ships-per-bg 2 --augment-bg
```

Expected: 5 composites produced, no errors, visual inspection shows ships below horizon and not overlapping.

- [ ] **Step 3: Verify outputs exist**

```bash
ls D:/projects/teamwork/hw_data_process/IRPaste-main/outputs/_bulk/clean/ | head
ls D:/projects/teamwork/hw_data_process/IRPaste-main/outputs/_bulk/vis/ | head
ls D:/projects/teamwork/hw_data_process/IRPaste-main/outputs/_bulk/labels/ | head
```

- [ ] **Step 4: Commit if changes needed**

Only if fixes were required during testing.
