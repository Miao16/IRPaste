# Horizon Calibration Tool & View-Classification Refactor

## Goal

1. Interactive tool to review, adjust, and save horizon curves for background images
2. Replace filename-based side/top classification with cache-file-driven classification
3. Rename background files to a clean `side_XXXXXX` / `top_XXXXXX` scheme

## Non-goals

- Changing the horizon detection algorithm itself
- Changing the paste pipeline placement logic
- Adding new blend methods or augmentation strategies

---

## Component A: Interactive Calibration Script (`scripts/calibrate_horizon.py`)

### Dependencies

OpenCV `highgui` (already a project dependency). No new packages.

### Workflow

```
for each image in bg/:
    1. load image
    2. run classify_background() → initial horizon curve
    3. show image + curve overlay in OpenCV window
    4. wait for user input (mouse + keyboard)
    5. on 's' → save cache .json, rename file, advance
    6. on 'd' → mark skip, advance
    7. on 'q' → quit, save progress
```

### Mouse interaction

- **Left-click** near a control point (within 8px) → select it (highlighted with white ring)
- **Left-click** elsewhere on image → add control point (green circle); this becomes the selected point
- **Right-click** on a control point → remove it
- When ≥3 control points exist → re-fit quadratic `y = ax² + bx + c` via `np.polyfit` on those points
- When <3 control points → auto-computed RANSAC curve is shown
- Control points are displayed as 6px green filled circles

### Keyboard controls

| Key | Action |
|-----|--------|
| `s` | Save — write `.json` cache, rename file to `side_XXXXXX.png` or `top_XXXXXX.png`, advance to next |
| `d` | Skip — add to `_skip.json`, advance to next |
| `t` | Toggle view kind between `side` and `top` (displayed in title bar) |
| `r` | Reset — clear manual points, restore auto-computed curve |
| `c` | Clear — remove all control points (curve becomes straight line at last row) |
| `q` / `Esc` | Quit — save progress (next_index to `_progress.json`) |
| `p` | Previous image |
| `←` / `→` | Nudge selected control point 1px left/right |
| `↑` / `↓` | Nudge selected control point 1px up/down |

### Visualization overlay

- **Cyan curve** — current horizon (manual if ≥3 points, else auto)
- **Green dots** — manual control points
- **Yellow dashed line** — auto-computed curve (always shown for reference, even when manual is active)
- **Title bar** — `[3/200] side | ctrl pts: 4 | S=save D=skip T=toggle`
- **Blue tint** — region above horizon (sky) gets a light blue semi-transparent overlay; below horizon (sea/water) is un-tinted. This gives immediate visual feedback about where ships will be placed.

### File renaming

On save:
1. Determine view kind (`side` or `top`)
2. Scan existing files in `bg/` to find the max sequence number for that kind
3. Assign next number: `side_000042.png` / `top_000042.json`
4. Rename both the image and its `.json` cache
5. Log old→new mapping to `bg/_rename_log.csv`

### Skip list (`bg/_skip.json`)

```json
{
  "version": 1,
  "skipped": ["000025", "000074"],
  "notes": {}
}
```

Skipped files are never renamed and are excluded from the paste pipeline.

### Progress file (`bg/_progress.json`)

```json
{
  "version": 1,
  "next_index": 47,
  "total": 200
}
```

Allows resuming after quit.

---

## Component B: Remove Filename-Based Classification

### Files changed

**`irpaste/viewcls.py`** — remove lines 449-458 (the filename-override block in `classify_background()`). The `filename` parameter is kept as a no-op for backward compat, deprecated.

```python
# REMOVE this block:
#     # Filename-based override (authoritative naming convention).
#     if filename is not None:
#         stem = Path(filename).stem
#         first = stem[0] if stem else ""
#         if first.isalpha():
#             is_side = True
#             ...
#         elif first.isdigit():
#             is_side = False
```

**`irpaste/horizon_cache.py`** — no API changes needed. Already handles `load_or_compute()`.

**`scripts/paste_bulk.py`** — `_index_bgs()` (lines 47-65) changes to use `load_or_compute()`:

```python
from irpaste.horizon_cache import load_or_compute

def _index_bgs(root, skip_set):
    side_paths, side_views = [], []
    top_paths, top_views = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg", ".tif"}:
            continue
        stem = p.stem
        # Skip files without proper prefix (not yet calibrated)
        if not (stem.startswith("side_") or stem.startswith("top_")):
            continue
        # Skip user-marked files
        if stem in skip_set:
            continue
        try:
            bg = load_background(p)
            data = load_or_compute(p, bg)
            v = data.to_background_view()
        except Exception:
            continue
        if v.kind == "side":
            side_paths.append(p)
            side_views.append(v)
        else:
            top_paths.append(p)
            top_views.append(v)
    return (side_paths, side_views), (top_paths, top_views)
```

The `_skip.json` is loaded at startup and used to filter backgrounds.

---

## File summary

| File | Action |
|------|--------|
| `scripts/calibrate_horizon.py` | **New** — interactive calibration tool |
| `irpaste/viewcls.py` | **Edit** — remove filename override block |
| `scripts/paste_bulk.py` | **Edit** — `_index_bgs()` uses horizon cache |
| `bg/_skip.json` | **New** (runtime artifact) — skip list |
| `bg/_progress.json` | **New** (runtime artifact) — resume state |
| `bg/_rename_log.csv` | **New** (runtime artifact) — rename audit trail |

## Error handling

- Corrupt/unreadable images: skip with warning, continue to next
- `classify_background()` crashing: treat as "top" view, allow manual toggle
- No images left after skipping: print message, exit cleanly
- Window closed via X button: same as `q` — save progress

## Testing

Manual verification:
1. Run `uv run python scripts/calibrate_horizon.py --bg-root bg/` on the full bg directory
2. Verify mouse-click adds points and curve re-fits
3. Verify `s` renames files correctly
4. Verify `d` adds to skip list
5. Verify `q` + restart resumes correctly
6. Run `uv run python scripts/paste_bulk.py` and confirm it uses renamed files + cache
