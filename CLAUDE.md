# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# All commands run from IRPaste-main/ root. Package management uses uv.
uv sync                              # install dependencies (Python ≥ 3.12)
uv run python scripts/paste_bulk.py --n 512 --seed 7
uv run python scripts/test_pipeline.py --sample-dir data/... --bg-dir background/...
uv run python scripts/extract_demo.py --folder data/... --limit 20
uv run python scripts/paste_demo.py --n 30 --method laplacian --tv
uv run python scripts/run_all.py     # full-dataset QA + bad-sample report
uv run python scripts/compare_tv.py  # TV-L1 vs no-TV seam comparison
```

## Architecture

### Core library (`irpaste/`)

Four modules exposing a pipeline: **load sample → extract mask → paste onto background**.

**`io_utils.py`** — Data loading:
- `load_sample(stem)` → `Sample(radiance, preview, annotation)` — loads `.dat` radiance + `.xml` annotation + optional `.png/.bmp` preview
- `.dat` format: 8-byte header (`uint32 w, uint32 h`) + `float32` pixels, row-major
- `.xml` format: `<imageSensor pitch="-30">` → `<targets><target centerX=... centerY=... width=... height=... pixelNum=...>` with 4 normalised corner points
- `Annotation.anchor_xyxy()` returns the union of XML bbox and corner-polygon AABB — this is the tightest guaranteed ship-enclosing rectangle

**`extract.py`** — Mask extraction (`build_mask`), 7-step algorithm:
1. Expand XML anchor by 5% → define context window → collect 4 side bands outside bbox
2. Detect sea/sky horizon using **only left+right columns** (not target columns) via row-median jump test
3. If horizon is near bbox, split bands into above/below sub-regions processed independently
4. Score bands by purity (MAD/spread), keep cleanest → robust background model (per-row median from lateral strips)
5. Hysteresis thresholding on residual: `T_high = k_high × 1.4826 × MAD`, `T_low = max(0.4 × T_high, k_low × 1.4826 × MAD)`
6. Edge-aided recovery: Sobel gradient on residual recovers thin mast structures
7. Morphology close → connected-component selection (prefer component closest to anchor center) → horizon-bleed trim → polygon-clip final mask

**`viewcls.py`** — View classification + RANSAC horizon curve fitting:
- `classify_background(gray)` → `"side"` or `"top"` — bilateral-filter row-mean step detection + histogram bimodality check
- `fit_horizon_curve(gray)` → `HorizonCurve(a, b, c)` — quadratic `y = ax² + bx + c` fitted via guided-sampling RANSAC + LO-RANSAC refinement + bimodal confidence guard
- `classify_target(xml_path)` → `"side"` or `"top"` — pitch ≤ -80° is top

**`paste.py`** — Compositing (`paste_target`):
- Background zoom augmentation (`augment_background`) with smart crop scoring
- Ship downscaling (`ship_scale_range`, default 0.55–0.90)
- Optional principal-axis/horizon alignment (`align_to_horizon`) via PCA angle + rotation
- View-aware paste-site selection: side-view ships placed near/below horizon; top-down ships anywhere
- Radiometric matching: shift target mean to `bg_med + Δ` preserving thermal polarity; contrast clamp at 8× bg_std
- Three blend methods: `alpha` (feathered), `poisson` (seamlessClone NORMAL_CLONE), `laplacian` (3-level pyramid, recommended)
- Adaptive boundary blur → optional noise injection → optional TV-L1 boundary smoothing

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `paste_bulk.py` | High-throughput batch generation, outputs composite + YOLO label + manifest CSV |
| `test_pipeline.py` | End-to-end smoke test with debug visualisation |
| `extract_demo.py` | Mask extraction quality inspection (triptych overlay) |
| `paste_demo.py` | Single-paste demo with parameter exploration |
| `run_all.py` | Full-dataset QA: extract all samples, flag bad masks |
| `compare_tv.py` | TV-L1 smoothing comparison experiment |

### Key design decisions

- **All images are uint8 grayscale** end-to-end (IR domain). Radiance data (.dat) is float32, used only for mask extraction.
- **Chinese path support**: Use `np.fromfile()` + `cv2.imdecode()` for reading, never `cv2.imread()`.
- **Blend default**: `laplacian` + `tv_smooth=True` — lowest seam gradient (~23% below plain alpha) while preserving IR radiometric contrast.
- **No pixel-level overlap avoidance** — each paste_target call handles one ship. Multi-ship compositing would need external orchestration.
- **ship_scale_range** default is (0.55, 0.90) — ships are *downscaled* to simulate distance. In contrast, ship_paste_project uses upscaling (1.15–1.65) for visibility.
- **Output size** is always 512×512 (cropped from the background, not the target).
