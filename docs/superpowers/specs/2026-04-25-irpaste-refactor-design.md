# IRPaste 批量张贴管线重构

Date: 2026-04-25

## 问题

1. 多艘船只张贴时重合 — occupied_mask 使用侵蚀后的 mask，且 fallback 跳过 overlap 检查
2. 船只出现在海天线上方 — side-view fallback 不遵守 horizon 约束
3. 每艘船重复做 augment_background + classify_background，且各船贴到不同的 augmented 背景上
4. 无 tqdm 进度条

## 方案：两阶段管线 + 磁盘缓存

### 阶段一：`scripts/pre_extract.py` — mask 预提取

```
扫描所有 .xml → 对每个 target:
  1. classify_target(xml) → view (side/top)
  2. load_sample(stem) → Sample
  3. build_mask(sample) → mask
  4. target_patch_from_sample(sample, mask) → (patch, mask, bbox)
  5. detect_target_on_horizon(sample, mask) → (on_horizon, sim_hr)
  6. 保存 {stem}.npz 到 cache_dir
```

**`.npz` 内容：**
- `patch`: uint8 (H,W) — 原始 target patch（未缩放、未侵蚀）
- `mask`: bool (H,W) — 原始 mask（完整，不侵蚀）
- `view`: str — "side" / "top"
- `on_horizon`: bool
- `sim_horizon_row`: float32（没有时存 -1）
- `stem`: str
- `pitch`: float32

**CLI：**
```
uv run python scripts/pre_extract.py \
  --targets-root data/burkeIIA长波 \
  --cache-dir outputs/_cache \
  --resume
```

**进度：** tqdm，显示总数、成功、跳过、失败。

**manifest.csv：** `stem,view,on_horizon,cache_file`

---

### 阶段二：`scripts/paste_bulk.py` — 批量张贴

**流程：**

```
1. 加载 manifest.csv → side/top 两个列表 → _Shuffler
2. 加载背景池 → 按 view 分组

对每个批次:
  3. 随机选 1~max_ships_per_bg 艘同 view target（从 Shuffler）
  4. 随机选 1 个同 view background
  5. 加载 bg → augment 1 次 → classify 1 次 → 批次共享
  6. 对每艘船:
     a. 从 .npz 加载 patch/mask/on_horizon
     b. 缩放/旋转 patch 和 mask（原 paste_target 逻辑）
     c. choose_paste_site(bg, bg_view, ..., occupied_mask=occupied_mask)
     d. radiometric_match + blend → PasteResult
     e. 更新 occupied_mask（用 dilated mask, dilate=2px）
     f. 用 feathered mask 叠加到 composite_full
  7. 裁剪 512×512 → 写 clean/vis/label
```

**occupied_mask 修复：**
- 记录时对 mask_patch dilate 2px 再标记为占用
- 补偿 mask 在 blend 前被侵蚀的 1px

**多船 compositing 修复：**
- 不用 `pr.composite`（全图）覆盖，改用 feathered alpha mask 只 blend 前景像素
- 保持 composite_full 的已有内容不被覆盖

**放置 fallback 修复（`paste.py:choose_paste_site`）：**
- retry 耗尽后 side-view 同样遵守 horizon 约束
- top-down fallback 也检查 overlap

**进度：** 外层 tqdm：`{produced}/{n_total} | {rate:.1f}/s`

### `irpaste/paste.py` 改动

**新增 `paste_patch()` 函数：**
- 入参：`patch: np.ndarray`, `mask: np.ndarray`（已 tight-crop）、`bg`, `bg_view`, `target_on_horizon`, `paste_xy` 等
- 处理：缩放 → 旋转 → re-tight-crop → 侵蚀 → radiometric_match → blend → boundary_blur → noise → tv_smooth
- `paste_target` 保持不动，内部调用 `paste_patch`
- 目的：阶段二从 .npz 加载已裁剪的 patch/mask 后直接调用，绕过 `load_sample` + `build_mask` + `target_patch_from_sample`

**修复 `choose_paste_site`：**
- retry 耗尽 fallback：side-view 保持 horizon 约束；top-down 也检查 overlap

### 不改动的文件
- `irpaste/extract.py`
- `irpaste/viewcls.py`
- `irpaste/io_utils.py`
