# IRPaste — 红外舰船合成数据集生成工具

从红外仿真数据（`.dat` 辐亮度图 + `.xml` 标注）中提取舰船掩码，粘贴到真实海面背景上，生成带 YOLO 标签的合成图像数据集。

---

## 快速开始

```bash
uv sync                              # 安装依赖 (Python ≥ 3.12)
uv run python scripts/paste_bulk.py --n 512 --seed 7
```

---

## 工作流程

### 1. 海天线标定（一次性）

在粘贴之前，先对背景图像进行海天线检测和手工校准：

```bash
uv run python scripts/calibrate_horizon.py --bg-root bg/
```

交互操作：

| 操作 | 说明 |
|------|------|
| 鼠标左键（空白处） | 添加控制点 |
| 鼠标左键（靠近点 ≤8px） | 选中该控制点 |
| 鼠标右键（靠近点） | 删除该控制点 |
| `h` `j` `k` `l` | 微调控点（左/下/上/右） |
| `s` | 保存并重命名为 `side_000001.png` / `top_000001.png` |
| `d` | 跳过（加入 `_skip.json`） |
| `t` | 切换视角类型（side ↔ top） |
| `r` | 重置为自动检测曲线 |
| `c` | 清除所有控制点 |
| `p` | 上一张 |
| `q` / `Esc` | 退出（自动保存进度，下次启动可续接） |

- 当 ≥3 个控制点时，自动拟合二次曲线 `y = ax² + bx + c`
- 蓝色半透明遮罩标记天空区域（海天线以上），下方为可粘贴区域
- 保存后在 `bg/` 生成 `side_XXXXXX.json` / `top_XXXXXX.json` 缓存文件
- 进度文件 `_progress.json` 支持中断续接

### 2. 掩码预提取（可选但推荐批量生成时使用）

```bash
uv run python scripts/pre_extract.py --targets-root data/... --cache-dir outputs/_cache
```

将目标舰船图像批量提取为 `.npz` 缓存（patch + mask），避免批量生成时重复计算。

### 3. 批量合成

```bash
uv run python scripts/paste_bulk.py --n 512 --seed 7 --cache-dir outputs/_cache
```

背景图像自动按 `side_` / `top_` 前缀和 `_skip.json` 过滤，并通过 `.json` 缓存读取海天线。

---

## 命令参考

```bash
# 批量生成（二阶段流程）
uv run python scripts/pre_extract.py --targets-root data/simulation --cache-dir outputs/_cache
uv run python scripts/paste_bulk.py --n 2000 --seed 7 --cache-dir outputs/_cache

# 流水线端到端测试
uv run python scripts/test_pipeline.py --sample-dir data/... --bg-dir background/...

# 掩码提取质量检查
uv run python scripts/extract_demo.py --folder data/... --limit 20

# 单张粘贴演示
uv run python scripts/paste_demo.py --n 30 --method laplacian --tv

# 全数据集质量评估
uv run python scripts/run_all.py

# TV-L1 平滑对比实验
uv run python scripts/compare_tv.py
```

---

## 输出结构

```
outputs/_bulk/
├── clean/{idx:06d}_{view}_{bg}_{target}_n{ships}.png  — 合成图像 (512×512)
├── vis/{idx:06d}_{view}_{bg}_{target}_n{ships}.png     — 标注可视化（边界框 + 海天线）
├── labels/{idx:06d}_{view}_{bg}_{target}_n{ships}.txt  — YOLO HBB 标签
├── _contact_*.png                                       — 缩略图索引
└── _manifest.csv                                        — 合成清单
```

---

## 批量生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache-dir` | `outputs/_cache` | 预提取缓存目录 |
| `--bg-root` | `data/background/test_1` | 背景图像目录 |
| `--n` | `None` | 生成数量（省略则处理全部） |
| `--seed` | 随机 | 随机种子 |
| `--max-ships-per-bg` | 1 | 每背景最多贴船数（1–5） |
| `--side-frac` | 0.80 | 侧视图占比 |
| `--augment-bg` | False | 随机放大 + 智能裁剪背景 |
| `--bg-scale-max` | 1.3 | 背景放大上限 |
| `--ship-scale-min` | 0.55 | 大船缩小倍率（更大的缩小） |
| `--ship-scale-max` | 0.90 | 小船缩小倍率（保持较大） |
| `--max-bbox-px` | — | 限制 bbox 最长边像素数（如 125） |
| `--align-axis` | False | 舰船主轴对齐海天线（侧视） |

---

## 核心库（irpaste/）

| 模块 | 功能 |
|------|------|
| `io_utils.py` | 加载 `.dat` 辐亮度 + `.xml` 标注，支持中文路径 |
| `extract.py` | 7 步自适应阈值掩码提取 |
| `viewcls.py` | 背景视角分类 + 二次 RANSAC 海天线拟合 |
| `horizon_cache.py` | 海天线 JSON 缓存读写，支持跨会话持久化 |
| `paste.py` | 合成引擎（Alpha / Poisson / Laplacian） |

### 掩码提取流程

1. 扩展 XML 包围框 5%
2. 用目标列以外的左右侧列检测海天线
3. 按海天线分割天空/海洋子区域
4. MAD 纯净度评分，取最优采样带估计背景辐亮度
5. 迟滞阈值分割：`T_high = k_high × 1.4826 × MAD`，`T_low = max(0.4 × T_high, k_low × 1.4826 × MAD)`
6. Sobel 梯度辅助恢复桅杆等细线结构
7. 形态学闭运算 → 连通域筛选 → 海天线渗漏裁剪 → 多边形最终截断

### 海天线检测

`fit_horizon_curve()` 使用二次曲线 `y = ax² + bx + c` 拟合海天线：

1. 双边滤波保留边缘去噪
2. 多尺度 Sobel-Y 融合（ksize=3 + ksize=5）
3. 空间一致性滤波（滑动窗口 MAD 剔除离群列峰值）
4. 三桶引导采样 RANSAC（覆盖全图宽度避免退化）
5. LO-RANSAC 加权最小二乘精化
6. 双峰残差置信度检查（排除船体/陆地误检）
7. 多峰验证（主梯度峰失败时尝试次强峰）

### 视角分类

- **目标视角**：从 XML `<imageSensor pitch>` 读取，`pitch ≤ -80°` 为俯视
- **背景视角**：通过海天线检测自动判定，结果缓存到 `.json` 文件
- **人工校准**：`calibrate_horizon.py` 支持手工调整并保存，分类由缓存文件驱动

### 背景文件命名规范

标定后背景文件重命名为：

```
bg/
├── side_000001.png      # 侧视图（有海天线，舰船贴在海天线以下）
├── side_000001.json     # 海天线缓存数据
├── top_000001.png       # 俯视图（无海天线，舰船可任意位置）
├── top_000001.json
├── _skip.json           # 跳过的背景列表
├── _progress.json       # 标定进度（支持中断续接）
└── _rename_log.csv      # 重命名审计日志
```

### 融合方式

| 方式 | 特点 |
|------|------|
| `alpha` | 软 Alpha 混合（最快） |
| `laplacian` | 3 级拉普拉斯金字塔（推荐，边界最平滑） |
| `poisson` | `cv2.seamlessClone(NORMAL_CLONE)`（保真度高） |

---

## 环境要求

- Python ≥ 3.12
- Windows / Linux
- 依赖：`numpy`、`opencv-python-headless`、`pillow`、`scikit-image`、`scipy`、`tifffile`、`tqdm`
- 包管理：`uv`
