# IRPaste-main — 红外舰船合成数据集生成工具

从红外仿真数据（.dat 辐亮度图 + .xml 标注）中提取舰船掩码，粘贴到真实海面背景上，生成带 YOLO 标签的合成图像数据集。

---

## 快速开始

```bash
uv sync                              # 安装依赖 (Python ≥ 3.12)
uv run python scripts/paste_bulk.py --n 512 --seed 7
```

---

## 使用

```bash
# 批量生成合成图像（推荐）
uv run python scripts/paste_bulk.py --n 2000 --blend-mode laplacian --tv

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

## 功能特性

### 1. 多舰船合成（新增）

- **每背景可贴 1–5 艘船**，由 `--max-ships-per-bg` 控制（默认 1，最大 5）
- 通过 `occupied_mask`（布尔画布）追踪已占用像素，重叠率超过 8% 自动重试（最多 20 次）
- 后续舰船使用软 Alpha 混合（`α = 0.85`）叠加到已合成图像上
- 每船位置独立搜索，支持任意排列组合

### 2. YOLO HBB 标签（改进）

- **自动裁剪透明边框**：掩码提取后裁掉透明像素，保留最小外接矩形
- **边界钳位**：标签坐标钳制到 [0, 1] 范围，防止放大/裁剪导致坐标越界
- **多行标签**：每船一行 `0 cx cy w h`，支持多目标检测训练
- **归一化坐标**：基于可见像素（alpha > 0）计算紧包围框

### 3. 海天线检测（优化）

参考 `irpaste/viewcls.py`：

- **空间一致性滤波**：滑动窗口 MAD 剔除异常列峰值（舰船/陆地/死像素干扰）
- **多尺度 Sobel 融合**：融合 ksize=3 和 ksize=5 的 Sobel-Y 响应，兼顾锐利和弥散边缘
- **LO-RANSAC 迭代优化**：在 RANSAC 最佳模型上做加权最小二乘精化，降低 RMSE
- **置信度检查**：双峰残差分布检测（拒绝将船体/陆地误检为海天线）
- **斜率约束**：海天线斜率限制在 ±18° 内，避免追踪陡峭陆地/云层边界
- **多峰验证**：主梯度峰失败时尝试次强峰，处理雾层等双重海天线场景
- **引导采样**：三桶策略保证 RANSAC 三元组覆盖全图宽度，避免退化

### 4. 融合方式

| 方式 | 特点 |
|------|------|
| `alpha` | 软 Alpha 混合（最快） |
| `laplacian` | 3 级拉普拉斯金字塔（推荐，边界最平滑） |
| `poisson` | `cv2.seamlessClone(NORMAL_CLONE)`（保真度高） |

### 5. 输出

```
outputs/_bulk/
├── clean/{idx:06d}_{view}_{bg}_{target}_n{ships}.png  — 合成图像
├── vis/{idx:06d}_{view}_{bg}_{target}_n{ships}.png     — 标注可视化（边界框+海天线）
├── labels/{idx:06d}_{view}_{bg}_{target}_n{ships}.txt  — YOLO HBB 多行标签
├── _contact_*.png                                       — 8×8 缩略图索引
└── _manifest.csv                                        — 合成清单（行=合成图像）
```

---

## 批量生成参数（paste_bulk.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | `None` | 生成数量（省略则处理全部目标） |
| `--seed` | 随机 | 随机种子 |
| `--max-ships-per-bg` | 1 | 每背景最多贴船数（1–5） |
| `--side-frac` | 0.80 | 侧视图占比 |
| `--augment-bg` | False | 随机放大+智能裁剪背景 |
| `--bg-scale-max` | 1.3 | 背景放大上限 |
| `--ship-scale-min` | 0.55 | 舰船缩小下限 |
| `--ship-scale-max` | 0.90 | 舰船缩小上限 |
| `--align-axis` | False | 舰船主轴平行海天线（侧视） |

---

## 核心库（irpaste/）

| 模块 | 功能 |
|------|------|
| `io_utils.py` | 加载 .dat 辐亮度 + .xml 标注 |
| `extract.py` | 7 步自适应阈值掩码提取 |
| `viewcls.py` | 视角分类 + 二次 RANSAC 海天线拟合 |
| `paste.py` | 合成引擎（Alpha/Poisson/Laplacian） |

### 掩码提取（7 步算法）

1. 扩展 XML 包围框 5%
2. 用目标列以外的列检测海天线
3. 按海天线分割天空/海洋区域
4. MAD 评分取最纯净采样带，中位数估计背景辐亮度
5. 迟滞阈值分割：`T_high = k_high × 1.4826 × MAD`，`T_low = max(0.4 × T_high, k_low × 1.4826 × MAD)`
6. Sobel 梯度辅助恢复桅杆等细线结构
7. 形态学闭运算 → 连通域筛选

### 视角分类

- **目标视角**：从 XML `<imageSensor pitch>` 读取，`pitch ≤ -80°` 为俯视
- **背景视角**：基于海天线检测（行均值梯度峰 + 直方图双峰性），支持文件名前缀覆盖

### 海天线算法

`fit_horizon_curve()` 使用二次曲线 `y = ax² + bx + c` 拟合海天线：

1. 双边滤波保留边缘去噪
2. 多尺度 Sobel-Y 融合（ksize=3 + ksize=5）
3. 空间一致性滤波（MAD 法剔除离群列峰值）
4. 引导采样 RANSAC（三桶策略覆盖全宽）
5. LO-RANSAC 加权最小二乘精化
6. 双峰残差置信度检查

---

## 环境要求

- Python ≥ 3.12
- Windows / Linux
- 依赖：`numpy`、`opencv-python`、`pillow`、`scikit-image`
- 包管理：`uv`
