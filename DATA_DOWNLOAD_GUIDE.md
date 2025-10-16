# BCI Competition IV 数据集下载指南

## 数据集概览

EEG-Conformer论文使用了两个数据集:

### 1. BCI Competition IV Dataset 2a (推荐先用这个)
- **任务**: 4分类运动想象 (左手、右手、双脚、舌头)
- **受试者**: 9人
- **EEG通道**: 22个电极
- **采样率**: 250 Hz
- **每试次时长**: 4秒 (1000时间点)
- **论文报告准确率**: 78.66% (hold-out)

### 2. BCI Competition IV Dataset 2b
- **任务**: 2分类运动想象 (左手 vs 右手)
- **受试者**: 9人
- **EEG通道**: 3个电极 (C3, Cz, C4)
- **采样率**: 250 Hz
- **每试次时长**: 4秒 (1000时间点)
- **论文报告准确率**: 84.63% (hold-out)

---

## 方法1: 官方网站下载 (手动下载)

### Dataset 2a 下载步骤:

1. **访问官方网站**:
   https://www.bbci.de/competition/iv/

2. **找到 Dataset 2a 部分**:
   在页面中找到 "Dataset 2a: Motor imagery, multi-class"

3. **下载数据文件**:
   点击 "Download" 链接下载以下文件:
   ```
   A01T.gdf - Subject 1 Training data
   A01E.gdf - Subject 1 Evaluation data
   A02T.gdf - Subject 2 Training data
   A02E.gdf - Subject 2 Evaluation data
   ...
   A09T.gdf - Subject 9 Training data
   A09E.gdf - Subject 9 Evaluation data

   true_labels.mat - Evaluation真实标签
   ```

4. **数据格式说明**:
   - 原始数据格式: `.gdf` (GDF - General Data Format for Biosignals)
   - 需要转换为 `.mat` 格式供EEG-Conformer使用

### Dataset 2b 下载步骤:

1. **同样在官方网站**找到 "Dataset 2b: Motor imagery, binary classification"

2. **下载数据文件**:
   ```
   B0101T.gdf - Subject 1, Session 1, Training
   B0102T.gdf - Subject 1, Session 2, Training
   B0103T.gdf - Subject 1, Session 3, Training
   B0104E.gdf - Subject 1, Session 4, Evaluation
   B0105E.gdf - Subject 1, Session 5, Evaluation
   ...
   (每个受试者有5个session文件)
   ```

---

## 方法2: 使用MNE-Python自动下载 (推荐)

我可以帮你创建一个Python脚本,使用MNE库自动下载和预处理数据集。

### 数据集2a下载脚本:

```python
import mne
from mne.datasets import bci2000
import numpy as np
import scipy.io

# MNE可以下载BCI Competition数据
# 但官方BCI Competition IV 2a/2b不在MNE默认数据集中
# 需要手动下载或使用第三方工具
```

---

## 方法3: 使用MOABB库 (最简单,强烈推荐)

MOABB (Mother of All BCI Benchmarks) 提供了自动下载和预处理功能。

### 安装MOABB:
```bash
pip install moabb
```

### 下载并转换数据集的Python脚本:

我会为你创建一个完整的数据下载和预处理脚本 `download_bci_data.py`

---

## 推荐的数据存储结构

下载后,建议按以下结构组织数据:

```
EEG-Conformer/
├── data/
│   ├── 2a/                    # Dataset 2a
│   │   ├── A01T.mat           # Subject 1 Training
│   │   ├── A01E.mat           # Subject 1 Evaluation
│   │   ├── A02T.mat
│   │   ├── A02E.mat
│   │   └── ...
│   └── 2b/                    # Dataset 2b
│       ├── B0101T.mat
│       ├── B0102T.mat
│       └── ...
├── conformer.py
└── ...
```

---

## 数据格式要求

EEG-Conformer期望的.mat文件结构:

```python
{
    'data': numpy.ndarray,     # Shape: (channels, timepoints, trials)
                               # 例如: (22, 1000, 288) for 2a
    'label': numpy.ndarray     # Shape: (1, trials)
                               # 值: 1,2,3,4 for 2a; 1,2 for 2b
}
```

---

## 下一步操作

**我强烈建议使用MOABB库自动下载**,因为:
1. 自动处理下载和格式转换
2. 数据质量有保证
3. 代码简单,一键完成

请告诉我你想用哪种方法,我可以:
1. 创建自动下载脚本 (使用MOABB)
2. 创建格式转换脚本 (如果你手动下载了.gdf文件)
3. 修改conformer.py的数据路径配置

你想用哪种方式?
