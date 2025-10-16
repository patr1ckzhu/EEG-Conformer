# EEG-Conformer 快速入门指南

## 完整操作流程

### Step 1: 安装依赖包

首先安装所有必需的Python包:

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果你使用GPU (RTX 5080),安装支持CUDA的PyTorch
# 访问 https://pytorch.org/get-started/locally/ 选择合适的版本
# Windows + CUDA 12.1 示例:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装MOABB用于数据下载
pip install moabb
```

---

### Step 2: 下载BCI Competition IV数据集

使用我创建的自动下载脚本:

```bash
# 下载Dataset 2a (推荐先从这个开始)
python download_bci_data.py --dataset 2a --output_dir ./data

# 或者下载Dataset 2b
python download_bci_data.py --dataset 2b --output_dir ./data

# 或者两个都下载
python download_bci_data.py --dataset both --output_dir ./data
```

**预期输出目录结构**:
```
EEG-Conformer/
└── data/
    ├── 2a/
    │   ├── A01T.mat    # Subject 1 训练集
    │   ├── A01E.mat    # Subject 1 测试集
    │   ├── A02T.mat
    │   ├── A02E.mat
    │   └── ... (A03-A09)
    └── 2b/
        ├── B0101T.mat  # Subject 1 Session 1 训练
        ├── B0102T.mat
        └── ...
```

**首次下载时间**: 每个数据集约5-10分钟 (取决于网络速度)

---

### Step 3: 修改数据路径配置

#### 对于Dataset 2a (conformer.py):

打开 `conformer.py`,找到第227行:

```python
# 修改前:
self.root = '/Data/strict_TE/'

# 修改后 (Windows):
self.root = 'C:/Users/YourUsername/Desktop/EEE/Fourth Year/BCI Project/EEG-Conformer/data/2a/'

# 或修改后 (Mac - 你当前的环境):
self.root = '/Users/patrick/Desktop/EEE/Fourth Year/BCI Project/EEG-Conformer/data/2a/'
```

同时修改第229行的log文件路径:

```python
# 修改前:
self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")

# 修改后 (需要先创建results目录):
# 运行: mkdir results
self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")
```

#### 对于Dataset 2b (conformer_BCIIV2b.py):

修改第236行和第240行:

```python
# 第236行:
self.root = './data/2b/'

# 第240行:
self.log_write = open("./results/2b/log_subject%d.txt" % self.nSub, "w")
```

---

### Step 4: 创建必要的目录

```bash
# 创建结果保存目录
mkdir results
mkdir results/2b
```

---

### Step 5: 运行训练

#### 测试Dataset 2a:

```bash
python conformer.py
```

#### 测试Dataset 2b:

```bash
python conformer_BCIIV2b.py
```

---

## 预期训练输出

训练开始后,你会看到类似这样的输出:

```
seed is 1234
Subject 1
Epoch: 0   Train loss: 1.386294   Test loss: 1.386294   Train accuracy 0.250000   Test accuracy is 0.250000
Epoch: 1   Train loss: 1.350123   Test loss: 1.370456   Train accuracy 0.280000   Test accuracy is 0.260000
...
Epoch: 1999   Train loss: 0.050123   Test loss: 0.350456   Train accuracy 0.980000   Test accuracy is 0.786600
The average accuracy is: 0.745123
The best accuracy is: 0.786600
```

---

## 训练参数说明

### 默认超参数 (conformer.py:217-224):

```python
self.batch_size = 72        # 批次大小
self.n_epochs = 2000        # 训练轮数
self.c_dim = 4              # 类别数 (2a是4类)
self.lr = 0.0002            # 学习率
self.dimension = (190, 50)  # 特征维度
```

### 模型参数 (conformer.py:205-211):

```python
Conformer(
    emb_size=40,    # 嵌入维度
    depth=6,        # Transformer层数
    n_classes=4     # 输出类别数
)
```

---

## GPU使用情况

### 检查GPU是否可用:

在Python中运行:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 预期GPU使用率:

- **EEG-Conformer参数量**: 约50K-100K (轻量级模型)
- **预期GPU利用率**: 20-40% (RTX 5080)
- **训练速度**: 每个epoch约1-2秒
- **总训练时间**: 约1小时/每个受试者 (2000 epochs)

---

## 常见问题排查

### 问题1: CUDA out of memory

```bash
# 解决方案: 减小batch size
# 在conformer.py第217行:
self.batch_size = 36  # 从72改为36
```

### 问题2: 找不到数据文件

```bash
# 错误信息: FileNotFoundError: A01T.mat
# 解决方案: 检查数据路径是否正确配置
# 确保data/2a/目录下有所有.mat文件
```

### 问题3: 中文编码错误 (Windows)

```bash
# 如果出现UnicodeEncodeError
# 在PowerShell中运行:
chcp 65001
```

### 问题4: ImportError: No module named 'einops'

```bash
pip install einops
```

---

## 性能基线对比

| 方法 | Dataset 2a | Dataset 2b |
|------|-----------|-----------|
| EEG-Conformer (论文) | 78.66% | 84.63% |
| EEGNet | ~74% | ~80% |
| DeepConvNet | ~75% | ~81% |
| 你的EEGNet | 60.62% (PhysioNet) | N/A |

---

## 下一步优化方向

训练完成后,如果想进一步改进:

1. **调整超参数**:
   - 学习率: 尝试0.0001-0.001
   - Transformer深度: 尝试4-8层
   - 嵌入维度: 尝试20-80

2. **数据增强**:
   - 代码中已包含S&R增强 (conformer.py:246-273)
   - 可以调整分段数量和随机程度

3. **正则化**:
   - Dropout比率 (默认0.5)
   - Weight decay

4. **学习率调度**:
   - 添加ReduceLROnPlateau
   - 余弦退火

---

## 结果文件

训练完成后,会生成:

```
results/
├── log_subject1.txt       # 每个epoch的测试准确率
├── log_subject2.txt
├── ...
├── sub_result.txt         # 所有受试者的汇总结果
└── model.pth              # 最后一个受试者的模型权重
```

---

## 需要帮助?

如果遇到任何问题,请告诉我:
1. 完整的错误信息
2. 你运行的命令
3. Python和CUDA版本信息

我会帮你快速解决!
