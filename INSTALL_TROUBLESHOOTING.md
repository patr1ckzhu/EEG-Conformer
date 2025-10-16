# moabb/mne 安装问题排查指南

## 常见安装问题

### 问题1: Python版本不兼容

**错误信息**:
```
ERROR: Package 'mne' requires a different Python: 3.13.5 not in '>=3.8,<3.12'
```

**原因**: 你的Python 3.13.5太新了,mne目前只支持Python 3.8-3.11

**解决方案A - 创建虚拟环境 (推荐)**:

```bash
# 使用conda创建Python 3.11环境
conda create -n eeg-conformer python=3.11
conda activate eeg-conformer

# 重新安装依赖
pip install -r requirements.txt
pip install moabb mne
```

**解决方案B - 使用pyenv**:

```bash
# 安装Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9

# 重新安装依赖
pip install -r requirements.txt
pip install moabb mne
```

---

### 问题2: 依赖冲突

**错误信息**:
```
ERROR: Cannot install mne because these package versions have conflicting dependencies:
  numpy>=2.0 vs numpy<2.0
```

**解决方案**:

```bash
# 先安装兼容版本的numpy
pip install "numpy<2.0"

# 再安装mne和moabb
pip install mne
pip install moabb
```

---

### 问题3: 网络问题/下载超时

**错误信息**:
```
ReadTimeoutError: HTTPSConnectionPool...
```

**解决方案 - 使用国内镜像源**:

```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mne moabb

# 或使用阿里镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ mne moabb
```

---

### 问题4: 编译错误 (Windows常见)

**错误信息**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解决方案**:

1. 安装 Microsoft C++ Build Tools
   - 下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - 安装时选择 "Desktop development with C++"

2. 或者使用预编译wheel:
```bash
pip install --only-binary :all: mne moabb
```

---

## 不依赖moabb的替代方案

### 方案1: 手动下载 + 简单加载脚本

如果moabb安装失败,你可以:

1. **手动下载BCI Competition IV 2a数据**
   - 访问: https://www.bbci.de/competition/iv/
   - 下载.gdf文件

2. **使用MNE直接读取.gdf文件**
   ```python
   import mne
   raw = mne.io.read_raw_gdf('A01T.gdf', preload=True)
   ```

3. **我可以帮你写一个简单的转换脚本**,不依赖moabb

---

### 方案2: 使用预处理好的数据

论文作者可能提供了预处理好的.mat数据。

如果你能找到已经转换好的.mat文件:
- A01T.mat - A09T.mat (训练集)
- A01E.mat - A09E.mat (测试集)

可以直接跳过下载步骤,直接训练!

---

## 最简安装流程 (从头开始)

### Step 1: 检查Python版本

```bash
python --version
# 如果是3.13.x,建议降级到3.11.x
```

### Step 2: 创建干净的虚拟环境

```bash
# Mac/Linux (使用venv)
python3.11 -m venv eeg_env
source eeg_env/bin/activate

# Windows
python -m venv eeg_env
eeg_env\Scripts\activate
```

### Step 3: 升级pip

```bash
pip install --upgrade pip
```

### Step 4: 按顺序安装依赖

```bash
# 1. 先安装基础科学计算库
pip install "numpy<2.0" scipy matplotlib

# 2. 安装PyTorch (根据你的CUDA版本)
# CPU版本:
pip install torch torchvision

# GPU版本 (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他深度学习依赖
pip install einops torchsummary

# 4. 安装机器学习库
pip install scikit-learn

# 5. 最后安装EEG处理库
pip install mne
pip install moabb
```

### Step 5: 验证安装

```python
python -c "import mne; print(mne.__version__)"
python -c "import moabb; print(moabb.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 我需要你的错误信息

请把以下命令的输出发给我:

```bash
# 1. Python版本
python --version

# 2. 尝试安装mne的完整输出
pip install mne 2>&1 | tee install_error.txt

# 3. 尝试安装moabb的完整输出
pip install moabb 2>&1 | tee install_error_moabb.txt
```

然后把 `install_error.txt` 的内容发给我,我会提供针对性的解决方案!

---

## 临时绕过方案

如果你急于开始训练,可以先不安装moabb/mne:

### 选项1: 从其他来源获取数据

- 联系同学/老师是否有已处理好的.mat文件
- 在Kaggle或其他平台搜索预处理好的BCI Competition IV数据

### 选项2: 先用论文提供的其他数据集

- 如果论文作者提供了示例数据,先用那个测试代码
- 验证模型架构是否正确

### 选项3: 我帮你写一个不依赖moabb的下载脚本

如果你能访问数据集官网,我可以写一个用requests/urllib下载的脚本,绕过moabb。

---

请把你的错误信息发给我!
