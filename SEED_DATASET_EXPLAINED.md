# SEED 数据集详解

## 什么是 SEED 数据集?

**SEED (SJTU Emotion EEG Dataset)** 是上海交通大学发布的情绪识别EEG数据集。

---

## SEED vs BCI Competition IV 对比

| 项目 | SEED | BCI Competition IV 2a |
|------|------|----------------------|
| **任务类型** | 情绪识别 | 运动想象 |
| **数据来源** | 上海交通大学 | 德国BCI竞赛 |
| **受试者数** | 15人 | 9人 |
| **EEG通道** | 62通道 | 22通道 |
| **采样率** | 200 Hz | 250 Hz |
| **类别数** | 3类 (积极、中性、消极情绪) | 4类 (左手、右手、脚、舌头) |
| **会话数** | 3 sessions/人 | 1 session/人 |
| **试次数** | 15 trials/session | 288 trials (训练+测试) |
| **论文准确率** | 95.30% (5-fold CV) | 78.66% (hold-out) |

---

## SEED 数据集详细结构

### 实验设计:

**情绪诱导**:
- 受试者观看15个电影片段
- 每个片段诱导一种特定情绪:
  - **积极情绪** (Positive): 5个片段
  - **中性情绪** (Neutral): 5个片段
  - **消极情绪** (Negative): 5个片段

**数据采集**:
- 每个受试者参与3个session (不同时间采集)
- 每个session观看15个电影片段
- 记录观看过程中的EEG信号

### 原始数据格式:

```
SEED/
├── S1_1_1.mat    # Subject 1, Session 1, Trial 1
├── S1_1_2.mat    # Subject 1, Session 1, Trial 2
├── ...
├── S1_1_15.mat   # Subject 1, Session 1, Trial 15
├── S1_2_1.mat    # Subject 1, Session 2, Trial 1
├── ...
├── S15_3_15.mat  # Subject 15, Session 3, Trial 15
```

**每个.mat文件内容**:
```matlab
trial_data: (62 channels × time_samples)
trial_label: 1个标签 (-1, 0, 1 代表消极、中性、积极)
```

**时间长度**:
- 每个试次时长不固定 (取决于电影片段长度)
- 大约几十秒到几分钟

---

## preprocessing/ 文件夹的作用

这个文件夹包含数据预处理脚本,用于将原始SEED数据转换为模型可用的格式。

### 1. seed_process_slice.py - 数据切片和训练/测试集划分

**作用**: 将长时间EEG信号切成固定长度片段

**处理流程** (seed_process_slice.py:10-62):

```python
for 每个受试者 (15人):
    for 每个session (3个):

        # 训练集: 前9个试次
        for trial in [1-9]:
            读取 S{sub}_{session}_{trial}.mat

            # 切片: 将长EEG信号切成1秒片段
            trial_number = 信号长度 / 200  # 200采样点 = 1秒
            for 每个1秒片段:
                one_trial.append(数据[0:200])
                one_trial.append(数据[200:400])
                ...

        保存: S{sub}_session{session}T.npy (训练数据)

        # 测试集: 后6个试次
        for trial in [10-15]:
            同样切片处理

        保存: S{sub}_session{session}E.npy (测试数据)
```

**输出格式**:
```
data_1second/
├── S1_session1T.npy         # Subject 1 Session 1 训练数据
├── S1_session1T_label.npy   # 对应标签
├── S1_session1E.npy         # Subject 1 Session 1 测试数据
├── S1_session1E_label.npy
├── ...
```

**数据形状**:
- 训练数据: (n_samples_train, 62, 200)
- 测试数据: (n_samples_test, 62, 200)
- n_samples = 所有试次切片后的总数

---

### 2. seed_process_slice_cv.py - 5折交叉验证数据准备

**作用**: 为5折交叉验证准备数据

**与seed_process_slice.py的区别**:
- seed_process_slice.py: 固定训练/测试集划分 (9 trials训练, 6 trials测试)
- **seed_process_slice_cv.py**: 将所有15个试次合并,不预先划分训练/测试集

**处理流程** (类似,但不分训练/测试):

```python
for 每个受试者:
    for 每个session:
        all_trials = []

        # 处理所有15个试次
        for trial in [1-15]:
            切片为1秒片段
            all_trials.append(片段)

        保存: S{sub}_session{session}.npy  # 所有数据
```

**5折交叉验证在训练时动态划分** (conformer_seed_1s_5fold.py:297-338):

```python
def get_source_data(self, fold):
    # 加载所有数据
    all_data = np.load('S%d_session1.npy' % self.nSub)
    all_label = np.load('S%d_session1_label.npy' % self.nSub)

    # 5折划分
    one_fold_num = len(all_data) // 5
    test_idx = [fold*one_fold_num : (fold+1)*one_fold_num]
    train_idx = 其余数据

    train_data = all_data[train_idx]
    test_data = all_data[test_idx]
```

---

## conformer_seed_1s_5fold.py - SEED训练脚本

### 关键配置 (conformer_seed_1s_5fold.py:233-247):

```python
batch_size = 200
n_epochs = 600
n_classes = 3        # 3类情绪
n_subjects = 15      # 15个受试者
n_folds = 5          # 5折交叉验证
```

### 输入数据维度:

```python
# PatchEmbedding (line 82)
nn.Conv2d(1, 40, (62, 1))  # 62通道 (vs BCI 2a的22通道)

# 时间切片
200个采样点 = 1秒 EEG信号 (vs BCI 2a的1000点)
```

### 训练流程 (conformer_seed_1s_5fold.py:480-531):

```python
def main():
    for subject in range(15):  # 15个受试者

        for fold in range(5):  # 5折交叉验证
            exgan = ExGAN(subject+1, fold)
            bestAcc, averAcc = exgan.train(fold)

        # 计算该受试者5折平均准确率
        avg_5fold = sum(bestAcc) / 5

    # 计算15个受试者的总平均准确率
    final_accuracy = sum(avg_all_subjects) / 15
    # 论文报告: 95.30%
```

---

## 数据量对比

### SEED 数据量计算:

每个受试者单个session:
- 15个试次 (电影片段)
- 每个试次约60秒 (假设平均)
- 切片为1秒片段: 15试次 × 60秒 = **900个样本/session**

5折交叉验证 (使用session 1):
- 训练集: 900 × 4/5 = 720个样本
- 测试集: 900 × 1/5 = 180个样本

**远多于BCI Competition IV 2a的288个样本!**

---

## 为什么SEED准确率更高 (95.30%)?

1. **任务相对简单**: 情绪识别的EEG特征比运动想象更明显
2. **数据量更大**: 每个受试者约900个样本 vs BCI 2a的288个
3. **信号更稳定**: 观看视频时的EEG信号比想象运动更稳定
4. **更多通道**: 62通道 vs 22通道,捕获更多空间信息

---

## 如何使用 SEED 数据集?

### 方法1: 官方下载

访问: http://bcmi.sjtu.edu.cn/~seed/

**注意**: 需要注册账号并申请下载权限

### 方法2: 使用已有的预处理脚本

如果你已经有原始SEED数据:

```bash
# 1. 修改路径 (seed_process_slice_cv.py:7-8)
root_path = '/your/path/to/SEED/raw/'
save_path = '/your/path/to/SEED/processed/'

# 2. 运行预处理
python preprocessing/seed_process_slice_cv.py

# 3. 训练模型
python conformer_seed_1s_5fold.py
```

---

## preprocessing 文件夹总结

| 文件 | 作用 | 输出 | 用于哪个脚本 |
|------|------|------|------------|
| **seed_process_slice.py** | 固定训练/测试集划分 (9/6) | S{sub}_session{sess}T.npy 和 E.npy | 可能有其他脚本使用 |
| **seed_process_slice_cv.py** | 合并所有试次,准备5折CV | S{sub}_session{sess}.npy (不分T/E) | conformer_seed_1s_5fold.py |

---

## 你是否需要使用 SEED?

### 建议:

**先从 BCI Competition IV 2a 开始!**

原因:
1. **更容易下载**: 公开数据集,不需要申请
2. **更经典**: BCI领域标准评测数据集
3. **与你的目标更接近**: 运动想象 vs 情绪识别
4. **文档更完善**: 社区支持更好

**SEED可以作为后续探索**:
- 验证EEG-Conformer在不同任务上的泛化能力
- 对比情绪识别 vs 运动想象的模型表现

---

## moabb/mne 安装问题

请把你的具体错误信息给我,我可以帮你解决。常见问题:

1. **Python版本不兼容**: mne需要Python 3.8-3.11
2. **依赖冲突**: numpy/scipy版本问题
3. **网络问题**: PyPI下载失败

**临时解决方案**:
如果安装失败,你可以先手动从官网下载BCI Competition IV 2a数据,我帮你写一个简单的数据加载脚本,不依赖moabb。

请把错误信息发给我!
