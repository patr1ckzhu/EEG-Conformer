"""
BCI Competition IV Dataset 2a/2b 自动下载和预处理脚本 (改进版)

这个版本精确复现了EEG-Conformer作者的MATLAB预处理流程:
1. 时间窗口: cue onset之后2-6秒 (而不是0-4秒)
2. 4-40 Hz Chebyshev Type II 带通滤波
3. 数据格式: (timepoints=1000, channels=22, trials=288)

使用方法:
    python download_bci_data_improved.py --dataset 2a --output_dir ./data
"""

import os
import argparse
import numpy as np
import scipy.io
import scipy.signal
from pathlib import Path
import mne

def apply_bandpass_filter(data, fs=250, lowcut=4, highcut=40):
    """
    应用Chebyshev Type II带通滤波器

    复现作者的MATLAB代码:
    fc = 250; % sampling rate
    Wl = 4; Wh = 40; % pass band
    Wn = [Wl*2 Wh*2]/fc;
    [b,a]=cheby2(6,60,Wn);

    参数:
        data: (n_trials, n_channels, n_times) 或 (n_channels, n_times)
        fs: 采样率 (默认250Hz)
        lowcut: 低频截止 (默认4Hz)
        highcut: 高频截止 (默认40Hz)

    返回:
        filtered_data: 滤波后的数据
    """
    # 归一化频率
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq

    # Chebyshev Type II 滤波器参数
    # order=6, stop-band attenuation=60dB
    b, a = scipy.signal.cheby2(6, 60, [low, high], btype='band')

    # 应用零相位滤波 (filtfilt)
    if data.ndim == 3:
        # (n_trials, n_channels, n_times)
        filtered_data = np.zeros_like(data)
        for trial_idx in range(data.shape[0]):
            filtered_data[trial_idx, :, :] = scipy.signal.filtfilt(b, a, data[trial_idx, :, :], axis=1)
    elif data.ndim == 2:
        # (n_channels, n_times)
        filtered_data = scipy.signal.filtfilt(b, a, data, axis=1)
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")

    return filtered_data


def download_dataset_2a(output_dir, use_improved_preprocessing=True):
    """
    下载BCI Competition IV Dataset 2a

    参数:
        output_dir: 输出目录
        use_improved_preprocessing: 是否使用改进的预处理(匹配作者的MATLAB代码)
    """
    print("=" * 80)
    print("开始下载 BCI Competition IV Dataset 2a")
    if use_improved_preprocessing:
        print("使用改进的预处理流程 (匹配作者的MATLAB代码)")
    print("=" * 80)

    try:
        from moabb.datasets import BNCI2014_001
    except ImportError:
        print("[ERROR] 未安装MOABB库!")
        print("请运行: pip install moabb")
        return False

    # 创建输出目录
    output_path = Path(output_dir) / "2a"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录: {output_path}")

    # 初始化数据集
    print("\n[Step 1/4] 初始化MOABB数据集对象...")
    dataset = BNCI2014_001()
    print(f"[OK] 数据集: {dataset.code}")
    print(f"[OK] 受试者数量: {len(dataset.subject_list)}")
    print(f"[OK] 任务: 4类运动想象 (左手、右手、双脚、舌头)")

    # 下载数据
    print("\n[Step 2/4] 下载数据 (首次下载可能需要几分钟)...")

    # 遍历所有受试者
    for subject_id in dataset.subject_list:
        print(f"\n处理 Subject {subject_id}...")

        try:
            # 获取原始数据
            sessions = dataset.get_data(subjects=[subject_id])
            session_list = list(sessions[subject_id].items())
            print(f"  发现 {len(session_list)} 个session")

            for session_idx, (session_name, session_data) in enumerate(session_list):
                print(f"  处理 session: '{session_name}'")

                # 提取run数据
                all_data = []
                all_labels = []

                for run_name, run_data in session_data.items():
                    raw = run_data

                    # 获取事件
                    events, event_id = mne.events_from_annotations(raw)

                    # 提取EEG数据 (22通道)
                    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

                    if use_improved_preprocessing:
                        # 改进版: 匹配作者的MATLAB代码
                        # 时间窗口: cue onset之后2-6秒 (总共4秒=1000个采样点@250Hz)
                        # MATLAB: s((Pos(j)+500):(Pos(j)+1499),1:22)
                        # 500/250Hz = 2秒, 1000个点 = 4秒
                        epochs = mne.Epochs(raw, events, event_id=event_id,
                                           tmin=2.0, tmax=6.0, picks=picks,
                                           baseline=None, preload=True)
                    else:
                        # 原始版本: 0-4秒
                        epochs = mne.Epochs(raw, events, event_id=event_id,
                                           tmin=0., tmax=4., picks=picks,
                                           baseline=None, preload=True)

                    # 提取数据
                    epoch_data = epochs.get_data()  # (n_trials, n_channels, n_times)
                    epoch_labels = epochs.events[:, -1]  # 事件标签

                    all_data.append(epoch_data)
                    all_labels.append(epoch_labels)

                # 合并所有runs
                all_data = np.concatenate(all_data, axis=0)  # (n_trials, n_channels, n_times)
                all_labels = np.concatenate(all_labels, axis=0)  # (n_trials,)

                print(f"    合并后数据shape: {all_data.shape}")

                # 截取前1000个时间点 (确保精确1000个点)
                if all_data.shape[2] > 1000:
                    all_data = all_data[:, :, :1000]
                    print(f"    截取到1000个时间点: {all_data.shape}")
                elif all_data.shape[2] < 1000:
                    print(f"    [WARNING] 时间点不足1000! 实际: {all_data.shape[2]}")

                # [Step 3/4] 应用4-40Hz带通滤波
                if use_improved_preprocessing:
                    print(f"\n[Step 3/4] 应用4-40Hz Chebyshev Type II带通滤波...")
                    all_data = apply_bandpass_filter(all_data, fs=250, lowcut=4, highcut=40)
                    print(f"    滤波完成")

                # 转换为EEG-Conformer期望的格式
                # 从 (trials, channels, times) 转为 (times, channels, trials)
                data_transposed = np.transpose(all_data, (2, 1, 0))

                # 标签reshape为 (trials, 1)
                labels = all_labels.reshape(-1, 1)

                # 判断是训练集还是测试集
                if 'T' in session_name or '0' in session_name or session_idx == 0:
                    session_suffix = 'T'
                else:
                    session_suffix = 'E'

                filename = f"A{subject_id:02d}{session_suffix}.mat"
                filepath = output_path / filename

                # [Step 4/4] 保存为.mat文件
                scipy.io.savemat(
                    filepath,
                    {
                        'data': data_transposed,
                        'label': labels
                    }
                )

                print(f"\n  [SAVED] {filename}")
                print(f"    - Session name: '{session_name}'")
                print(f"    - Data shape: {data_transposed.shape}")
                print(f"    - Label shape: {labels.shape}")
                print(f"    - Label range: {labels.min()} - {labels.max()}")
                print(f"    - Label unique: {np.unique(labels)}")

        except Exception as e:
            print(f"  [ERROR] Subject {subject_id} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("[SUCCESS] Dataset 2a 下载完成!")
    print(f"数据保存在: {output_path}")
    print("=" * 80)
    return True


def download_dataset_2b(output_dir, use_improved_preprocessing=True):
    """
    下载BCI Competition IV Dataset 2b
    """
    print("=" * 80)
    print("开始下载 BCI Competition IV Dataset 2b")
    if use_improved_preprocessing:
        print("使用改进的预处理流程")
    print("=" * 80)

    try:
        from moabb.datasets import BNCI2014_004
    except ImportError:
        print("[ERROR] 未安装MOABB库!")
        print("请运行: pip install moabb")
        return False

    # 创建输出目录
    output_path = Path(output_dir) / "2b"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录: {output_path}")

    # 初始化数据集
    print("\n[Step 1/4] 初始化MOABB数据集对象...")
    dataset = BNCI2014_004()
    print(f"[OK] 数据集: {dataset.code}")
    print(f"[OK] 受试者数量: {len(dataset.subject_list)}")
    print(f"[OK] 任务: 2类运动想象 (左手 vs 右手)")

    print("\n[Step 2/4] 下载数据...")

    # 遍历所有受试者
    for subject_id in dataset.subject_list:
        print(f"\n处理 Subject {subject_id}...")

        try:
            sessions = dataset.get_data(subjects=[subject_id])

            # BNCI2014_004有5个session
            for session_idx, (session_name, session_data) in enumerate(sessions[subject_id].items(), 1):
                all_data = []
                all_labels = []

                for run_name, run_data in session_data.items():
                    raw = run_data
                    events, event_id = mne.events_from_annotations(raw)
                    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

                    if use_improved_preprocessing:
                        epochs = mne.Epochs(raw, events, event_id=event_id,
                                           tmin=2.0, tmax=6.0, picks=picks,
                                           baseline=None, preload=True)
                    else:
                        epochs = mne.Epochs(raw, events, event_id=event_id,
                                           tmin=0., tmax=4., picks=picks,
                                           baseline=None, preload=True)

                    epoch_data = epochs.get_data()
                    epoch_labels = epochs.events[:, -1]

                    all_data.append(epoch_data)
                    all_labels.append(epoch_labels)

                all_data = np.concatenate(all_data, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # 截取到1000个时间点
                if all_data.shape[2] > 1000:
                    all_data = all_data[:, :, :1000]

                # 应用滤波
                if use_improved_preprocessing:
                    print(f"  [Step 3/4] 应用4-40Hz滤波...")
                    all_data = apply_bandpass_filter(all_data, fs=250, lowcut=4, highcut=40)

                # 转换格式
                data_transposed = np.transpose(all_data, (2, 1, 0))
                labels = all_labels.reshape(-1, 1)

                # Dataset 2b文件命名
                session_type = 'T' if session_idx <= 3 else 'E'
                filename = f"B{subject_id:02d}{session_idx:02d}{session_type}.mat"
                filepath = output_path / filename

                scipy.io.savemat(
                    filepath,
                    {
                        'data': data_transposed,
                        'label': labels
                    }
                )

                print(f"  [SAVED] {filename}")
                print(f"    - Data shape: {data_transposed.shape}")
                print(f"    - Label shape: {labels.shape}")

        except Exception as e:
            print(f"  [ERROR] Subject {subject_id} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("[SUCCESS] Dataset 2b 下载完成!")
    print(f"数据保存在: {output_path}")
    print("=" * 80)
    return True


def check_dependencies():
    """检查依赖包"""
    print("检查依赖包...")

    missing = []

    packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'mne': 'mne',
        'moabb': 'moabb'
    }

    for display_name, package_name in packages.items():
        try:
            __import__(package_name)
            print(f"[OK] {display_name}")
        except ImportError:
            missing.append(package_name)
            print(f"[X] {display_name}")

    if missing:
        print(f"\n[ERROR] 缺少依赖包: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="下载BCI Competition IV数据集 (改进版,匹配作者的MATLAB预处理)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['2a', '2b', 'both'],
        default='2a',
        help="选择要下载的数据集 (默认: 2a)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help="数据输出目录 (默认: ./data)"
    )
    parser.add_argument(
        '--no-improved-preprocessing',
        action='store_true',
        help="不使用改进的预处理(使用0-4秒窗口,不滤波)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("BCI Competition IV 数据集下载器 (改进版)")
    print("=" * 80)

    if not args.no_improved_preprocessing:
        print("\n改进的预处理流程:")
        print("  ✓ 时间窗口: cue onset之后 2-6秒 (匹配作者的MATLAB代码)")
        print("  ✓ 4-40 Hz Chebyshev Type II 带通滤波")
        print("  ✓ 数据格式: (1000, 22, 288) = (timepoints, channels, trials)")
    else:
        print("\n基础预处理流程:")
        print("  - 时间窗口: cue onset之后 0-4秒")
        print("  - 无显式滤波")

    print()

    # 检查依赖
    if not check_dependencies():
        return

    print()

    use_improved = not args.no_improved_preprocessing

    # 下载数据集
    if args.dataset in ['2a', 'both']:
        success = download_dataset_2a(args.output_dir, use_improved_preprocessing=use_improved)
        if not success:
            return

    if args.dataset in ['2b', 'both']:
        success = download_dataset_2b(args.output_dir, use_improved_preprocessing=use_improved)
        if not success:
            return

    print("\n" + "=" * 80)
    print("所有数据集下载完成!")
    print("=" * 80)
    print("\n下一步:")
    print("1. 检查数据文件: python check_data_format.py")
    print("2. 运行训练: python conformer.py")
    print("\n注意:")
    print("  改进版使用了2-6秒的时间窗口和4-40Hz滤波")
    print("  这更接近作者的MATLAB实现,可能会提高模型性能")


if __name__ == "__main__":
    main()
