"""
BCI Competition IV Dataset 2a/2b 自动下载和预处理脚本

使用MOABB库自动下载BCI Competition IV数据集并转换为.mat格式

使用方法:
    python download_bci_data.py --dataset 2a --output_dir ./data
"""

import os
import argparse
import numpy as np
import scipy.io
from pathlib import Path

def download_dataset_2a(output_dir):
    """
    下载BCI Competition IV Dataset 2a

    使用MOABB库自动下载和预处理
    """
    print("=" * 80)
    print("开始下载 BCI Competition IV Dataset 2a")
    print("=" * 80)

    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        print("[ERROR] 未安装MOABB库!")
        print("请运行: pip install moabb")
        return False

    # 创建输出目录
    output_path = Path(output_dir) / "2a"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录: {output_path}")

    # 初始化数据集
    print("\n[Step 1/3] 初始化MOABB数据集对象...")
    dataset = BNCI2014_001()
    print(f"[OK] 数据集: {dataset.code}")
    print(f"[OK] 受试者数量: {len(dataset.subject_list)}")
    print(f"[OK] 任务: 4类运动想象 (左手、右手、双脚、舌头)")

    # 下载数据
    print("\n[Step 2/3] 下载数据 (首次下载可能需要几分钟)...")

    # 遍历所有受试者
    for subject_id in dataset.subject_list:
        print(f"\n处理 Subject {subject_id}...")

        try:
            # 获取原始数据
            sessions = dataset.get_data(subjects=[subject_id])

            # BNCI2014_001包含两个session: session_T (训练) 和 session_E (测试)
            for session_name, session_data in sessions[subject_id].items():
                # 提取run数据
                all_data = []
                all_labels = []

                for run_name, run_data in session_data.items():
                    # run_data是mne.io.Raw对象
                    raw = run_data

                    # 获取事件
                    events, event_id = mne.events_from_annotations(raw)

                    # 提取EEG数据 (22通道)
                    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

                    # 分epoch: -0.5到4秒,相对于cue onset
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

                # 转换为EEG-Conformer期望的格式
                # 从 (trials, channels, times) 转为 (channels, times, trials)
                data_transposed = np.transpose(all_data, (1, 2, 0))

                # 标签reshape
                labels = all_labels.reshape(1, -1)

                # 保存为.mat文件
                session_suffix = 'T' if 'T' in session_name else 'E'
                filename = f"A{subject_id:02d}{session_suffix}.mat"
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
                print(f"    - Label range: {labels.min()} - {labels.max()}")

        except Exception as e:
            print(f"  [ERROR] Subject {subject_id} 处理失败: {e}")
            continue

    print("\n" + "=" * 80)
    print("[SUCCESS] Dataset 2a 下载完成!")
    print(f"数据保存在: {output_path}")
    print("=" * 80)
    return True


def download_dataset_2b(output_dir):
    """
    下载BCI Competition IV Dataset 2b
    """
    print("=" * 80)
    print("开始下载 BCI Competition IV Dataset 2b")
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
    print("\n[Step 1/3] 初始化MOABB数据集对象...")
    dataset = BNCI2014_004()
    print(f"[OK] 数据集: {dataset.code}")
    print(f"[OK] 受试者数量: {len(dataset.subject_list)}")
    print(f"[OK] 任务: 2类运动想象 (左手 vs 右手)")

    print("\n[Step 2/3] 下载数据...")

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

                    epochs = mne.Epochs(raw, events, event_id=event_id,
                                       tmin=0., tmax=4., picks=picks,
                                       baseline=None, preload=True)

                    epoch_data = epochs.get_data()
                    epoch_labels = epochs.events[:, -1]

                    all_data.append(epoch_data)
                    all_labels.append(epoch_labels)

                all_data = np.concatenate(all_data, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                data_transposed = np.transpose(all_data, (1, 2, 0))
                labels = all_labels.reshape(1, -1)

                # Dataset 2b文件命名: B0101T.mat (Subject 01, Session 01, Training/Evaluation)
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

        except Exception as e:
            print(f"  [ERROR] Subject {subject_id} 处理失败: {e}")
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

    try:
        import numpy
        print("[OK] numpy")
    except ImportError:
        missing.append("numpy")

    try:
        import scipy
        print("[OK] scipy")
    except ImportError:
        missing.append("scipy")

    try:
        import mne
        print("[OK] mne")
    except ImportError:
        missing.append("mne")

    try:
        import moabb
        print("[OK] moabb")
    except ImportError:
        missing.append("moabb")

    if missing:
        print(f"\n[ERROR] 缺少依赖包: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="下载BCI Competition IV数据集"
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

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("BCI Competition IV 数据集下载器")
    print("=" * 80 + "\n")

    # 检查依赖
    if not check_dependencies():
        return

    print()

    # 下载数据集
    if args.dataset in ['2a', 'both']:
        success = download_dataset_2a(args.output_dir)
        if not success:
            return

    if args.dataset in ['2b', 'both']:
        success = download_dataset_2b(args.output_dir)
        if not success:
            return

    print("\n" + "=" * 80)
    print("所有数据集下载完成!")
    print("=" * 80)
    print("\n下一步:")
    print("1. 检查数据文件是否正确生成")
    print("2. 修改conformer.py中的数据路径")
    print("3. 运行训练: python conformer.py")


if __name__ == "__main__":
    main()
