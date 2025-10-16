"""
检查.mat文件的数据格式

运行这个脚本查看数据的实际结构
"""

import scipy.io
import numpy as np

# 读取第一个文件
mat_file = './data/2a/A01T.mat'

print("=" * 80)
print(f"检查文件: {mat_file}")
print("=" * 80)

try:
    data = scipy.io.loadmat(mat_file)

    print("\n文件中的所有keys:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  - {key}")

    print("\n'data' 字段:")
    if 'data' in data:
        print(f"  Shape: {data['data'].shape}")
        print(f"  Dtype: {data['data'].dtype}")

    print("\n'label' 字段:")
    if 'label' in data:
        print(f"  Shape: {data['label'].shape}")
        print(f"  Dtype: {data['label'].dtype}")
        print(f"  Unique values: {np.unique(data['label'])}")
        print(f"  Min: {data['label'].min()}, Max: {data['label'].max()}")

    print("\n" + "=" * 80)
    print("conformer.py 期望的格式:")
    print("=" * 80)
    print("  data shape: (timepoints, channels, trials)")
    print("  例如: (1000, 22, 288)")
    print()
    print("  label shape: (trials, 1)")
    print("  例如: (288, 1)")
    print("  label values: 1, 2, 3, 4")
    print()
    print("  为什么是 (timepoints, channels, trials)?")
    print("  因为conformer.py会执行: transpose((2,1,0)) -> expand_dims(axis=1)")
    print("  (1000, 22, 288) -> transpose -> (288, 22, 1000) -> expand -> (288, 1, 22, 1000) ✓")

    print("\n" + "=" * 80)
    print("问题诊断:")
    print("=" * 80)

    if 'data' in data and 'label' in data:
        data_shape = data['data'].shape
        label_shape = data['label'].shape

        # 检查数据维度
        if len(data_shape) == 3:
            print(f"✓ data 有3个维度: {data_shape}")
            # 验证是 (times, channels, trials) 格式
            if data_shape[0] > 100 and data_shape[1] < 30:  # times > 100, channels < 30
                print(f"✓ data 看起来是 (times, channels, trials) 格式")
            else:
                print(f"⚠ data 可能不是 (times, channels, trials) 格式")
                print(f"  应该是: (1000, 22, 288)")
        else:
            print(f"✗ data 维度不对: {data_shape}")

        # 检查标签维度
        if len(label_shape) == 2 and label_shape[1] == 1:
            print(f"✓ label 格式正确: {label_shape} (trials, 1)")
        else:
            print(f"✗ label 格式不对: {label_shape}")
            print(f"  需要: (n_trials, 1), 实际: {label_shape}")

        # 检查数量是否匹配
        n_trials_data = data_shape[2] if len(data_shape) == 3 else data_shape[0]
        n_trials_label = label_shape[0] if len(label_shape) == 2 and label_shape[1] == 1 else label_shape[0]

        if n_trials_data == n_trials_label:
            print(f"✓ 数据和标签的trial数量匹配: {n_trials_data}")
        else:
            print(f"✗ 数量不匹配!")
            print(f"  data trials: {n_trials_data}")
            print(f"  label trials: {n_trials_label}")

except FileNotFoundError:
    print(f"\n[ERROR] 文件不存在: {mat_file}")
    print("请确保数据已经下载到 ./data/2a/ 目录")
except Exception as e:
    print(f"\n[ERROR] 读取文件失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)