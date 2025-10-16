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
    print("  data shape: (channels, timepoints, trials)")
    print("  例如: (22, 1000, 288)")
    print()
    print("  label shape: (1, trials)")
    print("  例如: (1, 288)")
    print("  label values: 1, 2, 3, 4")

    print("\n" + "=" * 80)
    print("问题诊断:")
    print("=" * 80)

    if 'data' in data and 'label' in data:
        data_shape = data['data'].shape
        label_shape = data['label'].shape

        # 检查数据维度
        if len(data_shape) == 3:
            print(f"✓ data 有3个维度: {data_shape}")
        else:
            print(f"✗ data 维度不对: {data_shape}")

        # 检查标签维度
        if label_shape[0] == 1:
            print(f"✓ label 第一维是1: {label_shape}")
        else:
            print(f"✗ label 第一维不是1: {label_shape}")
            print(f"  需要: (1, n_trials), 实际: {label_shape}")

        # 检查数量是否匹配
        n_trials_data = data_shape[2] if len(data_shape) == 3 else data_shape[0]
        n_trials_label = label_shape[1] if len(label_shape) == 2 else label_shape[0]

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