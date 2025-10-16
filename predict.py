"""
使用训练好的模型进行预测

功能:
1. 加载保存的模型
2. 对新的EEG数据进行预测
3. 支持单个trial或批量预测

使用方法:
    # 预测单个trial
    python predict.py --subject 1 --data_file test_trial.npy

    # 预测整个测试集
    python predict.py --subject 1 --test_set
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io
import json

from save_best_models import Conformer  # 导入模型定义


class EEGPredictor:
    def __init__(self, subject_id, model_dir='./models', device='cuda'):
        """
        初始化预测器

        参数:
            subject_id: Subject ID (1-9)
            model_dir: 模型保存目录
            device: 'cuda' 或 'cpu'
        """
        self.subject_id = subject_id
        self.model_dir = model_dir
        self.device = device

        # 加载模型
        self.model = self._load_model()

        # 加载统计信息
        self.stats = self._load_stats()

        # 类别映射
        self.class_names = {
            0: 'Left Hand',
            1: 'Right Hand',
            2: 'Feet',
            3: 'Tongue'
        }

    def _load_model(self):
        """加载训练好的模型"""
        model_path = f"{self.model_dir}/subject_{self.subject_id}_best.pth"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from: {model_path}")

        # 创建模型
        model = Conformer()

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 设置为评估模式
        model.eval()
        model = model.to(self.device)

        print(f"✓ Model loaded successfully")
        print(f"  Best accuracy: {checkpoint['best_acc']:.4f}")
        print(f"  Trained at epoch: {checkpoint['epoch']}")

        return model

    def _load_stats(self):
        """加载训练统计信息"""
        stats_path = f"{self.model_dir}/subject_{self.subject_id}_stats.json"

        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"✓ Stats loaded: Best acc = {stats['best_accuracy']:.4f}")
            return stats
        else:
            print("⚠ Stats file not found, using default normalization")
            return {'normalization_params': {'mean': 0.0, 'std': 1.0}}

    def preprocess(self, data):
        """
        预处理EEG数据

        参数:
            data: numpy array
                  - 单个trial: (22, 1000) 或 (1, 22, 1000)
                  - 多个trials: (n_trials, 22, 1000) 或 (n_trials, 1, 22, 1000)

        返回:
            torch.Tensor: (batch_size, 1, 22, 1000)
        """
        # 确保是 numpy array
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # 调整维度
        if data.ndim == 2:  # (22, 1000)
            data = np.expand_dims(data, axis=0)  # (1, 22, 1000)
            data = np.expand_dims(data, axis=1)  # (1, 1, 22, 1000)
        elif data.ndim == 3:
            if data.shape[1] == 22:  # (n_trials, 22, 1000)
                data = np.expand_dims(data, axis=1)  # (n_trials, 1, 22, 1000)
            # 否则假设已经是 (n_trials, 1, 22, 1000)

        # 标准化 (使用训练时的参数)
        norm_params = self.stats.get('normalization_params', {'mean': 0.0, 'std': 1.0})
        data = (data - norm_params['mean']) / norm_params['std']

        # 转换为 tensor
        data_tensor = torch.from_numpy(data).float().to(self.device)

        return data_tensor

    def predict(self, data, return_probabilities=False):
        """
        预测EEG数据的类别

        参数:
            data: numpy array or torch.Tensor
                  EEG数据 (见 preprocess 的说明)
            return_probabilities: 是否返回概率

        返回:
            predictions: numpy array of int
                         预测的类别 (0-3)
            probabilities (optional): numpy array of float
                                     每个类别的概率
        """
        # 预处理
        data_tensor = self.preprocess(data)

        # 预测
        with torch.no_grad():
            _, outputs = self.model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # 转换为 numpy
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions

    def predict_single(self, data, verbose=True):
        """
        预测单个trial并打印结果

        参数:
            data: numpy array (22, 1000) 或 (1, 22, 1000)
            verbose: 是否打印详细信息

        返回:
            prediction: int (0-3)
            class_name: str
            probability: float
        """
        predictions, probabilities = self.predict(data, return_probabilities=True)

        prediction = predictions[0]
        probability = probabilities[0]
        class_name = self.class_names[prediction]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Prediction Result:")
            print(f"{'='*60}")
            print(f"Predicted Class: {prediction} - {class_name}")
            print(f"Confidence: {probability[prediction]:.2%}")
            print(f"\nAll Probabilities:")
            for i, prob in enumerate(probability):
                print(f"  {i} ({self.class_names[i]:12s}): {prob:.2%}")
            print("=" * 60)

        return prediction, class_name, probability[prediction]

    def predict_test_set(self, data_root='./data/2a/'):
        """
        在测试集上评估模型

        参数:
            data_root: 数据根目录

        返回:
            accuracy: float
            predictions: numpy array
            true_labels: numpy array
        """
        # 加载测试数据
        test_file = f"{data_root}/A0{self.subject_id}E.mat"
        print(f"\nLoading test data from: {test_file}")

        test_data = scipy.io.loadmat(test_file)
        data = test_data['data']  # (1000, 22, n_trials) 或 (22, 1000, n_trials)
        labels = test_data['label']  # (n_trials, 1) 或 (1, n_trials)

        # 转换格式
        if data.shape[0] == 1000:  # (1000, 22, n_trials)
            data = np.transpose(data, (2, 1, 0))  # (n_trials, 22, 1000)
        elif data.shape[1] == 1000:  # (22, 1000, n_trials)
            data = np.transpose(data, (2, 0, 1))  # (n_trials, 22, 1000)

        # 标签
        labels = labels.flatten() - 1  # 转换为 0-3

        print(f"Test data shape: {data.shape}")
        print(f"Test labels shape: {labels.shape}")

        # 预测
        predictions = self.predict(data)

        # 计算准确率
        accuracy = np.mean(predictions == labels)

        # 打印结果
        print(f"\n{'='*60}")
        print(f"Test Set Evaluation:")
        print(f"{'='*60}")
        print(f"Subject: {self.subject_id}")
        print(f"Total samples: {len(labels)}")
        print(f"Accuracy: {accuracy:.2%}")

        # 每个类别的准确率
        print(f"\nPer-class Accuracy:")
        for i in range(4):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                print(f"  {self.class_names[i]:12s}: {class_acc:.2%} ({np.sum(class_mask)} samples)")

        # 混淆矩阵
        print(f"\nConfusion Matrix:")
        print(f"{'':12s}", end='')
        for i in range(4):
            print(f"{self.class_names[i][:8]:>10s}", end='')
        print()

        for i in range(4):
            print(f"{self.class_names[i]:12s}", end='')
            for j in range(4):
                count = np.sum((labels == i) & (predictions == j))
                print(f"{count:10d}", end='')
            print()

        print("=" * 60)

        return accuracy, predictions, labels


def main():
    parser = argparse.ArgumentParser(description='Predict using trained EEG-Conformer model')
    parser.add_argument('--subject', type=int, required=True, help='Subject ID (1-9)')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model directory')
    parser.add_argument('--test_set', action='store_true', help='Evaluate on test set')
    parser.add_argument('--data_file', type=str, help='Path to .npy file containing EEG data')
    parser.add_argument('--data_root', type=str, default='./data/2a/', help='Data root directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    # 创建预测器
    predictor = EEGPredictor(args.subject, model_dir=args.model_dir, device=args.device)

    if args.test_set:
        # 在测试集上评估
        accuracy, predictions, labels = predictor.predict_test_set(data_root=args.data_root)

    elif args.data_file:
        # 预测单个文件
        print(f"Loading data from: {args.data_file}")
        data = np.load(args.data_file)
        print(f"Data shape: {data.shape}")

        if data.ndim == 2 or (data.ndim == 3 and data.shape[0] == 1):
            # 单个trial
            predictor.predict_single(data)
        else:
            # 多个trials
            predictions, probabilities = predictor.predict(data, return_probabilities=True)
            print(f"\nPredictions for {len(predictions)} trials:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                print(f"  Trial {i+1}: {pred} ({predictor.class_names[pred]}) - {prob[pred]:.2%}")

    else:
        print("Please specify either --test_set or --data_file")


if __name__ == "__main__":
    main()
