"""
修改版的 conformer.py - 保存每个 subject 的最佳模型

这个脚本会:
1. 为每个 subject 训练模型
2. 保存最佳模型到 ./models/subject_X_best.pth
3. 保存训练统计信息到 ./models/subject_X_stats.json
"""

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import datetime
import time
import scipy.io
import json

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


# ========== 从 conformer.py 导入模型定义 ==========
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


# ========== 训练和保存逻辑 ==========
class ExP():
    def __init__(self, nsub, save_dir='./models'):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.root = './data/2a/'
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_write = open(f"{self.save_dir}/log_subject{self.nSub}.txt", "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # 保存标准化参数
        self.normalization_params = {
            'mean': float(target_mean),
            'std': float(target_std)
        }

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        best_epoch = 0

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img_batch, label_batch) in enumerate(self.dataloader):
                img_batch = Variable(img_batch.cuda().type(self.Tensor))
                label_batch = Variable(label_batch.cuda().type(self.LongTensor))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img_batch = torch.cat((img_batch, aug_data))
                label_batch = torch.cat((label_batch, aug_label))

                tok, outputs = self.model(img_batch)
                loss = self.criterion_cls(outputs, label_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label_batch).cpu().numpy().astype(int).sum()) / float(label_batch.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc

                if acc > bestAcc:
                    bestAcc = acc
                    best_epoch = e
                    Y_true = test_label
                    Y_pred = y_pred

                    # 保存最佳模型
                    model_path = f"{self.save_dir}/subject_{self.nSub}_best.pth"
                    torch.save({
                        'epoch': e,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_acc': bestAcc,
                        'normalization_params': self.normalization_params,
                    }, model_path)
                    print(f"  ✓ Saved best model (acc={bestAcc:.4f}) to {model_path}")

        averAcc = averAcc / num

        # 保存统计信息
        stats = {
            'subject': self.nSub,
            'best_accuracy': float(bestAcc),
            'average_accuracy': float(averAcc),
            'best_epoch': int(best_epoch),
            'total_epochs': self.n_epochs,
            'normalization_params': self.normalization_params
        }

        stats_path = f"{self.save_dir}/subject_{self.nSub}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f'\nSubject {self.nSub} Training Complete:')
        print(f'  Best accuracy: {bestAcc:.4f} (at epoch {best_epoch})')
        print(f'  Average accuracy: {averAcc:.4f}')
        print(f'  Model saved to: {self.save_dir}/subject_{self.nSub}_best.pth')
        print(f'  Stats saved to: {stats_path}')

        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.close()

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    parser = argparse.ArgumentParser(description='Train and save EEG-Conformer models')
    parser.add_argument('--subjects', type=str, default='all', help='Subject IDs (e.g., "1,3,5" or "all")')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random)')

    args = parser.parse_args()

    # 确定要训练的 subjects
    if args.subjects == 'all':
        subjects = list(range(1, 10))
    else:
        subjects = [int(s) for s in args.subjects.split(',')]

    print("=" * 80)
    print("EEG-Conformer Training with Model Saving")
    print("=" * 80)
    print(f"Subjects: {subjects}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    best_accs = []
    aver_accs = []
    result_write = open(f"{args.save_dir}/training_summary.txt", "w")

    for i in subjects:
        starttime = datetime.datetime.now()

        if args.seed is not None:
            seed_n = args.seed
        else:
            seed_n = np.random.randint(2021)

        print(f'\n{"="*80}')
        print(f'Subject {i} (Seed: {seed_n})')
        print("=" * 80)

        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        exp = ExP(i, save_dir=args.save_dir)
        bestAcc, averAcc, Y_true, Y_pred = exp.train()

        result_write.write(f'Subject {i} : Seed is: {seed_n}\n')
        result_write.write(f'Subject {i} : The best accuracy is: {bestAcc}\n')
        result_write.write(f'Subject {i} : The average accuracy is: {averAcc}\n\n')

        endtime = datetime.datetime.now()
        print(f'Subject {i} duration: {endtime - starttime}')

        best_accs.append(bestAcc)
        aver_accs.append(averAcc)

    overall_best = np.mean(best_accs)
    overall_aver = np.mean(aver_accs)

    result_write.write(f'\n{"="*80}\n')
    result_write.write(f'**The average Best accuracy is: {overall_best}\n')
    result_write.write(f'The average Aver accuracy is: {overall_aver}\n')
    result_write.close()

    print(f'\n{"="*80}')
    print('All Subjects Training Complete!')
    print("=" * 80)
    print(f'Average Best Accuracy: {overall_best:.4f}')
    print(f'Average Aver Accuracy: {overall_aver:.4f}')
    print(f'Models saved in: {args.save_dir}')


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
