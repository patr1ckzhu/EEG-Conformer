"""
EEG Conformer - IMPROVED VERSION (Issue #40)

Implements performance improvements from GitHub Issue #40:
1. Simplified fully connected layer (single layer instead of 3 layers)
2. Pre-Norm architecture (already implemented in original)
3. Increased data augmentation frequency (N_AUG = 3)
4. Validation set split (20-30% of training data)

Expected improvements: +5-10% accuracy
"""

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
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


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            # IMPROVEMENT #2: Pre-Norm architecture (LayerNorm before MHA)
            # This is already correct in the original code!
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

        # IMPROVEMENT #1: Simplified FC layer (single layer instead of 3 layers)
        # Original: 2440 -> 256 -> 32 -> 4 (3 layers with ELU and Dropout)
        # Improved: 2440 -> 4 (1 layer with Dropout)
        # Expected improvement: +2-3% accuracy
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2440, n_classes)  # Direct mapping!
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


class ExP():
    def __init__(self, nsub, validate_ratio=0.2):
        """
        Improved training class with validation set and increased data augmentation

        Args:
            nsub: Subject number (1-9)
            validate_ratio: Ratio of training data to use for validation (default: 0.2)
        """
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        # IMPROVEMENT #4: Add validation set
        self.validate_ratio = validate_ratio
        self.number_class = 4
        # IMPROVEMENT #3: Increase data augmentation frequency
        self.number_augmentation = 3  # Augment 3 times instead of 1

        self.start_epoch = 0
        self.root = './data/2a/'

        self.log_write = open("./results/log_subject%d_improved.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()


    # Segmentation and Reconstruction (S&R) data augmentation
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
        """
        Load and preprocess data with validation set split
        """
        # train data
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        # Shuffle before split
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # IMPROVEMENT #4: Split into train and validation sets
        num_samples = len(self.allData)
        num_validate = int(self.validate_ratio * num_samples)
        num_train = num_samples - num_validate

        self.trainData = self.allData[:num_train]
        self.trainLabel = self.allLabel[:num_train]
        self.valData = self.allData[num_train:]
        self.valLabel = self.allLabel[num_train:]

        print(f"Train samples: {num_train}, Validation samples: {num_validate}")

        # test data (unchanged)
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize (use training data statistics only)
        target_mean = np.mean(self.trainData)
        target_std = np.std(self.trainData)
        self.trainData = (self.trainData - target_mean) / target_std
        self.valData = (self.valData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # Keep allData for augmentation (also standardized)
        self.allData = (self.allData - target_mean) / target_std

        return self.trainData, self.trainLabel, self.valData, self.valLabel, self.testData, self.testLabel


    def train(self):
        """
        Training loop with validation set and increased data augmentation
        """
        train_data, train_label, val_data, val_label, test_data, test_label = self.get_source_data()

        # Create dataloaders
        train_img = torch.from_numpy(train_data)
        train_label_tensor = torch.from_numpy(train_label - 1)
        dataset = torch.utils.data.TensorDataset(train_img, train_label_tensor)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        val_data_tensor = torch.from_numpy(val_data)
        val_label_tensor = torch.from_numpy(val_label - 1)
        val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_label_tensor)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_data_tensor = torch.from_numpy(test_data)
        test_label_tensor = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_label_tensor)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Move validation and test data to GPU
        val_data_gpu = Variable(val_data_tensor.type(self.Tensor))
        val_label_gpu = Variable(val_label_tensor.type(self.LongTensor))
        test_data_gpu = Variable(test_data_tensor.type(self.Tensor))
        test_label_gpu = Variable(test_label_tensor.type(self.LongTensor))

        bestAcc = 0
        bestValAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # IMPROVEMENT #3: Increased data augmentation frequency
                # Augment N_AUG times instead of 1 time
                for aug_idx in range(self.number_augmentation):
                    aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                    img = torch.cat((img, aug_data))
                    label = torch.cat((label, aug_label))

                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation and test process
            if (e + 1) % 1 == 0:
                self.model.eval()

                # Validation set evaluation
                with torch.no_grad():
                    val_tok, val_cls = self.model(val_data_gpu)
                    loss_val = self.criterion_cls(val_cls, val_label_gpu)
                    val_pred = torch.max(val_cls, 1)[1]
                    val_acc = float((val_pred == val_label_gpu).cpu().numpy().astype(int).sum()) / float(val_label_gpu.size(0))

                # Test set evaluation
                with torch.no_grad():
                    test_tok, test_cls = self.model(test_data_gpu)
                    loss_test = self.criterion_cls(test_cls, test_label_gpu)
                    y_pred = torch.max(test_cls, 1)[1]
                    test_acc = float((y_pred == test_label_gpu).cpu().numpy().astype(int).sum()) / float(test_label_gpu.size(0))

                # Training accuracy
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Val loss: %.6f' % loss_val.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train acc: %.4f' % train_acc,
                      '  Val acc: %.4f' % val_acc,
                      '  Test acc: %.4f' % test_acc)

                self.log_write.write(str(e) + "  Val: " + str(val_acc) + "  Test: " + str(test_acc) + "\n")
                num = num + 1
                averAcc = averAcc + test_acc

                # IMPROVEMENT #4: Save best model based on VALIDATION accuracy (not test)
                if val_acc > bestValAcc:
                    bestValAcc = val_acc
                    bestAcc = test_acc  # Record corresponding test accuracy
                    Y_true = test_label_gpu
                    Y_pred = y_pred
                    # Save best model
                    torch.save(self.model.module.state_dict(), f'model_improved_subject{self.nSub}.pth')

        averAcc = averAcc / num
        print('='*60)
        print(f'IMPROVED MODEL - Subject {self.nSub}')
        print('The average test accuracy is:', averAcc)
        print('The best validation accuracy is:', bestValAcc)
        print('The best test accuracy is:', bestAcc)
        print('='*60)
        self.log_write.write('='*60 + '\n')
        self.log_write.write('The average test accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best validation accuracy is: ' + str(bestValAcc) + "\n")
        self.log_write.write('The best test accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result_improved.txt", "w")
    result_write.write("="*60 + "\n")
    result_write.write("IMPROVED MODEL - GitHub Issue #40\n")
    result_write.write("Improvements:\n")
    result_write.write("1. Simplified FC layer (2440 -> 4 direct)\n")
    result_write.write("2. Pre-Norm architecture (already correct)\n")
    result_write.write("3. Increased data augmentation (3x)\n")
    result_write.write("4. Validation set (20% of training data)\n")
    result_write.write("="*60 + "\n\n")

    for i in range(9):
        starttime = datetime.datetime.now()

        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        exp = ExP(i + 1, validate_ratio=0.2)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST TEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best test accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average test accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

    best = best / 9
    aver = aver / 9

    result_write.write('\n' + '='*60 + '\n')
    result_write.write('**The average Best test accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver test accuracy is: ' + str(aver) + "\n")
    result_write.write('='*60 + '\n')
    result_write.close()

    print('\n' + '='*60)
    print('FINAL RESULTS - IMPROVED MODEL')
    print('Average Best Test Accuracy:', best)
    print('Average Mean Test Accuracy:', aver)
    print('='*60)


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))