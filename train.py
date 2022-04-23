import sys
import os

sys.path.append('/mnt/workspace/tmp_user/ch/workspace')
import random
import time

print(sys.path)
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_pretrained_bert import BertTokenizer
from torch import nn, optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

from paper_work.my_model8.config import Config
from paper_work.data_pre_utils import MyTokenizer
from paper_work.maoyan.data_utils import MaoYanDataset
from paper_work.my_model8.model import GHAModel
from paper_work.my_model8.data_loader import GHADataset, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from paper_work.utils import DataDeal

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

config = Config()


def _weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model-one-bert
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.weight.data.normal_(0.0,1/2)
        # m.bias.data should be 0
        # m.bias.data.zero_()


def train(batch_size=config.batch_size, lr=config.lr, epoches=config.epoch, save_path='./model-one-bert', check_point=None):
    device = config.DEVICE
    tokenizer = config.tokenizer  # todo 分词表的选择：有可能需要重新生成
    maoyan_Dataset_test = MaoYanDataset(config.data_dir, '../maoyan/data/电影主页详情.csv')
    data = maoyan_Dataset_test.get_text_user_data()
    bert_dataset = GHADataset(data, tokenizer, flag=3)
    train_loader, val_loader, test_loader = get_dataloader(bert_dataset, batch_size=batch_size,
                                                           val_rate=config.val_rate,
                                                           test_rate=config.test_rate, random_noise=config.random_noise)

    if check_point:
        model = torch.load(check_point)
    else:
        # model_normal = NormalCnn(0, 0)
        # model-one-bert = ModelWithAttention()
        model = GHAModel()
        model = model.to(device)
        # model-one-bert.apply(_weights_init_normal)  # 加载权重

    model.train()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([p for p in model_normal.parameters() if p.requires_grad], lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    # optimizer = optim.SGD([{'params': model.knowledge_graph.parameters(), 'lr': lr * 10},
    #                     {'params': model.text_feature.parameters(), 'lr': lr * 10}], lr=lr, weight_decay=config.weight_decay)

    if config.cos_train:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)
    # optimizer = optim.SGD(model-one-bert.parameters(), lr=lr, weight_decay=config.weight_decay, momentum=config.momentum)

    # optimizer = optim.SGD([{'params': model-one-bert.user_feature_model.parameters(), 'lr': 0.01},
    #                        {'params': model-one-bert.knowledge_graph.parameters(), 'lr': 0.01},
    #                        {'params': model-one-bert.classifier.parameters(), 'lr': 0.01}
    #                        ],
    #                       lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)

    # 处理loss 和 混淆矩阵
    data_deal = DataDeal()

    for epoch in range(epoches):
        # train
        pbar = tqdm(train_loader)  # 进度条
        epoch_loss = []
        for text, mask, user_feature, movie_ids, label in pbar:
            # print(len(pbar))
            optimizer.zero_grad()
            model.zero_grad()
            text, mask, user_feature, movie_ids, target = text.to(device), mask.to(device), user_feature.to(
                device), movie_ids.to(
                device), label.to(device)

            output = model(text, mask, user_feature, movie_ids)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(f"flag:{config.flag}train：epoch{epoch} loss:{sum(epoch_loss) / len(pbar)}")

        train_loss = sum(epoch_loss) / len(pbar)
        data_deal.add_train_loss(train_loss)
        pbar.close()
        # print(f"train：epoch{epoch} loss{train_loss}")
        if config.cos_train:
            scheduler.step()  # 余弦退火
        cur_lr = optimizer.param_groups[-1]['lr']
        # cur_lr_list.append(cur_lr)
        print('cur_lr:', cur_lr)

        # val
        with torch.no_grad():
            model.eval()
            pbar = tqdm(val_loader)
            val_epoch_loss = []
            confusion_matrix = torch.zeros(2, 2)

            for text, mask, user_feature, movie_ids, label in pbar:
                text, mask, user_feature, movie_ids, target = text.to(device), mask.to(device), user_feature.to(
                    device), movie_ids.to(device), label.to(device)
                output = model(text, mask, user_feature, movie_ids)
                loss = criterion(output, target)

                val_epoch_loss.append(loss.item())

                result = output.argmax(1)
                for t, p in zip(result.view(-1), target.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            score = data_deal.add_confusion_matrix_and_cal(confusion_matrix)
            val_loss = sum(val_epoch_loss) / len(pbar)

            data_deal.add_val_loss(val_loss)
            pbar.close()
            # pbar.set_description(f"test: epoch{epoch} loss{val_loss} acc{score['准确率']} recall{score['召回率']}")
            print(
                f"test: epoch{epoch} loss{val_loss} acc{score['准确率']} recall{score['召回率']} F1 {score['F1']}")
            if epoch > 10 and (epoch + 1) % 5 == 0:
                name = f"./model-one-bert/flag{config.flag}batch_size{batch_size}-loss{val_loss}-F1{score['F1']}-acc{score['准确率']}-recall{score['召回率']}-pre{score['精确率']}-{epoch}"
                torch.save(model, name)

        # 画损失图
        data_deal.add_tensorboard_scalars(
            f'{config.info}flag{config.flag}epoch{epoches}-lr{lr}-batch_size{batch_size}/loss',
            {'train loss': train_loss, 'val loss': val_loss}, epoch)
        # 画混淆矩阵
        data_deal.add_tensorboard_scalars(
            f'{config.info}flag{config.flag}epoch{epoches}-lr{lr}-batch_size{batch_size}/confusion_matrix',
            {'[0,0]': confusion_matrix[0][0], '[0,1]': confusion_matrix[0][1], '[1,0]': confusion_matrix[1][0],
             '[1,1]': confusion_matrix[1][1], }, epoch)
        # 画结果图
        data_deal.add_tensorboard_scalars(
            f'{config.info}flag{config.flag}epoch{epoches}-lr{lr}-batch_size{batch_size}/result',
            {'acc': score['准确率'], 'recall': score['召回率'], 'pre': score['精确率'], 'F1': score['F1']}, epoch)

        data_deal.add_tensorboard_scalars(
            f'{config.info}flag{config.flag}epoch{epoches}-lr{lr}-batch_size{batch_size}/result_non',
            {'acc': score['准确率'], 'recall': score['召回率_non'], 'pre': score['精确率_non'], 'F1': score['F1_non']}, epoch)

    data_deal.write_confusion_matrix('./result.csv')


if __name__ == '__main__':
    train()
