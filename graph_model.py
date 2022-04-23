import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from paper_work.my_model8.config import Config

config = Config()
def normalize(A, symmetric=True):
    # A = A+I
    # A = A + torch.eye(A.size(0)).to(config.DEVICE)
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5)).to(config.DEVICE)
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN(nn.Module):
    """
    Z = AXW
    """

    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A = A.float()
        self.A = normalize(self.A)
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        # self.fc2 = nn.Linear(dim_in, dim_in // 2, bias=False)
        # self.fc3 = nn.Linear(dim_in // 2, dim_out, bias=False)
        # self.relu = nn.ReLU()
        # self.layer_nor = nn.LayerNorm(768)
        # self.layer_nor2 = nn.LayerNorm(384)
        # self.layer_nor3 = nn.LayerNorm(dim_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.tanh = nn.Tanh()


    def forward(self, X):
        """
        计算三层gcn
        :param X:
        :return:
        """
        x = self.fc1(self.A.mm(X))
        # x = self.layer_nor(x)
        x = self.relu(x)
        # x = self.tanh(x)
        # x = self.sigmoid(x)
        x = self.dropout(x)

        # x = self.fc2(self.A.mm(x))
        # # x = self.layer_nor2(x)
        # x = self.relu(x)
        # # x = self.tanh(x)
        # # x = self.sigmoid(x)
        # x = self.dropout(x)
        #
        # # X = self.layer_nor(X)
        # x = self.fc3(self.A.mm(x))
        # # x = F.sigmoid(x)
        # # x = self.layer_nor3(x)
        # # x = self.relu(x)
        # # x = self.tanh(x)
        # x = self.sigmoid(x)
        # x = self.dropout(x)

        return x


class KnowledgeGraph(nn.Module):
    def __init__(self,
                 vocab_size,  # 分词表大小
                 A,  # 邻接矩阵
                 x,  # 所有长评+电影简介
                 graph_out_size,  # 图嵌入后输出的大小
                 input_embedding_dim=60,
                 tnn_out_dim=10,
                 ):
        super(KnowledgeGraph, self).__init__()

        self.input_embedding_dim = input_embedding_dim
        self.tnn_out_dim = tnn_out_dim

        self.A = A.to(config.DEVICE)  # 邻接矩阵
        self.x = x.to(config.DEVICE)
        self.graph_out_size = graph_out_size  # 图嵌入后的输出大小

        self.conv1 = nn.Conv1d(self.input_embedding_dim, tnn_out_dim, kernel_size=3, padding=1)
        self.relu = nn.Sigmoid()
        # self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(len(x[0]) * self.tnn_out_dim, 1200)
        self.linear2 = nn.Linear(1200, 8 * self.graph_out_size)

        # self.gcn = GCN(self.A, 768, self.graph_out_size)  # 邻接矩阵、输入维度、输出维度

    def forward(self, movie_ids):
        #   输入是所有长评
        # x = self.x.permute(0, 2, 1)
        # x = self.conv1(x)
        # x = self.relu(x)
        #
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.gcn(self.x)  # [movie&long_comment_count, out_dim]

        feature_list = []
        for i in movie_ids:
            feature_list.append(x[i])
        # print(len(feature_list))
        # print(len(feature_list[0]))

        feature_list = torch.stack(feature_list, 0)   # todo

        return feature_list  # 所有电影嵌入的结果
