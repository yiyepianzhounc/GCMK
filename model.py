import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import *
from paper_work.my_model8.graph_section.graph_model import KnowledgeGraph
from pytorch_pretrained_bert import BertModel, BertTokenizer

config = Config()
device = config.DEVICE


class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        # 文本特征提取模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 冻结参数
        for param in self.bert.parameters():
            param.requires_grad = True

        self.linear_text = nn.Linear(768, config.output_dim_a)
        self.linear_sentence = nn.Linear(768, config.output_dim_g)
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x, mask):
        word_embedding, sentence_embedding = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
        return sentence_embedding


# 用户特征提取模型
class UserModel(nn.Module):
    def __init__(self):
        super(UserModel, self).__init__()
        self.input_dim = config.input_dim_u
        self.output_dim = config.output_dim_u
        self.li = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.li(x)
        x = self.relu(x)
        # x = self.LeakyReLU(x)
        x = self.dropout(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_dim):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.linear1 = nn.Linear(self.in_dim, 500)
        self.linear2 = nn.Linear(500, 300)
        self.linear3 = nn.Linear(300, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

        # self.att = SelfAttention(hidden_size=128, num_attention_heads=2, dropout_prob=0.4)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.sigmoid(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.linear3(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.softmax(x)
        return x


# 图+多层注意力模型
class GHAModel(nn.Module):
    def __init__(self, graph_dim=64,  # 图 电影表示的维度
                 ANN_dim=128,  # 文本特征输出维度
                 uf_dim=10,  # 用户特征输出维度
                 ):
        super(GHAModel, self).__init__()
        self.flag = config.flag
        # 外部知识图模型
        self.knowledge_graph = KnowledgeGraph(config.vocab_size, config.A, config.x, config.output_dim_g)
        # self.linear_graph = nn.Linear(config.output_dim_g, config.output_dim_g2)

        # 文本特征
        self.text_feature = TextModel()

        # 用户特征模型
        self.user_feature_model = UserModel()
        self.linear_u = nn.Linear(320,2)

        # 输入维度和flag有关
        if self.flag == 0:
            classifier_in_dim = 768
        elif self.flag == 1:
            classifier_in_dim = config.output_dim_u + config.output_dim_a
        elif self.flag == 2:
            # 乘以两倍
            classifier_in_dim = 2*768
        else:
            classifier_in_dim = 768*2
            # classifier_in_dim = config.output_dim_u + config.output_dim_g + config.output_dim_a
        # 分类器模型
        self.classifier = Classifier(in_dim=classifier_in_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, text, mask, user_features=None, movie_ids=None):
        """

        :param text: 评论文本
        :param user_features: 用户特征
        :param movie_ids: 电影序号
        :return:
        """
        sentence_feature = self.text_feature(text, mask)

        all_feature = sentence_feature
        if self.flag == 1:
            user_feature = self.user_feature_model(user_features)
            all_feature = torch.cat((all_feature, user_feature), 1)

        elif self.flag == 2:
            graph_feature = self.knowledge_graph(movie_ids)
            # graph_feature = torch.cat(((sentence_feature - graph_feature), torch.mul(sentence_feature, graph_feature)),
            #                           1)
            all_feature = torch.cat((all_feature, graph_feature), 1)
        elif self.flag == 3:
            user_feature = self.user_feature_model(user_features)
            graph_feature = self.knowledge_graph(movie_ids)
            # graph_feature = torch.cat(((sentence_feature - graph_feature), torch.mul(sentence_feature, graph_feature)),
            #                           1)
            all_feature = torch.cat((all_feature, graph_feature), 1)
            result = self.classifier(all_feature)
            all_feature = torch.cat((result, user_feature), 1)
            x = self.linear_u(all_feature)
            return x


        result = self.classifier(all_feature)
        return result


# https://www.cnblogs.com/jfdwd/p/11445135.html
if __name__ == '__main__':
    bert = BertModel.from_pretrained(config.bert_path)
    for param in bert.parameters():
        print(param.requires_grad)
    # x = torch.randint(100, [5, 128])
    # mask = torch.zeros([5, 128])
    # model_attention = ModelWithAttention()
    # # y = model_attention(x, mask)
    # self_attention = SelfAttention(128, 8, 0.2)
    # emb = nn.Embedding(6000, 128)
    # x = torch.zeros([5, 1, 128, 128])
    # # x = emb(x)
    # attention_block = AttentionBlock(1, 3, 128, mask)
    # y = attention_block(x)
    # print(y.shape)
    # writer = SummaryWriter('./logs')
    #
    # x = torch.randint(100, [8, 128]).to(device)
    # model = ANNModel().to(device)
    # # y = model-one-bert(x)
    #
    # x = torch.tensor([x, [1, 0]])
    #
    # writer.add_graph(model, x)
    # writer.close()
    # nn.MultiheadAttention(x, 8)
