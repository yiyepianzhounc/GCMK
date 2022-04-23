import torch
from pytorch_pretrained_bert import BertTokenizer

from paper_work.data_pre_utils import MyTokenizer
from paper_work.my_model8.graph_section.utils import MyAtrycx


class Config:
    def __init__(self):
        # 训练参数
        # self.DEVICE = torch.device("cpu")
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.info = '0 bert'
        self.epoch = 400
        self.lr = 0.1
        self.batch_size = 8
        self.flag = 0   #todo 0 只用文本特征；1文本+用户；2文本+图；3 全特征
        self.momentum = 0
        self.weight_decay = 0
        self.cos_train = False
        self.data_dir = '../maoyan/data/总表-修订.csv'

        # 训练集测试集比例
        self.test_rate = 0.2
        self.val_rate = 0.2
        self.random_noise = 0.2

        # 模型通用参数
        self.vocab_size = 51429  # 词表大小
        self.tokenizer = BertTokenizer.from_pretrained('../other_model/BERT/bert_pretrain/bert-base-chinese-vocab.txt')

        # 图参数
        adj_x = MyAtrycx(comment_intro_table_path='./graph_section/简介长评汇总表.csv')
        self.A = torch.tensor(adj_x.load_adjacency_matrix('./graph_section/adjacency_matrix.npy'))  # 邻接矩阵
        # self.x = adj_x.get_cut_x(tokenizer)   # todo
        # self.x = adj_x.save_cut_x(tokenizer)
        self.x = torch.tensor(adj_x.load_cut_x(path='./graph_section/cut_x.npy')).to(torch.float)   # todo
        print('X的维度:', self.x.shape)

        self.output_dim_g = 32    # 图输入对比网络的维度
        # self.output_dim_g2 = 16    # 图最后输出的维度

        # ANN模型参数
        self.embeding_dimension_a = 48     # 分词维度
        self.output_dim_a = 48     # 文本特征维度
        self.bert_path = '../other_model/BERT/bert_pretrain'

        # 用户特征输出参数
        self.input_dim_u = 11   # 用户特征数
        self.output_dim_u = 20    # 用户特征输出维度

