import json
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
from numpy.random import RandomState

class GraphConstructer():
    def __init__(self, max_nodes, max_seq_length, kg_fn=None, cached_graph_fn=None):
        user_fn = 'rating_final_kg'
        kg_fn = 'kg.txt'
        self.max_nodes = max_nodes
        self.max_seq_length = max_seq_length
        self.cached_graph_fn = cached_graph_fn
        self.cached_graph = None
        self.user_graph = self.read_edges(user_fn)
        self.kg_graph = self.read_edges(kg_fn)


    # 将txt文件转化为图谱
    def read_edges(self, kg_fn):
        graph = {}
        edges = []
        with open(kg_fn, 'r') as f:
            for i, line in enumerate(f.readlines()):
                edges.append([int(s) for s in line.split()])
        for edge in tqdm(edges, desc='constructing graphs'):
            if graph.get(edge[0]) is None:
                graph[edge[0]] = []
            if graph.get(edge[1]) is None:
                graph[edge[1]] = []
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        return graph

    # 将知识图谱和用户项目二部图结合
    def get_graph(self, node,kg_graph,user_graph):
        E = self.max_nodes
        cur_neighbors = []
        try:
            if (len(user_graph[node]) >= E):
                cur_neighbors = ([node] + user_graph[node][:10] + kg_graph[node])[:E]
            else:
                cur_neighbors = ([node] + user_graph[node]+kg_graph[node])[:E]
        except:
            cur_neighbors = ([node] + kg_graph[node])[:E]

        nodes = torch.zeros((E)).long().to(device)
        adj = torch.zeros((E, E)).to(device)  #邻接矩阵？定义任意两个节点有指向（或者说相邻、相关）

        for i in range(len(cur_neighbors)):
            nodes[i] = torch.LongTensor(cur_neighbors[i:i+1]).to(device)
            adj[0][i] = 1
            adj[i][0] = 1
            adj[i][i] = 1
        return nodes, adj, len(cur_neighbors)

    def get_seq_graph(self, seq):
        """
        :param seq: a list of nodes [l]
        :return: seq_neighbor [L x E] seq_adjs [L x E x E]
        """
        assert len(seq) <= self.max_seq_length

        neighbors, adjs = [], []
        for s in seq:
            n, adj, _ = self.get_graph(s, self.kg_graph, self.user_graph)
            neighbors.append(n.unsqueeze(0))   #unsqueeze（）对数据维度进行扩充，意为在0的位置加了一维
            adjs.append(adj.unsqueeze(0))

        E, L, l = self.max_nodes, self.max_seq_length, len(adjs)
        seq_adjs = torch.zeros((L, E, E)).to(device)
        seq_neighbors = torch.zeros((L, E)).long().to(device)

        seq_adjs[:l] = torch.cat(adjs, dim=0)  # [l x E x E]
        seq_neighbors[:l] = torch.cat(neighbors, dim=0)  # [l x E]

        return seq_neighbors, seq_adjs
# 图卷积
class GraphConvolution(Module):
    # __init__初始化了一些用到的参数，包括输入和输出的维度，并初始化了每一层的权重
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # torch.nn.Parameter()使得self.weight成为了模型中根据训练可以改动的参数
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_concat = Parameter(torch.FloatTensor(in_features*2, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    # 参数初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  #服从均匀分布U(a,b)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # forward()方法说明了每一层对数据的操作。先将输入特征矩阵与权重矩阵相乘，在左乘邻接矩阵
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphEncoder(Module):
    def __init__(self, entity, emb_size, max_node, max_seq_length,user_num, embeddings=None, fix_emb=False, hiddim=100, layers=1,
                 cash_fn=None):
        self.n_items = 10
        super(GraphEncoder, self).__init__()
        self.entity = entity
        # nn.Embedding是一个简单的存储固定大小的词典的嵌入向量的查找表，
        # 给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。
        self.entity_user_emb = nn.Embedding(entity+6040, emb_size)
        user_emb = nn.Parameter(torch.Tensor(6040,emb_size))
        nn.init.xavier_uniform_(user_emb,gain=nn.init.calculate_gain('relu')) #均匀分布，使用gain参数来自定义初始化的标准差匹配特定的激活函数

        entity_user_emb = torch.cat([embeddings,user_emb], dim=0)
        self.entity_user_emb.weight = nn.Parameter(entity_user_emb)
        self.random_state = RandomState(1)
        """
        self.embedding = nn.Embedding(entity, emb_size)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings, freeze=fix_emb)"""
        self.constructor = GraphConstructer(max_nodes=max_node, max_seq_length=max_seq_length, cached_graph_fn=cash_fn)
        self.layers = layers
        indim, outdim = emb_size, hiddim
        self.gnns = nn.ModuleList()
        #self.rnn = nn.GRU(50, 50, 1, batch_first=True)
        # 对每个边产生一个注意力权重，torch.from_numpy()方法把数组转换成张量。
        self.attention_weights = nn.Parameter(torch.from_numpy(0.1 * self.random_state.rand(self.n_items)).float())
        #self.attention_weights = Parameter(torch.FloatTensor(self.n_items))
        #nn.init.xavier_uniform_(self.attention_weights, gain=nn.init.calculate_gain('relu'))

        for l in range(layers):
            self.gnns.append(GraphConvolution(indim, outdim))
            indim = outdim

    # forward()方法说明了每一层对数据的操作
    # def forward(self, seq,user,choose_action=False):
    def forward(self, seq, user):
        user = (torch.LongTensor(user)).to(device)
        batch_seq_adjs = []
        batch_seq_neighbors = []
        for s in seq:
            neighbors, adj = self.constructor.get_seq_graph(s)
            batch_seq_neighbors.append(neighbors[None, :])
            batch_seq_adjs.append(adj[None, :])
        input_neighbors_ids = torch.cat(batch_seq_neighbors, dim=0)
        input_adjs = torch.cat(batch_seq_adjs, dim=0)
        input_state = self.entity_user_emb(input_neighbors_ids)
        # GRU的思想
        for gnn in self.gnns:
            output_state = gnn(input_state, input_adjs)
            input_state = output_state

        seq_embeddings = output_state[:, :, :1, :].contiguous().squeeze()  # [N x L x d]

        user_emb = self.entity_user_emb(user)
        '''
        if (choose_action):
            # torch.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。通过squeeze()函数转换后，要显示的数组变成了秩为1的数组
            user_emb = user_emb.squeeze(0)
            items = seq_embeddings.t()  #.t()得到转置矩阵
            right = items @ self.attention_weights
            middle = user_emb * right
            # torch.flatten作用是改变张量的维度和维数，从指定的维度开始将后面维度的维数全部展成一个维度，（即被”推平“）
            # flatten操作是当从一个卷积层过渡到一个全连接层时必须在神经网络中发生的操作。
            output = torch.cat((user_emb, middle, right), 0).flatten()
        else:
            items = seq_embeddings.permute(0, 2, 1)  #.permute(0,2,1)将维度索引1和维度索引2交换位置
            right = items @ self.attention_weights
            middle = user_emb * right
            output = torch.cat((user_emb, middle, right), 1)
        '''
        user_emb = user_emb.squeeze(0)
        items = seq_embeddings.t()
        right = items @ self.attention_weights
        middle = user_emb * right
        output = torch.cat((user_emb, middle, right), 0).flatten()
        print(output)
        # print(user_emb)
        return output


if __name__ == '__main__':
    # graph = GraphConstructer(max_nodes=20, max_seq_length=10, kg_fn=None, cached_graph_fn=None)
    # rec=rec.run()
    embedding_path = 'embedding.vec.json'
    embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
    entity = embeddings.shape[0]
    emb_size = embeddings.shape[1]
    output = GraphEncoder(entity, emb_size,user_num =6040*0.8,embeddings=embeddings, max_seq_length=10, max_node=20, hiddim=50,
                                layers=2,cash_fn=None, fix_emb=False).to(device)
    output_state=[[0,1,2]]
    user_emb=[4]
    user_like = output(output_state,user_emb)
    entity_user_emb = torch.FloatTensor([2.2119e-02,  1.5431e-01, -1.3462e-01, -2.0840e-01,  1.0316e-01,
         6.0314e-02, -4.6969e-02,  3.6325e-01, -1.4954e-01,  3.0862e-01,
        -1.0560e-01, -4.9485e-02,  2.0255e-02, -1.0944e-01, -2.1185e-03,
        -9.8946e-02, -5.9448e-02,  1.2312e-01, -2.9753e-02,  1.6686e-02,
         4.3468e-02, -6.3488e-02,  1.0569e-01,  1.1969e-01,  4.9157e-02,
        -9.3380e-02,  1.3321e-01, -1.7150e-01,  1.3460e-01, -5.1788e-02,
         4.3006e-03, -1.8060e-01, -1.2939e-01, -5.7142e-03, -2.6813e-01,
        -1.4891e-01, -1.3504e-01,  4.2394e-01, -2.1512e-02, -2.0356e-01,
         1.6119e-01, -3.5521e-01,  3.5796e-01,  4.8034e-02,  1.7306e-01,
         3.4239e-02,  2.1302e-02,  1.6394e-02, -2.5969e-01, -1.0587e-01,
         3.6225e-03,  2.1393e-02, -3.5038e-02,  1.6958e-02, -4.3091e-03,
        -9.3749e-04, -1.0550e-02,  4.8271e-02, -1.0318e-02, -9.5484e-03,
        -5.6712e-04,  9.9557e-03, -2.3516e-03, -1.2833e-02, -8.8858e-04,
         6.4192e-03,  1.0874e-02, -3.3720e-02, -5.6151e-03,  8.7321e-04,
        -2.5284e-03,  4.9485e-03,  2.5253e-02,  3.0815e-02, -4.2043e-05,
         1.9149e-02, -7.9120e-03,  2.0819e-04, -2.7587e-02, -3.5310e-03,
         5.8832e-04,  2.6306e-02, -1.5342e-02, -1.3050e-04,  3.5746e-02,
        -2.2094e-02,  1.8817e-02,  1.1329e-02, -2.5747e-03, -2.6005e-03,
         5.1991e-03,  2.8432e-02,  1.1927e-01,  1.8158e-04, -4.0209e-02,
        -3.4483e-03,  1.0086e-03, -3.6051e-03,  2.5712e-02, -3.6974e-02,
         1.6377e-01,  1.3864e-01,  2.6026e-01, -8.1369e-02, -4.1772e-02,
        -1.5544e-02,  2.2462e-01,  1.3289e-01,  6.8999e-02, -3.0939e-02,
         5.3704e-03, -2.0118e-01, -1.1610e-01,  1.1726e-01,  4.1945e-01,
        -6.4875e-02, -1.8291e-01, -2.7389e-01,  1.8872e-01,  5.2333e-02,
        -5.8167e-02, -7.7944e-02,  2.3893e-01,  2.5747e-01, -8.5528e-04,
        -2.0507e-01, -5.9393e-02, -1.2140e-03, -2.0496e-01,  6.8181e-02,
         1.3680e-01, -1.4566e-01,  1.1857e-01,  2.2838e-02, -1.3332e-01,
         1.4837e-01, -1.3935e-01,  2.6722e-02,  1.1969e-01,  1.2775e-02,
         3.2253e-02, -8.0045e-02,  3.3319e-01,  3.7803e-03, -2.3235e-01,
        -1.0071e-01,  4.7347e-02, -2.1991e-01, -9.9011e-02,  3.4926e-01]).to(device)
    rec = user_like @ entity_user_emb
    print("rec:",rec)





