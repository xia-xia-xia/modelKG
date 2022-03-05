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
        if kg_fn is None:
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
                """
                for i in range(E - len(graph[node])):
                    cur_neighbors = cur_neighbors + [node]
                cur_neighbors = cur_neighbors[:E]"""
        except:
            cur_neighbors = ([node] + kg_graph[node])[:E]
            """
            for i in range(E):
                cur_neighbors = cur_neighbors + [node]"""
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
    def forward(self, seq,user,choose_action=False):
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
        """
        if (choose_action == True):
            state_emb = torch.unsqueeze(seq_embeddings, 0)
            out, h = self.rnn(state_emb)
            state = h.permute(1, 0, 2)
        else:
            out, h = self.rnn(seq_embeddings)
            state = h.permute(1, 0, 2)"""
        return output

if __name__ == '__main__':
    rec=GraphConstructer(max_nodes=20, max_seq_length=10, kg_fn=None, cached_graph_fn=None)
    # rec = GraphEncoder(entity, emb_size,user_num =self.boundary_userid,embeddings=embeddings, max_seq_length=10, max_node=20, hiddim=50,
    #                             layers=2,cash_fn=None, fix_emb=False).to(device)
    # rec=rec.run()
    # embedding_path = 'embedding.vec.json'
    # embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
    # entity = embeddings.shape[0]
    # emb_size = embeddings.shape[1]
    rec = GraphEncoder(entity, emb_size,user_num =6040*0.8,embeddings=embeddings, max_seq_length=10, max_node=20, hiddim=50,
                                layers=2,cash_fn=None, fix_emb=False)
    print(rec)



