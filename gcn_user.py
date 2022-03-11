import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
from numpy.random import RandomState
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

class GraphConstructer():
    def __init__(self, max_nodes, max_seq_length, kg_fn=None, cached_graph_fn=None):
        kg_fn = '../Data/kg.txt'
        user_fn = '../Data/rating_final_kg_train'
        self.graph = self.read_edges(kg_fn)
        self.user_graph = self.read_edges(user_fn)
        self.cached_graph_fn = cached_graph_fn
        self.cached_graph = None
        #self.graph = self.read_edges(kg_fn)
        self.max_nodes = max_nodes
        self.max_seq_length = max_seq_length

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

    def get_graph(self, node):
        E = self.max_nodes

        try:
            if (len(self.user_graph[node]) >= 10):
                cur_neighbors = ([node] + self.user_graph[node][:10] + self.graph[node])[:E]
            else:
                cur_neighbors = ([node] + self.user_graph[node]+self.graph[node])[:E]
                """
                for i in range(E - len(graph[node])):
                    cur_neighbors = cur_neighbors + [node]
                cur_neighbors = cur_neighbors[:E]"""
        except:
            cur_neighbors = ([node] + self.graph[node])[:E]
        """
        cur_neighbors = ([node] + (self.user_graph[node])[:5]+self.graph[node])[:E]"""
        nodes = torch.zeros((E)).long().to(device)
        adj = torch.zeros((E, E)).to(device)

        for i in range(len(cur_neighbors)):
            nodes[i] = torch.LongTensor(cur_neighbors[i:i+1]).to(device)
            adj[0][i] = 1
            adj[i][0] = 1
            adj[i][i] = 1
        return nodes, adj, len(cur_neighbors)

    def get_cached_graph(self, node):
        if self.cached_graph is None:
            self.cached_graph = pickle.load(gzip.open(self.cached_graph_fn, 'rb'))
        node, adj = self.cached_graph[node]

        num_neighbors = np.sum(node.numpy() != 0)
        node = node.to(device)
        adj = adj.to(device)
        return node, adj, num_neighbors

    def get_seq_graph(self, seq):
        """
        :param seq: a list of nodes [l]
        :return: seq_neighbor [L x E] seq_adjs [L x E x E]
        """
        assert len(seq) <= self.max_seq_length

        neighbors, adjs = [], []
        for s in seq:
            n, adj, _ = self.get_graph(s)
            #n, adj, _ = self.get_cached_graph(s)
            neighbors.append(n.unsqueeze(0))
            adjs.append(adj.unsqueeze(0))

        E, L, l = self.max_nodes, self.max_seq_length, len(adjs)
        seq_adjs = torch.zeros((L, E, E)).to(device)
        seq_neighbors = torch.zeros((L, E)).long().to(device)

        seq_adjs[:l] = torch.cat(adjs, dim=0)  # [l x E x E]
        seq_neighbors[:l] = torch.cat(neighbors, dim=0)  # [l x E]

        return seq_neighbors, seq_adjs


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            in_features: int
                节点输入特征的维度
            out_features: int
                输出特征维度
            bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        """
        在看过很多博客的时候发现了一个用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size)),首先可以把这个函数理解为类型转换函数，
        将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的
        parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v
        变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        """
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#权重初始化进行参数初始化的操作有Linear,Conv,BN等
        #这个方法是系统默认的初始化方法
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphEncoder(Module):
    def __init__(self,  entity, emb_size, max_node, max_seq_length,embeddings=None,fix_emb=False, hiddim=100, layers=1, cash_fn=None):
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(entity + 6040,emb_size)
        user_emb = nn.Parameter(torch.Tensor(6040, emb_size))
        nn.init.xavier_uniform_(user_emb, gain=nn.init.calculate_gain('relu'))
        entity_user_emb = torch.cat([embeddings, user_emb], dim=0)
        self.embedding.weight = nn.Parameter(entity_user_emb)
        self.random_state = RandomState(1)
        """
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)
        """

        self.constructor = GraphConstructer(max_nodes=max_node, max_seq_length=max_seq_length, cached_graph_fn=cash_fn)
        self.layers = layers

        indim, outdim = emb_size, hiddim
        self.gnns = nn.ModuleList()
        #ModuleList 可以存储多个 model，传统的方法，
        # 一个model 就要写一个 forward ，但是如果将它们存到一个 ModuleList 的话，就可以使用一个 forward。
        for l in range(layers):
            self.gnns.append(GraphConvolution(indim, outdim))
            indim = outdim

    def forward(self, seq):
        """

        :param seq: [N x L] ;candi:[N x K]
        :return: [N x L x d]
        """
        batch_seq_adjs = []
        batch_seq_neighbors = []
        for s in seq:
            neighbors, adj = self.constructor.get_seq_graph(s)
            batch_seq_neighbors.append(neighbors[None, :])
            batch_seq_adjs.append(adj[None, :])

        input_neighbors_ids = torch.cat(batch_seq_neighbors, dim=0)
        input_adjs = torch.cat(batch_seq_adjs, dim=0)   # [N x L x E x E]
        input_state = self.embedding(input_neighbors_ids)  # [N x L x E x d]

        for gnn in self.gnns:
            output_state = gnn(input_state, input_adjs)
            input_state = output_state

        seq_embeddings = output_state[:, :, :1, :].contiguous().squeeze()  # [N x L x d]

        return seq_embeddings
