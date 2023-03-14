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
import torch.nn.functional as F
import YinGuo_utils
from minepy import MINE
import json
use_cuda = torch.cuda.is_available()
#device=torch.device("cuda:0" if use_cuda else "cpu")
device=torch.device("cpu")
class GraphNetwork(nn.Module):
    def __init__(self,dim_in, dim_out):
        super().__init__()
        self.factor_k = 2
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rela_k = 8 #所包含的元关系个数
        self.iterate = 4 #细化意图分布的迭代次数
        self.node_catory = 8 #实体节点的类型
        self._init_weight()
        self.n_factors = 2  #n_factors=2,这可能是意图的个数
    def _init_weight(self):
        dim_k = int(self.dim_out / self.factor_k)
        # 节点类型u（用户）i（项目）D（导演）..... 总共有8种类型
        self.Wtk = nn.Parameter(torch.Tensor(self.node_catory, self.factor_k, self.dim_in, dim_k))
        nn.init.xavier_uniform_(self.Wtk, gain=nn.init.calculate_gain('relu'))

        self.at = nn.Parameter(torch.Tensor(self.rela_k, self.factor_k, 2 * dim_k))
        nn.init.xavier_uniform_(self.at, gain=nn.init.calculate_gain('relu'))
        #3*2 表示有u->i，和 i->u
        self.W = nn.Parameter(torch.Tensor(dim_k, dim_k))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))

        self.q_rela = nn.Parameter(torch.Tensor(self.rela_k, dim_k))
        nn.init.xavier_uniform_(self.q_rela, gain=nn.init.calculate_gain('relu'))

    def forward(self, edge_list, shape_list,emb):
        def fac(emb, W): #Content转换的过程，将u_emb映射到k个空间
            fac_emb = torch.matmul(emb, W)
            fac_emb = nn.LeakyReLU(negative_slope=0.2)(fac_emb)
            fac_emb = F.normalize(fac_emb, p=2, dim=2)
            return fac_emb

        def rela_update(indices, new_emb, old_emb, a, r_node, shape, q_rela):
            u, i = indices   #当e=0，表示(u-i)下的连接的边
            all_u = new_emb[:, u.long()] #提取出交互中所有的u的嵌入表示,总共交互77495次
            all_i = old_emb[:, i.long()] #提取出交互中所有i的交互表示

            """Disentangled Propagation Layer"""
            """公式(3.8)的操作，获得e_ts_k表示，该关系下不同意图下的注意力分数"""
            ui = torch.cat([all_u, all_i], dim=2)
            # print("ui",ui.shape) # 4,77495,10
            e_ts = torch.matmul(ui, a.unsqueeze(2)).squeeze()
            #print("e_ts",e_ts.shape) # 4,77495 每个目标节点在聚合一个源节点所需的权重
            e_ts_k = torch.relu(e_ts)

            """公式(3.9)的操作，所有意图下加权求和，计算节点的重要性"""
            r_k_edge = r_node[:, u.long()]
            #print("r_k_edge",r_k_edge.shape) #在特定关系下的，每个意图k的影响权重 4*77495
            e_ts_rela = torch.mul(e_ts_k, r_k_edge)
            # print("e_ts_rela",e_ts_rela.shape)#考虑不同节点对于在每一个意图下的重要程度不同 77495
            e_ts_rela = torch.sum(e_ts_rela, dim=0)  #edges 77495 通过加权求和所有方面来计算节点的重要程度

            """u-i变成稀疏矩阵，indices表示其指定的位置，e_ts_rela表示值（也就是相邻节点聚合的注意力分数）"""
            adj = torch.sparse_coo_tensor(indices, e_ts_rela, shape, device=old_emb.device) #稀疏矩阵转化成稠密矩阵
            #创建稀疏矩阵，indices为非零元素所在的位置，此参数为一个二维的数组，
            #第一维表示行、第二维表示列e.g indices=[[1, 4, 6], [3, 6, 7]]
            #e_ts_rela表示指定了非零元素的值，shape表示设置稀疏矩阵的大小
            adj = torch.sparse.softmax(adj, dim=1)
            #print("adj",adj.shape)  1843*65877
            emb_z = []
            #print("old_emb",old_emb[0].shape)

            for k in range(self.factor_k):
                #old_emb[k] = 65877 * 5
                #print("adj",adj.shape) 1843*65877
                zk = torch.sparse.mm(adj, old_emb[k]) #1843*5(k)
                zk = nn.LeakyReLU(negative_slope=0.2)(zk) #对应公式（3.10）
                emb_z.append(zk)
            emb_z = torch.stack(emb_z) #将k个意图的特征堆叠，形成[4,1843,5]    emb_z表示z_t_k四个堆叠在一起

            """inter-relation Aggregation 关系间的聚合"""
            """下面是公式(3.13), 所有元关系所揭示的语义信息，每个元关系关注特定的意图"""
            emb_z = torch.matmul(emb_z, self.W)  # self.W = 5(k)*5(k)
            new_r = torch.matmul(torch.tanh(emb_z), q_rela)
            #print("new_r",new_r.shape) #shape = 4 * 1843
            r = torch.softmax(new_r, dim=0)  #r代表更新后的k在各个源节点所影响的程度
            return r, emb_z


        """公式(3.15) 不同关系下汇聚到目标节点"""
        def new_fac(ego_emb, r_list, e_list, i_list):
            """i_list表示u的不同连接的索引位置  i = 0（表示u - i连接） i = 2 （表示u - tag连接）"""
            """[0,2]表示将u-i 和 u-tag 的信息都聚合称为u的最终的表示"""
            for i in i_list:
                ego_emb = ego_emb + torch.mul(e_list[i], r_list[i].unsqueeze(2))
            ego_emb = F.normalize(ego_emb, p=2, dim=2)
            #print("ego_emb.shape",ego_emb.shape)
            return ego_emb   #最终ego_emb表示为 user->[4,1843,5] item -->[4,65877,5] tag-->[4,3508,5]


        """1.将不同类型的节点的意图特征映射映射到不同的子空间中"""
        fac_entity_emb = []
        for i in range(len(emb)):
            fac_entity_emb.append(fac(emb[i], self.Wtk[i]))

        """2.进行初始化操作，初始每个意图的都是一样的"""
        r_rela_list = []  # 存储的是r_rela_k表示概率，表示是由于意图k从而使得关系rela连接到目标节点的意图分数
        for e in range(len(shape_list)):
            #print("shape_list[e][0]",shape_list[e][0])
            r_rela_list.append(torch.ones((self.factor_k, shape_list[e][0])) / self.factor_k)

        all_ego_emb = fac_entity_emb   # 意图分割后的整体向量
        all_new_emb = all_ego_emb   # 初始化更新后的向量表示
        index = [[0, 1], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],[1,6],[1,7]]  #[0,1]代表用户-项目，[1,0]代表项目-用户
        i_list = [[0],[1,2,3,4,5,6,7]]

        """3.开始迭代，从而细化意图分布，解开耦合的特征"""
        for t in range(self.iterate):  #迭代更新的轮数
            rela_list = []
            emb_list = []
            for e in range(len(edge_list)):
                # edge_list中包含着不同类型下的边，当e=0取的是（u-i）的边 index[0][0]表示当前index=0
                # 取得user的emb, ego_emb表示邻居节点的向量表示
                new_emb = all_new_emb[index[e][0]]
                ego_emb = all_ego_emb[index[e][1]]

                """Intra-relation Aggregation.进入关系内的信息聚合"""
                new_r, new_e = rela_update(edge_list[e], new_emb, ego_emb, \
                                           self.at[e], r_rela_list[e], shape_list[e], self.q_rela[e])
                rela_list.append(new_r)
                emb_list.append(new_e)

            """进入new_fac函数，不同关系下汇聚到目标节点"""
            for i in range(2):  #2的意思可能是i_list有两个列表[0]和[1,2,3,4,5,6,7]
                new_emb = new_fac(all_ego_emb[i], rela_list, emb_list, i_list[i])
                all_new_emb[i] = new_emb
                r_rela_list = rela_list  #这是根据model_disHan.py文件的137行增加的

        emb_list = []

        for e in all_new_emb:
            emb_list.append(torch.cat(list(e), dim=1))  #这个把k意图都给拼接
        return emb_list,all_new_emb

class GraphEncoder(Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()
        graph_edge = "disentangled_data/movie_kg_c/graph_edge"
        graph_dict = YinGuo_utils.pickle_load(graph_edge)
        self.edge_list = graph_dict['edge_list']
        self.shape_list = graph_dict['shape_list']
        num_list = graph_dict['num_list']
        self.num_list = num_list
        emb_size = 50
        embedding_path = 'disentangled_data/movie_kg_c/embedding_final_disInt'
        embeddings_list = YinGuo_utils.pickle_load(embedding_path)

        #embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
        id_save_path = 'disentangled_data/movie_kg_c/kg_re_Writer_film_id.txt'
        dict_old_entity_id = YinGuo_utils.pickle_load(id_save_path)
        old_entity_id = dict_old_entity_id['old_entity_id']
        self.embed = nn.ParameterList()
        user_emb = nn.Parameter(torch.Tensor(6040, emb_size)) #nn.Embbing参考gcn_kgat_attention文件GCN的user_embing
        nn.init.xavier_uniform_(user_emb, gain=nn.init.calculate_gain('relu'))
        self.embed.append(user_emb)
        for i in range(len(num_list)-1):
            embedding_entity = nn.Parameter(embeddings_list[i])
            self.embed.append(embedding_entity)
        self.gnns = nn.ModuleList()
        self.layers = 2 #卷积网络的层数
        self.n_factors = 2  #n_factors=2,这可能是意图的个数
        indim, outdim = 50, 50

        #self.ind = 'con'
        self.ind ='MIC'
        """初始化图神经网络"""
        for i in range(2):  #2的意思可能是卷积网络的层数layers=2
            self.gnns.append(GraphNetwork(indim, outdim))
            indim = outdim

        all_emb_int_dis = []
        """耦合相似度衡量"""

    # 总体独立性度量
    def _cul_cor(self,tensor_1,tensor_2,tensor_all):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = F.normalize(tensor_1)
            normalized_tensor_2 = F.normalize(tensor_2)
            distance =normalized_tensor_1.mm(normalized_tensor_2.t())
            n_samples = torch.tensor(tensor_1.shape[0],dtype=torch.float64)
            k = torch.tensor(0.0)
            dcov = (torch.maximum(torch.sum(distance) / (n_samples * n_samples), k) + 1e-8)
            return dcov  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            normalized_tensor_1 = F.relu(tensor_1)
            normalized_tensor_2 = F.relu(tensor_2)
            r1 = torch.sum(torch.sqrt(normalized_tensor_1),1,keepdim=True)
            k = torch.tensor(0.0)
            D1 = torch.sqrt(torch.maximum(r1 - 2 * torch.matmul(tensor_1,tensor_1.t()) + r1.transpose(0,1), k) + 1e-8)
            D1 = D1 - torch.mean(D1,dim=0,keepdim=True) - torch.mean(D1,dim=0,keepdim=True)+torch.mean(D1)

            r2 = torch.sum(torch.sqrt(normalized_tensor_2), 1, keepdim=True)
            D2 = torch.sqrt(torch.maximum(r2 - 2 * torch.matmul(tensor_2, tensor_1.t()) + r2.transpose(0, 1), k) + 1e-8)
            D2 = D2 - torch.mean(D2, dim=0, keepdim=True) - torch.mean(D2, dim=0, keepdim=True) + torch.mean(D2)

            n_samples = torch.tensor(tensor_2.shape[0], dtype=torch.float64)
            dcov_12 = (torch.maximum(torch.sum(D1*D2) / (n_samples * n_samples), k) + 1e-8)
            dcov_11 = (torch.maximum(torch.sum(D1 * D1) / (n_samples * n_samples), k) + 1e-8)
            dcov_22 = (torch.maximum(torch.sum(D2 * D2) / (n_samples * n_samples), k) + 1e-8)
            dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, k)) + 1e-10)

            return dcor

        def MAXMutualInformation(tensor_1, tensor_2):
            score = 0.0
            mine = MINE(alpha=0.6, c=15)
            for i in range(tensor_1.shape[0]-1):
                for j in range(tensor_2.shape[0]):
                    if(i>j):continue
                    else:
                        mine.compute_score(tensor_1[i].detach().numpy(), tensor_2[j].detach().numpy())
                        score+=mine.mic()
                        # print("score",score)
            dcor = score/ (tensor_1.shape[0] * tensor_1.shape[0]-1)
            return dcor

        def MutualInformation(tensor_all):
            disen_T = tensor_all.t()

            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            cor = MutualInformation(tensor_all)
        else:
            if self.ind == 'distance':
                cor = DistanceCorrelation(tensor_1, tensor_2)
            if self.ind == 'con':
                cor = CosineSimilarity(tensor_1, tensor_2)
            if self.ind == 'MIC':
                cor = MAXMutualInformation(tensor_1, tensor_2)
        return cor

    def create_cor_loss(self, u_i_factor):
        cor_loss = torch.tensor(0.0)
        for i in range(0, self.n_factors - 1):  #n_factors=2,这可能是意图的个数
            x = u_i_factor[i]
            y = u_i_factor[i+1]
            cor_loss += self._cul_cor(x,y,u_i_factor)  #调用了_cul_cor函数进行独立性检测
        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)
        return cor_loss

    def user_item_embedding(self):
        input_state = self.embed
        for gnn in self.gnns:
            output_state,all_emb = gnn(self.edge_list, self.shape_list, input_state)
            input_state = output_state
        return output_state,all_emb

    # def seq_embbing(self, state, state_, user, action, candi):
    def seq_embbing(self):

        # batch_seq = []
        # batch_seq_ = []
        cor_user = [1,2,3]
        cor_item = [7,8]
        all_emb,factor_emb = self.user_item_embedding()  #该函数返回值 output_state , all_emb

        user_all_emb = all_emb[0]
        new_emb = all_emb[1]

        cor_user_emb = user_all_emb[cor_user]
        cor_item_emb = new_emb[cor_item]

        cor_emb = torch.cat((cor_user_emb, cor_item_emb), 0)
        cor_factor_emb = torch.chunk(cor_emb,self.n_factors,dim=1)  #将tensor分为很多个块，n_factors是分割的块数
        cor_factor_emb = torch.stack(cor_factor_emb)
        cor = self.create_cor_loss(cor_factor_emb)
        """user_emb = user_all_emb[user]
        candi_emb = new_emb[candi]
        action_emb = new_emb[action]

        for s in state:
            seq_embedding = new_emb[s]
            seq_embedding = seq_embedding.unsqueeze(0)  #.unsqueeze(0)增加维度（0表示在第一个位置增加维度）
            batch_seq.append(seq_embedding)

        seq_embeddings = torch.cat(batch_seq, dim=0)
        for s in state_:
            seq_embedding_ = new_emb[s]
            seq_embedding_ = seq_embedding_.unsqueeze(0)
            batch_seq_.append(seq_embedding_)

        seq_embeddings_ = torch.cat(batch_seq_, dim=0)"""
        return cor,new_emb,user_all_emb

    def forward(self,mode, *input):
        if mode == 'sampling':
            return self.user_item_embedding()
        if mode =="learning":
            return self.seq_embbing(*input)

gnn = GraphEncoder()
# seq = [7,8]
# user = [1,2,3]
# a,b=gnn('sampling')
# print("a:",a)
# print("b:",b)
cor,new_emb,user_all_emb= gnn('learning')
print("uesr_all_emb:",user_all_emb.shape)
print("new_emb:",new_emb.shape)
predict = user_all_emb @ new_emb.T
print("score:",predict.shape)
rating = torch.nn.Sigmoid()(torch.matmul(user_all_emb, new_emb.t())) #多了sigmoid，激活函数的一种，它会将样本值映射到0到1之间
print("rating:",rating)




