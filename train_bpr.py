import torch
import json
from gcn_kgat_attention import *
import numpy as np
import random
from collections import defaultdict
from collections import Counter
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
class BPR:
    embedding_path = 'data/movie_embedding.vec.json'
    embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
    item_count = embeddings.shape[0]    # self.item_count = 182011
    emb_size = embeddings.shape[1]
    output = GraphEncoder(item_count, emb_size, user_num=6040 * 0.8, embeddings=embeddings, max_seq_length=10, max_node=20,
                          hiddim=50,
                          layers=2, cash_fn=None, fix_emb=False).to(device)

    def train2(self, user_items_dict):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1000  #训练次数
        beginUserId = 182011
        # 设定的用户矩阵
        U = np.zeros((len(user_items_dict), 50))
        for userId in user_items_dict.keys():
            items = list(user_items_dict[userId])
            for item in range(0, len(items), 10):
                seq = np.array([items[item:item + 10]])
                user_like = self.output(seq, [userId]) * 0.01
                user_like = user_like.cuda().data.cpu()
            U[userId-beginUserId] = user_like.numpy()
        # 设定的项目矩阵
        V = (self.embeddings * 0.01).numpy()
        biasV = np.random.rand(self.item_count) * 0.01

        for count in range(train_count):
            for userId in user_items_dict.keys():
                # 从用户的U-I中随机选取1个Item
                i = random.choice(list(user_items_dict[userId]))
                # 随机选取一个用户u没有评分的项目
                j = random.randint(1, self.item_count-1)
                while j in user_items_dict[userId]:
                    j = random.randint(1, self.item_count)
                # BPR
                r_ui = np.dot(U[userId-beginUserId], V[i].T) + biasV[i]
                r_uj = np.dot(U[userId-beginUserId], V[j].T) + biasV[j]
                r_uij = r_ui - r_uj
                loss_func = -1.0 / (1 + np.exp(r_uij))
                # 更新2个矩阵
                U[userId-beginUserId] += -lr * (loss_func * (V[i] - V[j]) + reg * U[userId-beginUserId])
                V[i] += -lr * (loss_func * U[userId-beginUserId] + reg * V[i])
                V[j] += -lr * (loss_func * (-U[userId-beginUserId] + reg * V[j]))
                # 更新偏置项
                biasV[i] += -lr * (loss_func + reg * biasV[i])
                biasV[j] += -lr * (-loss_func + reg * biasV[j])
        user_scoreList_dict = U @ V.T  # 将训练完成的矩阵內积
        return user_scoreList_dict

    def train3(self,userId,user_items_dict,ignoredItem):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1000  # 训练次数
        # 用户矩阵
        items = list(user_items_dict[userId])
        items.remove(ignoredItem)
        for item in range(0, len(items), 10):
            seq = np.array([items[item:item + 10]])
            user_like = self.output(seq, [userId]) * 0.01
        user_like = user_like.cuda().data.cpu().numpy()
        # 项目矩阵
        V = (self.embeddings * 0.01).numpy()
        biasV = np.random.rand(self.item_count) * 0.01

        for count in range(train_count):
            # 从用户的U-I中随机选取1个Item
            i = random.sample(items, 1)[0]
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count-1)
            while j in user_items_dict[userId]:
                j = random.randint(1, self.item_count - 1)
            # BPR
            r_ui = np.dot(user_like, V[i].T) + biasV[i]
            r_uj = np.dot(user_like, V[j].T) + biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            user_like += -lr * (loss_func * (V[i] - V[j]) + reg * user_like)
            V[i] += -lr * (loss_func * user_like + reg * V[i])
            V[j] += -lr * (loss_func * (-user_like + reg * V[j]))
            # 更新偏置项
            biasV[i] += -lr * (loss_func + reg * biasV[i])
            biasV[j] += -lr * (-loss_func + reg * biasV[j])
        predict = user_like @ V.T  # 将训练完成的矩阵內积
        return list(predict)

    def train4(self, userId, user_items_dict, ignoredItemList):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1000  # 训练次数
        # 用户矩阵
        items = list(user_items_dict[userId])
        for igt in ignoredItemList:
            items.remove(igt)
        for item in range(0, len(items), 10):
            seq = np.array([items[item:item + 10]])
            user_like = self.output(seq, [userId]) * 0.01
        user_like = user_like.cuda().data.cpu().numpy()
        # 项目矩阵
        V = (self.embeddings * 0.01).numpy()
        biasV = np.random.rand(self.item_count) * 0.01

        for count in range(train_count):
            # 从用户的U-I中随机选取1个Item
            i = random.sample(items, 1)[0]
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count-1)
            while j in user_items_dict[userId]:
                j = random.randint(1, self.item_count)
            # BPR
            r_ui = np.dot(user_like, V[i].T) + biasV[i]
            r_uj = np.dot(user_like, V[j].T) + biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            user_like += -lr * (loss_func * (V[i] - V[j]) + reg * user_like)
            V[i] += -lr * (loss_func * user_like + reg * V[i])
            V[j] += -lr * (loss_func * (-user_like + reg * V[j]))
            # 更新偏置项
            biasV[i] += -lr * (loss_func + reg * biasV[i])
            biasV[j] += -lr * (-loss_func + reg * biasV[j])
        predict = user_like @ V.T  # 将训练完成的矩阵內积
        return list(predict)

    def load_data(self,path):
        f = open(path, 'r', encoding='utf-8')
        contents = f.readlines()
        user_items_dict = defaultdict(set)
        for content in contents:
            all = content.strip('\n').split('\t')
            intLine = list(map(int,all))
            user_items_dict[intLine[0]].add(intLine[1])
        return user_items_dict

    def main(self):
        user_items_dict = self.load_data('test/movie_rating_final_test_kg')
        # 测试train2
        user_scoreList_dict = self.train2(user_items_dict)
        # print(len(user_scoreList_dict))
        for userr in range(len(user_scoreList_dict)):
            print(user_scoreList_dict[userr])
            scores = Counter({idx: val for idx, val in enumerate(user_scoreList_dict[userr]) if idx not in user_items_dict[userr+182011]})
            topk = scores.most_common(5)
            print(f'{userr}_top5:{topk}')

        # 测试train3
        # new_scores = self.train3(182011,user_items_dict,3)
        # print(new_scores)

        # 测试train4
        # renew_scores = self.train4(182011, user_items_dict, [2,3])
        # print(renew_scores)

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()