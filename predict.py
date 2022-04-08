import torch
import json
from gcn_kgat_attention import *
import numpy as np
import random
from collections import defaultdict
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
class BPR:
    embedding_path = 'data/embedding.vec.json'
    embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
    item_count = embeddings.shape[0]    # self.item_count = 182011
    emb_size = embeddings.shape[1]
    output = GraphEncoder(item_count, emb_size, user_num=6040 * 0.8, embeddings=embeddings, max_seq_length=10, max_node=20,
                          hiddim=50,
                          layers=2, cash_fn=None, fix_emb=False).to(device)

    def train(self,user_items_dict):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        # for i in range(len(user_items_dict)):
        for user in range(2):
            # 得到用户偏好嵌入矩阵
            userId = random.randint(182011, 182011 + len(user_items_dict))  # 随机获取一个用户
            if userId not in user_items_dict.keys():
                continue
            items = list(user_items_dict[userId])
            for i in range(0, len(items), 10):
                seq = np.array([items[i:i + 10]])
                user_like = self.output(seq, [userId])
                user_like = user_like.cuda().data.cpu()
            # 随机设定的项目矩阵
            V = np.random.rand(self.item_count, 50) * 0.01
            biasV = np.random.rand(self.item_count) * 0.01
            # 从用户的U-I中随机选取1个Item
            i = random.sample(user_items_dict[userId], 1)[0]
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count)
            while j in user_items_dict[userId]:
                j = random.randint(1, self.item_count)
            # BPR
            r_ui = np.dot(user_like, self.embeddings[i].T) + biasV[i]
            r_uj = np.dot(user_like, self.embeddings[j].T) + biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            user_like += -lr * (loss_func * (self.embeddings[i] - self.embeddings[j]) + reg * user_like)
            self.embeddings[i] += -lr * (loss_func * user_like + reg * self.embeddings[i])
            self.embeddings[j] += -lr * (loss_func * (-user_like + reg * self.embeddings[j]))
            # 更新偏置项
            biasV[i] += -lr * (loss_func + reg * biasV[i])
            biasV[j] += -lr * (-loss_func + reg * biasV[j])
            return user_like,V
            # print("user_like,V:", user_like, V)

    def train2(self, user_items_dict):

        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1  #训练次数
        user_scoreList_dict = defaultdict(list)
        for i in range(train_count):
            for userId in user_items_dict.keys():
                items = list(user_items_dict[userId])
                for i in range(0, len(items), 10):
                    seq = np.array([items[i:i + 10]])
                    user_like = self.output(seq, [userId]) *0.01
                    user_like = user_like.cuda().data.cpu()
                # 随机设定的项目矩阵
                V = self.embeddings * 0.01
                biasV = np.random.rand(self.item_count) * 0.01
                # 从用户的U-I中随机选取1个Item
                i = random.sample(user_items_dict[userId], 1)[0]
                # 随机选取一个用户u没有评分的项目
                j = random.randint(1, self.item_count)
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
                user_scoreList_dict[userId]=list(predict.cpu().numpy())
        return user_scoreList_dict

    def train3(self,userId,user_items_dict,ignoredItem):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1  # 训练次数
        for i in range(train_count):
            items = list(user_items_dict[userId])
            items.remove(ignoredItem)
            for i in range(0, len(items), 10):
                seq = np.array([items[i:i + 10]])
                user_like = self.output(seq, [userId]) * 0.01
                user_like = user_like.cuda().data.cpu()
            # 随机设定的项目矩阵
            V = self.embeddings * 0.01
            biasV = np.random.rand(self.item_count) * 0.01
            # 从用户的U-I中随机选取1个Item
            i = random.sample(user_items_dict[userId], 1)[0]
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
        return  list(predict.cpu().numpy())

    def train4(self, userId, user_items_dict, ignoredItemList):
        lr = 0.01  # 步长α
        reg = 0.01  # 参数λ
        train_count = 1  # 训练次数
        for i in range(train_count):
            items = list(user_items_dict[userId])
            for i in ignoredItemList:
                items.remove(i)
            for i in range(0, len(items), 10):
                seq = np.array([items[i:i + 10]])
                user_like = self.output(seq, [userId]) * 0.01
                user_like = user_like.cuda().data.cpu()
            # 随机设定的项目矩阵
            V = self.embeddings * 0.01
            biasV = np.random.rand(self.item_count) * 0.01
            # 从用户的U-I中随机选取1个Item
            i = random.sample(user_items_dict[userId], 1)[0]
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count)
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
        return list(predict.cpu().numpy())

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
        train_count = 3  # 训练次数1000
        user_items_dict = self.load_data('data/train_test')
        user_scoreList_dict = self.train2(user_items_dict)
        print(len(user_scoreList_dict))
        # user_scoreList_dict=defaultdict(list)
        # for i in range(train_count):
        #     user_like,V = self.train(user_items_dict)
        #     # print("user_like,V",user_like,V)
        #     predict = user_like @ V.T   # 将训练完成的矩阵內积
        #     print("scoreList:", predict)
if __name__ == '__main__':
    bpr = BPR()
    bpr.main()