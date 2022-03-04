import torch
import json
import random
import numpy as np
from tqdm import tqdm
from gcn_kgat_attention import *
import time
import os
import itertools
from ddpg_env import Env
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
class Recommender(object):
    def __init__(self, dataset,max_rating, min_rating, boundary_rating, alpha, episode_length):
        set_global_seeds(28)
        # Env
        self.dataset = dataset
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.boundary_rating = boundary_rating
        self.alpha = alpha
        # self.episode_length = episode_length
        # self.env = Env(episode_length=self.episode_length, alpha=self.alpha, boundary_rating=self.boundary_rating,
        #                max_rating=self.max_rating, min_rating=self.min_rating, ratingfile=self.dataset)
        # self.user_num, self.item_num, self.rela_num = self.env.get_init_data()
        self.user_num=6040
        self.item_num=656461
        self.rela_num=1241995
        self.boundary_userid = int(self.user_num * 0.8)
        # train
        self.max_training_step = 100
        self.log_step = 1
        self.target_update_step = 100
        self.sample_times = 100
        self.update_times = 100
        self.batch_size = 10
        self.learning_rate = 5e-4
        self.l2_norm = 1e-6
        self.topk = 300
        embedding_path = 'embedding.vec.json'
        embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
        entity = embeddings.shape[0]
        emb_size = embeddings.shape[1]
        user_emb = nn.Parameter(torch.Tensor(6040, emb_size))
        entity_user_emb = torch.cat([embeddings, user_emb], dim=0)
        # hot_items_path = '../Data/run_time/movielens_pop1000'
        # if os.path.exists(hot_items_path):
        #     self.hot_items = utils.pickle_load(hot_items_path).tolist()
        # else:
        #     utils.popular_in_train_user(self.dataset, self.topk, self.boundary_rating)
        #     self.hot_items = utils.pickle_load(
        #         '../Data/pre_em/' + self.dataset + '_pop%d' % self.topk).tolist()
        # self.result_file_path = '../Data/result/' + time.strftime(
        #     '%Y%m%d%H%M%S') + '_' + self.dataset + '_ddpg%f' % self.alpha
        # self.candi_dict = utils.pickle_load('../Data/neighbors.pkl')
        #NET
        self.gcn = GraphEncoder(entity, emb_size,user_num =self.boundary_userid,embeddings=embeddings, max_seq_length=10, max_node=20, hiddim=50,
                                layers=2,cash_fn=None, fix_emb=False).to(device)
        self.storage = []
        self.tmp = []
        for i in range(self.item_num):
            self.tmp.append(i)
        tmp = torch.LongTensor(self.tmp)

    # def candidate(self, obs, mask):
    #     tmp = []
    #     candict = []
    #     tmp += self.hot_items
    #     for s in obs:
    #         if s in self.candi_dict:
    #             candict += self.candi_dict[s]
    #             tmp += self.candi_dict[s]
    #     tmp = set(tmp) - set(mask)
    #     candi = random.sample(tmp, self.candi_num)
    #     return candi
    def train(self):
        for itr in tqdm(range(self.sample_times), desc='sampling'):
            cumul_reward, done = 0, False
            user_id = random.randint(0, self.boundary_userid)
            cur_state, new_state = self.env.reset(user_id)
            # cur_state, new_state = entity_user_emb(user_id)
            mask = []
            while not done:  # 每一个人每一次到32截止推荐

                if len(cur_state) == 0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.ddpg.choose_action(list(cur_state), candi)
                new_state, r, done,positive = self.env.step(action_chosen)
                mask.append(action_chosen)
                candi = self.candidate(new_state,mask)
                if len(cur_state) != 0:
                    self.ddpg.memory.push(cur_state, action_chosen, r, new_state)
                if r > positive:
                    cur_state.append(action_chosen)
                else:
                    cur_state = cur_state

        for itr in tqdm(range(self.update_times), desc='updating'):
            self.ddpg.learn()

    def evaluate(self):
        ave_reward = []
        tp_list = []
        for itr in tqdm(range(self.user_num), desc='evaluate'):
            cumul_reward, done = 0, False
            cur_state, new_state = self.env.reset(itr)
            step = 0
            mask = []
            while not done:
                cur_candi = self.candidate(new_state, mask)
                if len(cur_state) == 0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.ddpg.choose_action(cur_state, cur_candi)
                new_state, r, done,positive = self.env.step(action_chosen)
                if r > positive:
                    cur_state.append(action_chosen)
                else:
                    cur_state = cur_state
                cumul_reward += r
                step += 1
                mask.append(action_chosen)
            ave = float(cumul_reward) / float(step)
            tp = float(len(cur_state))
            ave_reward.append(ave)
            tp_list.append(tp)

        train_ave_reward = np.mean(np.array(ave_reward[:self.boundary_userid]))
        test_ave_reward = np.mean(np.array(ave_reward[self.boundary_userid:]))

        precision = np.array(tp_list) / self.episode_length
        recall = np.array(tp_list) / (self.rela_num + 1e-20)
        train_ave_precision = np.mean(precision[:self.boundary_userid])
        train_ave_recall = np.mean(recall[:self.boundary_userid])
        test_ave_precision = np.mean(precision[self.boundary_userid:self.user_num])
        test_ave_recall = np.mean(recall[self.boundary_userid:self.user_num])

        self.storage.append(
            [train_ave_reward, train_ave_precision, train_ave_recall, test_ave_reward, test_ave_precision,
             test_ave_recall])
        print('\ttrain average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f' % (
        train_ave_reward, self.episode_length, train_ave_precision, self.episode_length, train_ave_recall))
        print('\ttest  average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f' % (
        test_ave_reward, self.episode_length, test_ave_precision, self.episode_length, test_ave_recall))
        utils.pickle_save(self.storage, self.result_file_path)

    def run(self):
        for i in range(0, self.max_training_step):
            self.train()
            if i % self.log_step == 0:
                self.evaluate()
if __name__ == '__main__':
    rec = Recommender('movielens',max_rating=5,min_rating=0,boundary_rating=3.5,alpha=0.0,episode_length=10)
    rec.run()