import os
import pandas as pd
from time import time
import numpy as np
from cg_helper import *
from book_train_bpr import *
# from book_predict import *

def find_counterfactual_multiple_k(user_id,ks,user_items_dict,user_scoreList_dict,bpr1):
	beginUserId = 77903
	cur_scores=user_scoreList_dict[user_id-beginUserId]  # 分数列表
	visited = user_items_dict[user_id]  # 交互的pos_item项目(eg,评分>3)
	_, topk = get_topk(cur_scores, set(visited), ks[-1])
	recommended_item = topk[0][0]
	# recommended_item_score = topk[0][1]
	# 初始化用户历史交互项目对top k items的影响
	influences = np.zeros((ks[-1], len(visited)))
	# enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
	for i, itemId in enumerate(visited):
		predict_tmp = bpr1.train3(user_id, user_items_dict, itemId)
		for j in range(ks[-1]):
			# 老分数减新分数
			old_sorce=cur_scores[topk[j][0]]
			new_sorce=predict_tmp[topk[j][0]]
			influences[j][i] =old_sorce-new_sorce # 交互的items对于top k items分数差距的影响

	res = None  # 移除的项目集removed_items
	best_gap = 1e9  # score_gap
	best_repl = -1
	best_i = -1
	ret = []
	for i in range(1, ks[-1]):  # for each item in the original top k
		# 尝试替换rec，得到反事实集、替换项目与rec之间的分数差距
		tmp_res, tmp_gap = try_replace(topk[i][0], topk[0][1] - topk[i][1], influences[0] - influences[i])
		if tmp_res is not None and (res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
			res, best_repl, best_i, best_gap = tmp_res, topk[i][0], i, tmp_gap

		if i + 1 == ks[len(ret)]:
			if res is not None:
				predicted_scores = np.array([cur_scores[item] for item, _ in topk[:(i + 1)]])
				for item in res:
					predicted_scores -= influences[:(i + 1), item]
				assert predicted_scores[0] < predicted_scores[best_i]
				assert abs(predicted_scores[0] - predicted_scores[best_i] - best_gap) < 1e-6
				ret.append((set(list(visited)[idx] for idx in res), recommended_item, [item for item, _ in topk[:(i + 1)]],
							list(predicted_scores), best_repl))
			else:
				ret.append((None, recommended_item, [item for item, _ in topk[:(i + 1)]], None, -1))
	# print('counterfactual time', time() - begin)
	return ret

def init_all_results(ks):
    """
    init a list of results to store explanations produced by explanation algorithms
    :param ks: list of k values to considered
    :return: a list of dictionaries where each one stores the result of one k value
    """
    all_results = []
    for _ in ks:
        all_results.append(
            {
                'user': [],
                'item': [],
                'topk': [],
                'counterfactual': [],
                'predicted_scores': [],
                'replacement': []
            }
        )
    return all_results

def append_result(ks, all_results, user_id, res):
    """
    append res to all_results where res is the result of an explanation algorithm
    :param ks: list of k values considered
    :param all_results: a dataset of results
    :param user_id: id of user explained
    :param res: the result produced by the explanation algorithms
    """
    for j in range(len(ks)):
        all_results[j]['user'].append(user_id)
        counterfactual, rec, topk, predicted_scores, repl = res[j]
        all_results[j]['item'].append(rec)
        all_results[j]['topk'].append(topk)
        all_results[j]['counterfactual'].append(counterfactual)
        all_results[j]['predicted_scores'].append(predicted_scores)
        all_results[j]['replacement'].append(repl)
        print('k =', ks[j])
        if not counterfactual:
            print(f"Can't find counterfactual set for user {user_id}")
        else:
            print(f"Found a set of size {len(counterfactual)}: {counterfactual}")
            print("Old top k: ", topk)
            print("Replacement: ", repl, predicted_scores)

def generate_cf(ks):
	"""
	generate counterfactual explanations for multiple k values
	"""
	n_samples = len(user_items_dict)   #总用户数
	all_results = init_all_results(ks)
	for i, user_id in enumerate(user_items_dict):
		print(f'testing user {i+1}/{n_samples}: {user_id}')
		res = find_counterfactual_multiple_k(user_id, ks, user_items_dict, user_scoreList_dict, bpr1)
		append_result(ks, all_results, user_id, res)

	for j in range(len(ks)):
		df = pd.DataFrame(all_results[j])
		df.to_csv(f'result/all_book_{ks[j]}.csv', index=False)

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

if __name__ == "__main__":
	#generate_cf([5, 10, 20])
	set_global_seeds(28)
	bpr1 = BPR1()
	user_items_dict=bpr1.load_data('data/book_rating_kg')
	user_scoreList_dict = bpr1.train2(user_items_dict)
	generate_cf([2])


