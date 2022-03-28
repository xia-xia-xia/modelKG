from pathlib import Path
import numpy as np
from retrain_counterfactual import counterfactual2path
from help_getNewScores import *
from cg_helper import get_scores_per_user
from ast import literal_eval

def get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
	"""
	get scores of all items after retrained
	Args:
		idx: test number
		user_id: ID of user
		item_id: ID of item
		topk: the top-k items
		counterfactual: the counterfactual set
		predicted_scores: the predicted scores
		replacement: the replacement item
		item2scores: a dict for caching
		home_dir: the directory where trained models are stored
	Returns:
		a 2d array where each row is the scores of all items in one retrain.
	"""
	key = counterfactual2path(user_id, counterfactual)
	if key in item2scores:  # if cached
		return item2scores[key]

	subfolders = prepare_new_scores(user_id, key, home_dir)  #找到pretrained model的子文件夹
	if subfolders is None:
		return None

	data = Dataset(ignored_user=user_id, ignored_items=counterfactual)   #dataset.py  h30
	pretrain = -1   #-1: save the 'model to pretrain file'
	new_scores = np.zeros(shape=(5, data.num_items))   #5代表seeds个数
	# enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
	for i, path in enumerate(subfolders):
		model = get_pretrained_RCF_model(data, args, path)
		print('begin scoring', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, path)
		new_scores[i] = model.get_scores_per_user(user_id, data, args)
	item2scores[key] = new_scores
	return new_scores

def get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir,
                    get_scores):
    """
    get the new scores of top-k items
    Args:
        idx: test number
        user_id: ID of user
        item_id: ID of item
        topk: the top-k items
        counterfactual: the counterfactual set
        predicted_scores: the predicted scores
        replacement: the replacement item
        item2scores: a dict for caching
        home_dir: the home directory, where trained models are stored
    Returns: a 2d array where each row is the scores of top-k items in one retrain.
    """
    scores = get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores,
                        home_dir)
    if scores is None:
        return None

    res = np.zeros((5, len(topk)))
    for i in range(5):   #5个seeds产生,即actual_scores 0-4
        res[i] = [scores[i][item] for item in topk]
    return res

def get_new_scores_main(home_dir, input_files, get_scores):
    """
    get new scores after retrained for the given input_files
     home_dir: home directory where pretrained models are stored
     input_files: files containing the counterfactual sets
     get_scores: a method to get new scores
    """
    item2scores = dict()
    for file in input_files:
        print('begin file', file)
        inputs = pd.read_csv(file)
        for row in inputs.itertuples():
            idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:7]
            topk = literal_eval(topk)   #literal_eval自动的检查表达式安全性和合法性
            if not isinstance(counterfactual, str):
                print('skip', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
                continue
            counterfactual = literal_eval(counterfactual)
            if isinstance(predicted_scores, str):
                predicted_scores = literal_eval(predicted_scores)
            else:
                predicted_scores = None
            assert item_id == topk[0]
            print('begin idx', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)

            scores = get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement,
                                     item2scores, home_dir, get_scores)
            if scores is None:
                print('bad scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
                continue
            assert len(scores) == 5
            # 由5个seeds产生的再训练模型得到新分数放到accent_5.csv
            for i in range(5):
                inputs.at[idx, f'actual_scores_{i}'] = str(list(scores[i]))
            s = np.mean(scores, axis=0)
            inputs.at[idx, f'actual_scores_avg'] = str(list(s))
            print('avg new scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, s)
        inputs.to_csv(file, index=False)

def get_new_scores(ks):
	"""
	get new scores after retrained for the given values of k
	"""
	input_files = [f"kgcsir_{k}.csv" for k in ks]
	home_dir = str(Path.home()) + '/pretrain-rcf-counterfactual'
	get_new_scores_main(home_dir, input_files, get_scores)

if __name__ == "__main__":
	#get_new_scores([5, 10, 20])
	get_new_scores([5])