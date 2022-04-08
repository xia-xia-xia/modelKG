from pathlib import Path
from time import time
import numpy as np
import pandas as pd
from predict import *
from ast import literal_eval
def retrain(ks,bpr,user_items_dict):
	"""
	对于没有反事实集得模型进行再训练，验证是否 rec* 取代 rec
	retrain models without counterfactual sets for given values of k.
	Trained models are saved to user's home directory
	"""
	inputs = []
	input_files = [f"result/kgcsir_{k}.csv" for k in ks]
	for file in input_files:
		inputs.append(pd.read_csv(file))
	inputs = pd.concat(inputs, ignore_index=True)  #对index重新安排, 为False的时候会保留之前的index
	print(inputs)

	for row in inputs.itertuples():
		idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = read_row_from_result_file(row)
		if counterfactual is None:
			continue
		# 再训练不包括反事实集的模型
		print('begin retraining', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
		begin = time()
		new_predict_scoreList = bpr.train4(user_id, user_items_dict,counterfactual)
		print(f"done retraining {time() - begin}")
		# print("new_predict_scoreList:", len(new_predict_scoreList))
		return new_predict_scoreList

def read_row_from_result_file(row):
    """
    read a row from the result file
    return: if the counterfactual set is None then return None, else:
        idx: the id of the instance
        user_id: id of user
        item_id: top1 recommendation item
        topk: top k recommendations
        counterfactual: counterfactual set
        predicted_scores: predicted scores of the original top k items
        replacement: the predicted replacement item
    """
    idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:7]
    if not isinstance(counterfactual, str):
        print('skip', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
        return None, None, None, None, None, None, None
    topk = literal_eval(topk)
    counterfactual = literal_eval(counterfactual)
    if isinstance(predicted_scores, str):
        predicted_scores = literal_eval(predicted_scores)
    else:
        predicted_scores = None
    print('begin idx', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
    return idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement

if __name__ == "__main__":
	#retrain([5, 10, 20])
	ks = [5]
	bpr = BPR()
	user_items_dict = bpr.load_data('data/train_test')
	user_scoreList_dict = bpr.train2(user_items_dict)
	retrain(ks,bpr,user_items_dict)