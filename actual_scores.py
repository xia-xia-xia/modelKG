import numpy as np
from kgcn_bpr import *
# from no_kg_bpr import *
# from train_bpr import *
# from kg_bpr import *
from ast import literal_eval
import pandas as pd

def get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement):
    """
    get the new scores of top-k items
    Returns: a 2d array where each row is the scores of top-k items in one retrain.
    """
    times = 5
    res = np.zeros((times, len(topk)))
    for i in range(times):  #增加一列actual_scores 0
        print('begin retraining', idx, user_id, item_id, topk, counterfactual, replacement)
        new_predict_scoreList = bpr.train4(user_id, user_items_dict, counterfactual)
        print('done retraining')
        res[i] = [new_predict_scoreList[item] for item in topk]
        # scores = retrain(ks, bpr, user_items_dict)
        # res[i] = [scores[item] for item in topk]
    return res

def get_new_scores_main(input_files):
    """
    get new scores after retrained for the given input_files
     input_files: files containing the counterfactual sets
    """
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

            scores = get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
            if scores is None:
                print('bad scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
                continue
            assert len(scores) == 5
            # 由1个seeds产生的再训练模型得到新分数放到accent_5.csv
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
	input_files = [f"result/kgcn604_{k}.csv" for k in ks]
	get_new_scores_main(input_files)

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
if __name__ == "__main__":
    set_global_seeds(28)
    ks = [5]
    bpr = BPR()
    user_items_dict = bpr.load_data('test/movie_train604')
    get_new_scores(ks)