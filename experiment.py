import numpy as np
import pandas as pd
from ast import literal_eval

def evaluate_files(ks):
    input_files = [f"kgcsir_{k}.csv" for k in ks]
    for file in input_files:
        print(file)
        data = pd.read_csv(file)
        swap = 0
        set_size = 0

        for id, row in data.iterrows():
            user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
            if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
                continue
            topk = literal_eval(topk) #使用literal_eval，会自动的检查表达式安全性和合法性，如果有问题就会直接抛出异常
            counterfactual = literal_eval(counterfactual)
            assert item_id == topk[0]
            actual_scores = literal_eval(row['actual_scores_avg'])
            replacement_rank = topk.index(replacement)
            if actual_scores[replacement_rank] > actual_scores[0]:
                swap += 1   #加1说明replacement替换了rec
                set_size += len(counterfactual)
        print('data', set_size,data.shape[0])
        print('swap', swap, swap / data.shape[0])  #实际替换百分比（CF百分比）假设有5个用户，3个用户替换正确，则为0.6
        print('size', set_size / swap)  #反事实集的平均大小（CF集大小） 替换正确的用户的反事实集总和/替换正确的用户个数

def main():
    """
    run the full experiment for an algorithm
    """
    #ks = [5, 10, 20]
    ks = [5]
    # generate_cf(ks)
    # retrain(ks)
    # get_new_scores(ks)
    evaluate_files(ks)

if __name__ == "__main__":
    main()
