import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict

def evaluate_files(ks):
    input_files = [f"result/kgcn604_{k}.csv" for k in ks]
    user_items_dict = load_data('test/movie_train604')
    for file in input_files:
        print(file)
        data = pd.read_csv(file)
        swap = 0
        set_size = 0
        value=0
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
                value += 1 - len(counterfactual)/len(user_items_dict[user_id])
        print('data', set_size,data.shape[0])
        print('fidelity', swap, swap / data.shape[0])  #实际替换百分比（CF百分比）假设有5个用户，3个用户替换正确，则为0.6
        print('size', set_size / swap)  #反事实集的平均大小（CF集大小） 替换正确的用户的反事实集总和/替换正确的用户个数
        print('sparsity', value / swap)  #反事实集的平均大小（CF集大小） 替换正确的用户的反事实集总和/替换正确的用户个数

def main():
    """
    run the full experiment for an algorithm
    """
    #ks = [5, 10, 20]
    ks = [5]
    # generate_cf(ks)
    # get_new_scores(ks)
    evaluate_files(ks)

def load_data(path):
    f = open(path, 'r', encoding='utf-8')
    contents = f.readlines()
    user_items_dict = defaultdict(set)
    for content in contents:
        all = content.strip('\n').split('\t')
        intLine = list(map(int,all))
        user_items_dict[intLine[0]].add(intLine[1])
    return user_items_dict

if __name__ == "__main__":
    main()
