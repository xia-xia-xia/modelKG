import pandas as pd
user_path = 'data/ratings.txt'

def load_train_data(path):
    # f = open(path, 'r', encoding='utf-8')
    # contents = f.readlines()
    # user_items_dict = defaultdict(set)
    # for content in contents:
    #     all = content.strip('\n').split('\t')
    #     intLine = list(map(int,all))
    #     user_items_dict[intLine[0]].add(intLine[1])
    # return user_items_dict

    writer = open('data/rating_final_trainData', 'w', encoding='utf-8')
    user_rating_dict = defaltdict(set)
    with open(user_path) as f:
        for l in f.readlines():
            l = l.strip('\n').split('\t')
            uid = int(l[0])
            item = int(l[1])
            rate = int(l[2])
            user_rating_dict[uid].add[rate]
            while (len(user_rating_dict[uid][rate]>3)<20):
                continue
            writer.write('%d\t%d\t1\n' % (uid+182011, item))
        print("ending")
if __name__ == '__main__':
    load_kg(user_path)





