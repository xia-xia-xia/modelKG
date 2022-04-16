import pandas as pd
kg_path = 'data/book_kg.txt'

def load_kg(filename):
    kg_data = pd.read_csv(filename, sep='\t', names=['h', 'r', 't'], engine='python')
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
    print("n_entities",n_entities)

user_path = 'data/book_ratings.txt'

# writer = open('data/book_rating_kg', 'w', encoding='utf-8')
# with open(user_path) as f:
#     user_bound = 17860 * 0.05
#     for l in f.readlines():
#         l = l.strip('\n').split('\t')
#         uid = int(l[0])
#         item = int(l[1])
#         if (uid == user_bound):
#             break
#         writer.write('%d\t%d\t1\n' % (uid+77903, item))
#     print("ending")

writer = open('data/book_rating_kg_final', 'w', encoding='utf-8')
with open(user_path) as f:
    user_bound = 17860 * 0.1
    tmp_uid=0
    tmp_dict =[]
    index=77903
    for l in f.readlines():
        l = l.strip('\n').split('\t')
        uid = int(l[0])
        item = int(l[1])
        if (uid == user_bound):
            break

        if(tmp_uid==uid):
            tmp_dict.append(item)
        else:
            if(len(tmp_dict)>=15):
                for i, itemId in enumerate(tmp_dict):
                    writer.write('%d\t%d\t1\n' % (index, itemId))
                index=index+1
            tmp_uid = uid
            tmp_dict = []
            tmp_dict.append(item)
    print("ending")
if __name__ == '__main__':
    load_kg(kg_path)





