import pandas as pd
kg_path = 'kg.txt'

def load_kg(filename):
    kg_data = pd.read_csv(filename, sep='\t', names=['h', 't', 'r'], engine='python')
    #kg_data = kg_data.drop_duplicates()  # 去除重复项
    #print(kg_data)
    print(max(kg_data['h']))
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
    print("n_entities",n_entities)

load_kg(kg_path)
user_path = 'ratings.txt'
writer = open('rating_final_kg', 'w', encoding='utf-8')
with open(user_path) as f:
    for l in f.readlines():
        l = l.strip('\n').split('\t')
        user_bound = 6040*0.8

        uid = int(l[0])
        item = int(l[1])
        if (uid == user_bound):
            break
        writer.write('%d\t%d\t1\n' % (uid+182011, item))
    print("ending")






