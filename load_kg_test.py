import pandas as pd
kg_path = 'data/movie_kg.txt'

def load_kg(filename):
    kg_data = pd.read_csv(filename, sep='\t', names=['h', 't', 'r'], engine='python')
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
    print("n_entities",n_entities)

user_path = 'data/movie_ratings.txt'
writer = open('test/movie_train1208', 'w', encoding='utf-8')
with open(user_path) as f:
    for l in f.readlines():
        l = l.strip('\n').split('\t')
        user_bound = 6040*0.2

        uid = int(l[0])
        item = int(l[1])
        if (uid == user_bound):
            break
        writer.write('%d\t%d\t1\n' % (uid+182011, item))
    print("ending")
if __name__ == '__main__':
    load_kg(kg_path)





