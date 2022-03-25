from pathlib import Path
from time import time
import numpy as np
import pandas as pd

def retrain(ks):
	"""
	对于没有反事实集得模型进行再训练，验证是否 rec* 取代 rec
	retrain models without counterfactual sets for given values of k.
	Trained models are saved to user's home directory
	"""
	inputs = []
	input_files = [f"kgcsir_{k}.csv" for k in ks]
	for file in input_files:
		inputs.append(pd.read_csv(file))
	inputs = pd.concat(inputs, ignore_index=True)  #对index重新安排, 为False的时候会保留之前的index
	print(inputs)

	home_dir = str(Path.home()) + '/pretrain-rcf-counterfactual'
	np.random.seed(1802)
	seeds = np.random.randint(1000, 10000, 5)
	# seeds = np.random.randint(1000, 10000, 2)
	seeds[0] = 2512

	for row in inputs.itertuples():
		idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = read_row_from_result_file(row)
		if counterfactual is None:
			continue
		# 再训练不包括反事实集的模型
		data = Dataset(ignored_user=user_id, ignored_items=counterfactual)  #dataset.py  h30
		pretrain = -1   #-1: save the 'model to pretrain file'
        hidden_factor = 64
		for i, seed in enumerate(seeds):
			model = get_new_RCF_model(data, args, save_file=f'{home_dir}/{counterfactual2path(user_id, counterfactual)}/{seed}/' + f'ml1M_{hidden_factor}')
			print('begin retraining', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, seed)
			begin = time()
			model.train(data, args, seed=seed)  #rcf h666
			print(f"done retraining {time() - begin}")

def counterfactual2path(user, counterfactual_set):
    """
    find a directory name to store the retrained model for a user-explanation pair
    :param user: id of the user
    :param counterfactual_set: the counterfactual explanation
    :return: a directory name
    """
    res = f'{user}-{"-".join(str(x) for x in sorted(counterfactual_set))}'
    if len(res) < 255:
        return res
    return hashlib.sha224(res.encode()).hexdigest()

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

def get_new_RCF_model(data, args, save_file):
    """
	get a new RCF model with all default params
	Args:
		data: the dataset used for training, see dataset.py
		args: extra arguments for the model
		save_file: the path to save the new model
	Returns:  the model
	"""
    activation_function = gelu
    args = parse_args()
    if args.pretrain == 1:   # 1: initialize from pretrain
        args.pretrain = 0    # 0: randomly initialize
    from rcf import MF
    model = MF(data.num_users, data.num_items, data.num_genres, data.num_directors, data.num_actors,
               data.train_data.shape[0], data.rel_data.shape[0], args.pretrain,
               args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose,
               eval(args.layers), activation_function, eval(args.keep_prob), save_file, args.attention_size, args.reg_t)
    return model

def train(self, data, args, seed):  # fit a dataset
    np.random.seed(seed)
    for epoch in range(self.epoch):
        train_data = data.train_data.sample(frac=1)
        train_data = train_data.reset_index(drop=True)
        rel_data = data.rel_data.sample(frac=1)
        rel_data = rel_data.reset_index(drop=True)
        num_iter = (train_data.shape[0] + self.batch_size - 1) // self.batch_size
        rel_batch_size = (rel_data.shape[0] + num_iter - 1) // num_iter
        total_loss = 0
        i2 = 0
        for i in range(0, train_data.shape[0], self.batch_size):
            j = min(i + self.batch_size, train_data.shape[0])
            batch_xs = self.prepare_batch(train_data.iloc[i:j], data, args)
            j2 = min(i2 + rel_batch_size, rel_data.shape[0])
            batch_xs.update(self.prepare_rel_data_batch(rel_data.iloc[i2:j2]))
            i2 += rel_batch_size
            loss = self.partial_fit(batch_xs)
            total_loss = total_loss + loss
        attention_type = self.get_attention_type_scalar()
        avearge = np.mean(attention_type, axis=0)
        print("the total loss in %d th iteration is: %f, the attentions are %.4f, %.4f, %.4f, %.4f" % (
            epoch, total_loss, avearge[0], avearge[1], avearge[2], avearge[3]))
        # self.evaluate(data, args)
    if self.pretrain_flag < 0:
        print("Save model to file as pretrain.")
        self.print_weights()
        self.saver.save(self.sess, self.save_file)

if __name__ == "__main__":
	#retrain([5, 10, 20])
	retrain([5])