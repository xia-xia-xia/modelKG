from pathlib import Path

def prepare_new_scores(user_id, key, home_dir):
    """
    prepare to get new scores for a pretrained model
    :param user_id: id of user to be scored
    :param key: directory name where the pretrained models are stored
    :param home_dir: home directory where all pretrained models are stored
    :return: None if the pretrained model doesn't exist or the subfolders where the pretrained models are stored
    """
    # load model from disk
    if not Path(f'{home_dir}/{key}/').exists():
        print('missing', user_id, key)
        return None
    #sorted()函数对所有可迭代的对象进行排序操作。使用 os.scandir() 来遍历目录，is_dir() 函数检查指定的文件是否是目录
    subfolders = sorted([f.path for f in os.scandir(f'{home_dir}/{key}/') if f.is_dir()])
    if len(subfolders) != 5:  #5对应retrain_counterfactual.py中的seeds
        print('missing', user_id, key, len(subfolders))
        return None
    return subfolders

def get_pretrained_RCF_model(data, args, path):
    """
	load a pretrained RCF model from disk
	Args:
		data: the dataset used for training, see dataset.py
		args: extra arguments for the model
		path: the path that stores that pretrained model
	Returns:  the model
	"""
    activation_function = gelu
    save_file = '%s/%s_%d' % (path, 'ml1M', args.hidden_factor)
    args.pretrain = 1  # 1: initialize from pretrain
    from rcf import MF
    model = MF(data.num_users, data.num_items, data.num_genres, data.num_directors, data.num_actors,
               data.train_data.shape[0], data.rel_data.shape[0], args.pretrain,
               args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose,
               eval(args.layers), activation_function, eval(args.keep_prob), save_file, args.attention_size, args.reg_t)
    return model