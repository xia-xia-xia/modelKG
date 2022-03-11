# rcf.py
# evaluate the results for an user context, return scorelist
def get_scores_per_user(self, user_id, data, args,
                        ignored_id=None):
    scorelist = []
    users = [user_id] * self.batch_size
    for j in range(0, self.num_items, self.batch_size):
        k = min(j + self.batch_size, self.num_items)
        feed_dict = self.prepare_feed_dict_batch(users[:(k - j)], range(j, k), data, args, ignored_user=user_id,
                                                 ignored_id=ignored_id)
        scores = self.sess.run(self.pos, feed_dict=feed_dict)
        scores = scores.reshape(k - j)
        scorelist = np.append(scorelist, scores)
    return scorelist

# src/helper.py
def get_topk(scores: list, visited: set, k: int):
    """
	given the scores, get top k recommendations
	Args:
		scores: list
		visited: list of interacted items
		k: number of items to return
	Returns:
		dict from item to score,
		top k items
	"""
    scores = Counter({idx: val for idx, val in enumerate(scores) if idx not in visited})
    topk = scores.most_common(k)
    return scores, topk

# rcf.py
    # 每移除一个用户历史交互项目，看对top k items的分数影响，得到分数差距
    def get_influence3(self, user_id, item_id, data, args):  # old params - new params
        print(f'get influence {user_id} {item_id}')
        # 从训练数据集中取user=user_id的数据,并重置索引index
        train_data = data.train_data[data.train_data['user'] == user_id].reset_index(drop=True)
        l = len(train_data['pos_item'])
        res = np.zeros(l)
        for i, row in train_data.iterrows():
            hessian = self.get_hessian(user_id, item_id, row['pos_item'], data, args, batch=32, verbose=0)
            inv_hessian = np.linalg.pinv(hessian)
            loss_grad = self.get_loss_grad_individual(user_id, item_id, row['pos_item'], row['neg_item'], data, args)
            params_infl = -np.matmul(loss_grad, inv_hessian) / train_data.shape[0]
            res[i] = self.get_score_influence(user_id, item_id, row['pos_item'], params_infl, data, args)
        return res

# accent_template.py
    def try_replace(repl, score_gap, gap_infl):
        """
        given a replacement item, try to swap the replacement and the recommendation
        Args:
            repl: the replacement item
            score_gap: the current score gap between repl and the recommendation
            gap_infl: a list of items and their influence on the score gap
        Returns: if possible, return the set of items that must be removed to swap and the new score gap
                else, None, 1e9
        """
        print(f'try replace', repl, score_gap)
        sorted_infl = np.argsort(-gap_infl)   #对分数差距的影响由大到小排序
        removed_items = set()
        for idx in sorted_infl:
            if gap_infl[idx] < 0:  # cannot reduce the gap any more
                break
            removed_items.add(idx)
            score_gap -= gap_infl[idx]
            if score_gap < 0:  # the replacement passed the predicted
                break
        if score_gap < 0:
            print(f'replace {repl}: {removed_items}')
            return removed_items, score_gap
        else:
            print(f'cannot replace {repl}')
            return None, 1e9