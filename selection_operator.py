import numpy as np


class Selection(object):
    @staticmethod
    def roulette_selection(a_, k):
        a_ = np.asarray(a_)
        idx = np.argsort(a_)
        idx = idx[::-1]
        sort_a = a_[idx]
        sum_a = np.sum(a_).astype(np.float)
        selected_index = []
        for _ in range(k):
            u = np.random.rand() * sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break

        return selected_index
