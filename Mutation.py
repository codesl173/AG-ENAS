import numpy as np
from selection_operator import Selection
from collections import Counter
import math
from population import hiddenState_num, op_num, block_num, op_category

selection = Selection()

hidden_unit_num = hiddenState_num
h = hidden_unit_num
w = op_category
op_num = op_num
block_num = block_num
op_origin_op = (1 / w) * np.ones((h, w))
op_filename = 'Op_M'


class Mutation(object):
    def __init__(self, individuals, prob_):
        self.individuals = individuals
        self.prob = prob_

    def do_mutation(self):
        pass


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def do_op_mutation(indi):
    mutated_block_index = np.random.randint(0, indi.block_num)
    mutation_point = np.random.randint(0, h)
    mutation_num = np.random.randint(1, op_num + 1)
    print('the number of op selected is: ', mutation_num)
    op_matrix = read_op_matrix(mutated_block_index, )
    selected_index = selection.roulette_selection(a_=op_matrix[mutation_point, :], k=mutation_num)
    print('op index selected is: ', selected_index)
    for i in range(mutation_num):
        indi.genotype[mutated_block_index][mutation_point][2][i] = selected_index[i]

    print('after mutation', indi.genotype[mutated_block_index][mutation_point])
    indi.reset_acc()
    return indi


def do_indices_mutation(indi):
    mutated_block_index = np.random.randint(0, indi.block_num)
    mutation_point = np.random.randint(0, h)
    input_node = 1
    new_indices = np.random.randint(0, mutation_point + input_node, 2)

    indi.genotype[mutated_block_index][mutation_point][0][0], indi.genotype[mutated_block_index][mutation_point][0][1] = \
        new_indices[0], new_indices[1]
    indi.reset_acc()
    return indi


def uniform_initialize_op_matrix():
    op_m = np.ones((h, w)) * (1 / w)
    op_m = sigmoid(op_m)
    return op_m


def get_ucb(wk, nk, n, c):
    nk = nk if nk != 0 else 1
    score = wk / nk + c * math.sqrt(math.log(n/nk, math.e))
    return score


def get_ucb_opm(pops_individual, block_index, c):
    f_avg = 0
    pop_num = len(pops_individual)
    hidden_1_op, hidden_2_op, hidden_3_op, hidden_4_op, hidden_5_op = [], [], [], [], []
    hidden_1_wop, hidden_2_wop, hidden_3_wop, hidden_4_wop, hidden_5_wop = [], [], [], [], []
    for indi in pops_individual:
        f_avg += indi.acc / pop_num
        hidden_1_op += indi.op_names[block_index][0:2]
        hidden_2_op += indi.op_names[block_index][2:4]
        hidden_3_op += indi.op_names[block_index][4:6]
        hidden_4_op += indi.op_names[block_index][6:8]
        hidden_5_op += indi.op_names[block_index][8:10]
    for indi in pops_individual:
        if indi.acc > f_avg:
            hidden_1_wop += indi.op_names[block_index][0:2]
            hidden_2_wop += indi.op_names[block_index][2:4]
            hidden_3_wop += indi.op_names[block_index][4:6]
            hidden_4_wop += indi.op_names[block_index][6:8]
            hidden_5_wop += indi.op_names[block_index][8:10]

    hidden_1_op_prob = get_op_ucb_prob(get_op_dict(hidden_1_op), get_op_dict(hidden_1_wop), pop_num * 2, c)
    hidden_2_op_prob = get_op_ucb_prob(get_op_dict(hidden_2_op), get_op_dict(hidden_2_wop), pop_num * 2, c)
    hidden_3_op_prob = get_op_ucb_prob(get_op_dict(hidden_3_op), get_op_dict(hidden_3_wop), pop_num * 2, c)
    hidden_4_op_prob = get_op_ucb_prob(get_op_dict(hidden_4_op), get_op_dict(hidden_4_wop), pop_num * 2, c)
    hidden_5_op_prob = get_op_ucb_prob(get_op_dict(hidden_5_op), get_op_dict(hidden_5_wop), pop_num * 2, c)
    op_matrix = hidden_1_op_prob + hidden_2_op_prob + hidden_3_op_prob + hidden_4_op_prob + hidden_5_op_prob
    op_matrix = np.array(op_matrix)
    op_matrix.resize(h, w)
    op_matrix = sigmoid(op_matrix)

    return op_matrix


def get_op_dict(hidden_op):
    op_dict = dict(Counter(hidden_op))
    op_dict = sorted(op_dict.items(), key=lambda op_dict: op_dict[0], reverse=False)
    op_dict = dict(op_dict)
    return op_dict


def get_op_ucb_prob(all_dict, win_dict, n, c):
    score = []
    for i in range(w):
        if i in all_dict.keys():
            nk = all_dict[i]
        else:
            nk = 0
        if i in win_dict.keys():
            wk = win_dict[i]
        else:
            wk = 0
        score.append(get_ucb(wk, nk, n, c))
    prob_arr = []
    score_total = sum(score)
    for i in range(len(score)):
        prob_arr.append(score[i] / score_total)
    return prob_arr


def save_op_matrix(operation_matrix, block_index, filename=op_filename):
    filename = filename + '_' + str(block_index)
    np.save(filename, operation_matrix)


def read_op_matrix(block_index, filename=op_filename):
    store_name = filename + '_' + str(block_index) + '.npy'
    return np.load(store_name)


def get_prob_of_op(hidden_op):
    op_dict = dict(Counter(hidden_op))

    op_dict = sorted(op_dict.items(), key=lambda op_dict: op_dict[0], reverse=False)
    op_dict = dict(op_dict)

    op_total = sum(op_dict.values())
    op_prob = []
    for i in range(w):
        if i in op_dict.keys():
            op_prob += [op_dict[i] / op_total]
        else:
            op_prob += [0]
    return op_prob
