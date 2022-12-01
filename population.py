import numpy as np
import hashlib
import copy

hiddenState_num = 5
op_num = 2
block_num = 3
op_category = 10


def parse(genotype, num):
    count = 0

    indices_list = []

    op_names_list = []
    for i in range(num):
        indices = []
        op_names = []
        for hidden in genotype[i]:
            count += 1
            for i in hidden[0]:
                indices.append(i)
            for j in hidden[2]:
                op_names.append(j)
        indices_list.append(indices)
        op_names_list.append(op_names)

    return indices_list, op_names_list, indices, op_names


class Individual(object):
    def __init__(self, indi_no):
        self.acc = -1.0
        self.id = indi_no
        self.params = 0
        self.units = []
        self.hidden_unit_num = hiddenState_num
        self.concat_list = [0.8, 0.8, 0.8, 0.8, 0.8]
        self.cell = []
        self.op_num = op_num
        self.op_category = 10
        self.init_age = 1
        self.block_num = block_num
        self.genotype = self.init_gene3()

        self.each_block_num = []
        for item in self.genotype:
            self.each_block_num.append(len(item))

    @property
    def indices(self):
        self._indices, _, _, _ = parse(self.genotype, self.block_num)
        return self._indices

    @property
    def op_names(self):
        _, self._op_name, _, _ = parse(self.genotype, self.block_num)
        return self._op_name

    @property
    def age(self):
        self.init_age += 1
        return self.init_age

    def reset_acc(self, Reset_Acc=-1.0):
        self.acc = Reset_Acc
        self.init_age = 1

    def init_gene3(self):
        genotype = []
        for i in range(self.block_num):
            genotype.append(self.init_gene())
        return genotype

    def init_gene(self):
        init_gene = []
        for i in range(self.hidden_unit_num):
            indices = np.random.randint(i + 1, size=2).tolist()

            op_name = np.random.randint(self.op_category, size=self.op_num).tolist()
            init_gene.append([indices, int(self.concat_list[i] // 0.5), op_name])
        return init_gene

    def __str__(self):
        _str = ['indi:%s' % self.id, 'Acc:%.5f' % self.acc, 'Genotype:[predecessor,concat,operation]',
                str(self.genotype), 'age:' + str(self.init_age)]

        return '\n'.join(_str)

    def uuid(self):
        _str = []
        for i in range(self.block_num):
            indices = self.indices[i]
            op_names = self.op_names[i]
            _str.append('block_{}_genotype'.format(i))
            for i in range(self.hidden_unit_num):
                h1 = indices[2 * i]
                h2 = indices[2 * i + 1]
                op1 = op_names[2 * i]
                op2 = op_names[2 * i + 1]

                _str.append('hidden_state_{}'.format(i))
                if h1 == h2:
                    if op1 <= op2:
                        _str.append('indices:[{},{}]'.format(h1, h2))
                        _str.append('ops:[{},{}]'.format(op1, op2))
                    else:
                        _str.append('indices:[{},{}]'.format(h1, h2))
                        _str.append('ops:[{},{}]'.format(op2, op1))
                elif h1 < h2:
                    _str.append('indices:[{},{}]'.format(h1, h2))
                    _str.append('ops:[{},{}]'.format(op1, op2))
                elif h1 > h2:
                    _str.append('indices:[{},{}]'.format(h2, h1))
                    _str.append('ops:[{},{}]'.format(op2, op1))
        _final_str_ = ','.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_


class Population(object):
    def __init__(self, pop_size, gen_no):
        self.gen_no = gen_no
        self.number_id = 0
        self.pop_size = pop_size
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(indi_no)
            indi.init_gene()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)

    def get_old(self):
        for i in range(len(self.individuals)):
            self.individuals[i].age


def get_arch_gene(indi):
    arch_gene = ''
    for i in range(indi.block_num):
        arch_gene += get_block_gene(indi, i)

    return arch_gene


def get_block_gene(indi, index):
    type_list = []
    for indices in indi.indices[index]:
        type_list.append(str(indices))
    for op in indi.op_names[index]:
        type_list.append(str(op))

    block_gene = ''.join(type_list)
    return block_gene


def get_hd(individual1, individual2):
    assert len(individual1) == len(individual2)

    distance = 0
    for item_x, item_y in zip(individual1, individual2):
        if item_x != item_y:
            distance += 1

    return distance


def get_best_indi(pops_individuals):
    fitness_list = []
    for i in range(len(pops_individuals)):
        fitness_list.append(pops_individuals[i].acc)
    max_index = np.argmax(fitness_list)
    best_indi = pops_individuals[max_index]
    return best_indi


def get_spd_i(indi, best_indi):
    spd_i = get_hd(get_arch_gene(indi), get_arch_gene(best_indi))
    str_len1 = len(get_arch_gene(indi))
    str_len2 = len(get_arch_gene(best_indi))
    spd_i = spd_i / (max(str_len1, str_len2) - 2)

    return spd_i
