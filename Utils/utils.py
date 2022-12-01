import os
import logging
import numpy as np
from collections.abc import Iterable
import multiprocessing
import sys
import configparser

from torch.autograd import Variable
from torch import FloatTensor

from population import Population, Individual


def get_params(model, clever=True):
    if clever:
        return clever_format(count_parameters(model))
    else:
        return count_parameters(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clever_format(nums, format_="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format_ % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format_ % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format_ % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format_ % (num / 1e3) + "K")
        else:
            clever_nums.append(format_ % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).cuda()
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("BAE-CNN2")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def load_cache_data(cls):
        file_name = './populations/cache.txt'
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f' % (float(rs_[1]))
            f.close()
        return _map

    @classmethod
    def save_fitness_at_once(cls, indi):
        _map = cls.load_cache_data()
        _key, _str = indi.uuid()
        _acc = indi.acc
        if _key not in _map:
            Log.info('Add record into cache At once, id:%s, acc:%.5f' % (_key, _acc))
            f = open('./populations/cache.txt', 'a+')
            _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
            f.write(_str)
            f.close()
            _map[_key] = _acc

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                f = open('./populations/cache.txt', 'a+')
                _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = './populations/%s_%02d.txt' % (prefix, np.min(gen_no))
        params = StatusUpdateTool.get_init_params()
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('Genotype:'):
                        lines = next(f)
                        indi.genotype = eval(lines)
                    elif line.startswith('age:'):

                        indi.init_age = int(line[4:])
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            pop.individuals.append(indi)
        f.close()

        if gen_no == 0:
            after_file_path = './populations/after_%02d.txt' % (gen_no)
            if os.path.exists(after_file_path):
                fitness_map = {}
                f = open(after_file_path)
                count = 0
                for line in f:
                    if len(line.strip()) > 0:
                        line = line.strip().split('=')
                        count += 1
                        fitness_map[line[0]] = float(line[1])
                f.close()
                for indi in pop.individuals:
                    if indi.id in fitness_map:
                        indi.acc = fitness_map[indi.id]
        return pop

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        if not os.path.exists(r'./populations'):
            os.mkdir(r'./populations')
        file_name = './populations/begin_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_for_temp(cls, _str, gen_no):
        file_name = './populations/temp_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        file_name = './populations/crossover_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = './populations/mutation_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk('./populations'):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    id_list.append(int(file_name[6:8]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()


class StatusUpdateTool(object):
    epoch = 0

    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        config.read('global.ini')
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)

                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
        config.write(open('../global.ini', 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read('global.ini')
        config.set(section, key, value)
        config.write(open('../global.ini', 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('./global.ini')
        return config.get(section, key)

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def change_epoch(cls):
        section = 'network'
        key = 'epoch'
        cls.epoch = cls.get_epoch_size() + 5
        cls.__write_ini_file(section, key, str(cls.epoch))
        return cls.epoch

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False

    @classmethod
    def get_channel_limit(cls):
        rs = cls.__read_ini_file('network', 'C')

        return int(rs)

    @classmethod
    def get_N_limit(cls):
        rs = cls.__read_ini_file('network', 'N')

        return int(rs)

    @classmethod
    def get_input_channel(cls):
        rs = cls.__read_ini_file('network', 'input_channel')
        return int(rs)

    @classmethod
    def get_output_channel(cls):
        rs = cls.__read_ini_file('network', 'output_channel')
        channels = []
        for i in rs.split(','):
            channels.append(int(i))
        return channels

    @classmethod
    def get_num_class(cls):
        rs = cls.__read_ini_file('network', 'num_class')
        return int(rs)

    @classmethod
    def get_pop_size(cls):
        rs = cls.__read_ini_file('settings', 'pop_size')
        return int(rs)

    @classmethod
    def get_epoch_size(cls):
        rs = cls.__read_ini_file('network', 'epoch')
        return int(rs)

    @classmethod
    def get_individual_max_length(cls):
        rs = cls.__read_ini_file('network', 'max_length')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p

    @classmethod
    def get_batchsize(cls):
        rs = cls.__read_ini_file('network', 'batchsize')
        return int(rs)

    @classmethod
    def get_lr(cls):
        rs = cls.__read_ini_file('network', 'lr')
        return float(rs)

    @classmethod
    def get_init_params(cls):
        params = {'pop_size': cls.get_pop_size(), 'C': cls.get_channel_limit(), 'N': cls.get_N_limit(),
                  'num_class': cls.get_num_class(), 'lr': cls.get_lr(), 'batch_size': cls.get_batchsize(),
                  'epochs': cls.get_epoch_size()}

        return params

    @classmethod
    def get_mutation_probs_for_each(cls):
        rs = cls.__read_ini_file('settings', 'mutation_probs').split(',')
        assert len(rs) == 4
        mutation_prob_list = [float(i) for i in rs]
        return mutation_prob_list
