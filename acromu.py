from Utils.utils import Utils
import numpy as np
import random
import math
import copy
import crossover
import Mutation
from population import get_arch_gene, get_spd_i, get_best_indi


def get_spd(pops_individuals):
    best_indi = get_best_indi(pops_individuals)
    spd = 0
    for i in range(len(pops_individuals)):
        indi = pops_individuals[i]
        spd += get_spd_i(indi, best_indi)
    return spd / len(pops_individuals)


def get_sum_fitness(pops_individuals):
    sum_fitness = 0
    for i in range(len(pops_individuals)):
        indi = pops_individuals[i]
        sum_fitness += indi.acc
    return sum_fitness


def get_wi(indi, sum_fitness):
    wi = math.exp(indi.acc / sum_fitness)
    return wi


def get_hpd_i(spd_i, wi):
    hpd_i = (1 + spd_i) * wi
    return hpd_i


def get_hpd(pops_individuals):
    best_indi = get_best_indi(pops_individuals)
    sum_fitness = get_sum_fitness(pops_individuals)
    hpd = 0
    for i in range(len(pops_individuals)):
        indi = pops_individuals[i]
        hpd_i = get_spd_i(indi, best_indi)
        wi = get_wi(indi, sum_fitness)
        hpd += get_hpd_i(hpd_i, wi)
    return hpd / len(pops_individuals)


def get_r_by_spd(k1, k2, spd, spd_max):
    r = (spd / spd_max) * (k2 - k1) + k1
    return r


def get_pm_diversity(k, spd, spd_max):
    pm_diversity = ((spd_max - spd) / spd_max) * k
    return pm_diversity


def get_pm_fitness(k, f_max, f_min, f_average, f):
    if (f_max - f_average) < 0.02:
        pm = 0.7
    else:
        if f > f_average:
            pm = k * (f_max - f) / (f_max - f_average)
            if pm < 0.2:
                pm = 0.3
        else:
            pm = 0.5

    return pm


def get_pm(pm_diversity, pm_fitness):
    pm = max(pm_fitness, pm_diversity)
    return pm


def get_exploitation_and_exploration(pop_size, r):
    p = np.random.rand(pop_size)
    temp_list = [i for i in p if i < r]
    exploitation_size = (len(temp_list) // 2) * 2
    exploration_size = pop_size - exploitation_size
    return exploitation_size, exploration_size


class ACROMU(object):
    def __init__(self, k1, k2, k, _log, pops_individuals, _params=None):
        self.k1, self.k2, self.K = k1, k2, k
        self.pops_individuals = copy.deepcopy(pops_individuals)
        self.pop_size = len(self.pops_individuals)
        self.params = _params
        self.log = _log
        self.offspring = []
        self.fitness_max, self.fitness_min, self.fitness_average = 1, 0.8, 0.9
        self.SPD = get_spd(self.pops_individuals)
        self.HPD = get_hpd(self.pops_individuals)
        self.best_indi = get_best_indi(self.pops_individuals)
        self.v_list = []
        self.sum_fitness = get_sum_fitness(self.pops_individuals)
        for indi in self.pops_individuals:
            self.v_list.append(indi.acc)
        self.SPD_max = 1
        self.HPD_max = 2
        self.Pc = get_r_by_spd(self.k1, self.k2, self.SPD, self.SPD_max)
        self.T_size_max = self.pop_size / 6
        self.T_size = math.ceil(self.T_size_max * (self.HPD / self.HPD_max))
        self.Pm_diversity = get_pm_diversity(self.K, self.SPD, self.SPD_max)
        self.Exploitation_size, self.Exploration_size = get_exploitation_and_exploration(self.pop_size, self.Pc)
        self.log.info('Current Pc : %.2f' % self.Pc)
        self.log.info('Exploitation_size:%d, Exploration_size:%d' % (self.Exploitation_size, self.Exploration_size))

    def set_fitness(self):
        fitness_list = []
        for i in range(len(self.pops_individuals)):
            fitness_list.append(self.pops_individuals[i].acc)
        self.fitness_max = max(fitness_list)
        self.fitness_min = min(fitness_list)
        self.fitness_average = np.mean(fitness_list)
        self.log.info('Before ACROMU Fitness_max: {:.5f} Fitness_average: {:.5f} Fitness_min: {:.5f}'
                      .format(self.fitness_max, self.fitness_average, self.fitness_min))

    def process(self):
        self.set_fitness()
        exploration = Exploration(self.Exploration_size, self.pops_individuals, self.log, self.T_size, self.fitness_max,
                                  self.fitness_min, self.fitness_average, self.Pm_diversity)
        offspring1 = exploration.do_exploration()
        self.offspring = offspring1
        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])

        exploitation = Exploitation(self.Exploitation_size, self.pops_individuals, self.log, self.T_size)
        offspring2 = exploitation.do_exploitaion()
        self.offspring = offspring2
        Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])
        offspring = offspring1 + offspring2
        self.offspring = offspring
        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d' % (self.params['gen_no'], i)
            indi.id = indi_no

        return self.offspring

    def individuals_to_string(self):
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


class Exploitation(object):
    def __init__(self, exploitation_size, pops_individuals, _log, T_size):
        self.Exploitation_size = exploitation_size
        self.pops_individuals = copy.deepcopy(pops_individuals)
        self.log = _log
        self.T_size = T_size
        self.k = 2
        self.best_indi = get_best_indi(self.pops_individuals)
        self.v_list = []
        self.sum_fitness = get_sum_fitness(self.pops_individuals)
        for indi in self.pops_individuals:
            self.v_list.append(indi.acc)

    def tournament_selection(self, v_list, t_size, k):
        selected_index = []
        index_list = range(len(v_list))
        for i in range(k):
            tournament = [random.choice(index_list) for i in range(t_size)]
            fitnesses = [v_list[tournament[i]] for i in range(t_size)]
            selected_index.append(tournament[np.argmax(fitnesses)])
        while selected_index[0] == selected_index[1]:
            self.log.info('error,need fix')
            selected_index[0], selected_index[1] = self.tournament_selection(v_list, t_size, k)

            if selected_index[0] != selected_index[1]:
                self.log.info('error fix')
                self.log.info(str(selected_index))
                break
        else:
            self.log.info('no need fix')

        return selected_index[0], selected_index[1]

    def do_crossover(self, _stat_param):
        new_offspring_list = []

        for _ in range(self.Exploitation_size // 2):
            ind1, ind2 = self.tournament_selection(self.v_list, self.T_size, self.k)

            parent1, parent2 = copy.deepcopy(self.pops_individuals[ind1]), copy.deepcopy(self.pops_individuals[ind2])
            if get_arch_gene(parent1) != get_arch_gene(parent2):
                _stat_param['offspring_new'] += 2
            else:
                print('Two same indi')
                ind1, ind2 = self.tournament_selection(self.v_list, self.T_size, self.k)
                parent1, parent2 = copy.deepcopy(self.pops_individuals[ind1]), copy.deepcopy(self.pops_individuals[ind2])
            offspring1, offspring2 = crossover.do_1point_crossover(parent1, parent2, self.log)
            new_offspring_list.append(offspring1)
            new_offspring_list.append(offspring2)

        self.log.info('CROSSOVER-%d offspring are generated, new:%d, others:%d' % (
            len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list, _stat_param

    def do_exploitaion(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0, 'indices': 0, }

        offspring, _stat_param = self.do_crossover(_stat_param)
        offspring, _stat_param = self.little_mutation(offspring, _stat_param)
        self.log.info('After Exploitation   [Crossover: %d ,Change indices :%2d]' % (
            _stat_param['offspring_new'],
            _stat_param['indices']))

        return offspring

    @staticmethod
    def little_mutation(offspring, _stat_param):
        new_offspring_list = []
        for indi in offspring:
            p_ = random.random()
            pm = 0.3
            if p_ < pm:
                indi = Mutation.do_indices_mutation(indi)
                _stat_param['offspring_new'] += 1
                _stat_param['indices'] += 1
            new_offspring_list.append(indi)
        return new_offspring_list, _stat_param


class Exploration(object):
    def __init__(self, exploration_size, pops_individuals, _log, t_size, fitness_max, fitness_min, fitness_average,
                 pm_diversity):
        self.Exploration_size = exploration_size
        self.pops_individuals = copy.deepcopy(pops_individuals)
        self.log = _log
        self.T_size = t_size
        self.K = 1
        self.fitness_max = fitness_max
        self.fitness_min = fitness_min
        self.fitness_average = fitness_average
        self.Pm_diversity = pm_diversity
        self.best_indi = get_best_indi(self.pops_individuals)
        self.v_list = []
        self.sum_fitness = get_sum_fitness(self.pops_individuals)
        for indi in self.pops_individuals:
            wi = get_wi(indi, self.sum_fitness)
            spd_i = get_spd_i(indi, self.best_indi)
            self.v_list.append(get_hpd_i(spd_i, wi))

    @staticmethod
    def tournament_selection(v_list, t_size, k):
        selected_index = []
        index_list = range(len(v_list))
        for i in range(k):
            tournament = [random.choice(index_list) for i in range(t_size)]
            fitnesses = [v_list[tournament[i]] for i in range(t_size)]
            selected_index.append(tournament[np.argmax(fitnesses)])

        return selected_index

    def do_exploration(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0, 'OP': 0}
        selected_index = self.tournament_selection(self.v_list, self.T_size, self.Exploration_size)
        selected_indi = [self.pops_individuals[i] for i in selected_index]

        new_offspring_list = []
        for indi in selected_indi:
            p_ = random.random()
            pm_fitness = get_pm_fitness(self.K, self.fitness_max, self.fitness_min, self.fitness_average, indi.acc)
            self.log.info('Current pm_fitness: %.2f' % pm_fitness)
            self.log.info('Current Pm_diversity: %.2f' % self.Pm_diversity)

            pm = get_pm(pm_fitness, self.Pm_diversity)
            self.log.info('Current Pm: %.4f' % pm)
            if p_ < pm:
                _stat_param['offspring_new'] += 1
                _stat_param['OP'] += 1
                indi = Mutation.do_op_mutation(indi)
            else:
                _stat_param['offspring_from_parent'] += 1
            new_offspring_list.append(indi)
        self.log.info(
            'After Exploration New individuals: %d [op:%2d]  no_changes:%d' % (
                _stat_param['offspring_new'],
                _stat_param['OP'],
                _stat_param['offspring_from_parent'])
        )

        return new_offspring_list
