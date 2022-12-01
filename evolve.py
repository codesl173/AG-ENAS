import copy
import numpy as np
from acromu import ACROMU
from evaluate import FitnessEvaluate
from population import Population
from selection_operator import Selection
from Utils.utils import StatusUpdateTool, Utils, Log
import Mutation


class EvolveCNN(object):
    def __init__(self, parameters):
        self.parent_pops = None
        self.params = parameters
        self.pops = None
        self.prev_matrix = []

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(params['pop_size'], 0)
        pops.initialize()
        self.pops = pops
        op_m = Mutation.uniform_initialize_op_matrix()
        Log.info('initialize_matrix:\n%s' % op_m)
        for i in range(pops.individuals[0].block_num):
            print('the {} matrix.'.format(i))
            Mutation.save_op_matrix(op_m, i, )
            self.prev_matrix.append(op_m)
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log, self.params)
        fitness.evaluate()

    def crossover_and_mutation(self):
        k1, k2, k = 0.4, 0.8, 1
        cm = ACROMU(k1, k2, k, Log, self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        self.parent_pops.get_old()
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        indi_list, v_list, params_list, age_list = [], [], [], []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc if indi.init_age < 4 else 0)
            params_list.append(indi.params)
            age_list.append(indi.init_age)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc if indi.init_age < 4 else 0)
            params_list.append(indi.params)
            age_list.append(indi.init_age)
        print('#' * 100)
        Log.info('Before Environment_selection age_list: {}'.format(age_list))
        print('#' * 100)
        _str = []
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'Indi-%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        for _, indi in enumerate(self.parent_pops.individuals):
            _t_str = 'Pare-%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        max_index = np.argmax(v_list)
        selection = Selection()
        selected_index_list = selection.roulette_selection(v_list, k=self.params['pop_size'])
        if max_index not in selected_index_list:
            first_selected_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selected_v_list)
            selected_index_list[min_idx] = max_index
        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.params['pop_size'], self.pops.gen_no + 1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
        next_age_list = []
        for indi in self.pops.individuals:
            next_age_list.append(indi.init_age)
        print('#' * 100)
        Log.info('After Environment_selection age_list: {}'.format(next_age_list))
        print('#' * 100)
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'After environment_selection new -%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        _file = './populations/ENVI_%2d.txt' % self.pops.gen_no
        Utils.write_to_file('\n'.join(_str), _file)
        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def do_work(self, max_gen):
        Log.info('*' * 25)
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % gen_no)
                self.pops = Utils.load_population('begin', gen_no)
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % gen_no)
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % gen_no)
        gen_no += 1
        alpha, c = 0.5, 1.414
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            print('Current_gen:', curr_gen)
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % curr_gen)
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % curr_gen)
            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % curr_gen)
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % curr_gen)
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % curr_gen)
            for i in range(self.pops.individuals[0].block_num):
                op_matrix = Mutation.get_ucb_opm(self.pops.individuals, i, c)
                refine_opm = alpha * self.prev_matrix[i] + (1 - alpha) * op_matrix
                self.prev_matrix[i] = refine_opm
                print('the length of the pre matrix', len(self.prev_matrix))
                Mutation.save_op_matrix(refine_opm, i, )
                Log.info('gen%d \n block_%d Op_matrix:\n%s' % (curr_gen, i, refine_opm))
        StatusUpdateTool.end_evolution()


if __name__ == '__main__':
    params = StatusUpdateTool.get_init_params()
    evoCNN = EvolveCNN(params)
    evoCNN.initialize_population()
    evoCNN.do_work(max_gen=20)
