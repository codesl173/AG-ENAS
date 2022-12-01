import time
from evaluate_genotype import evaluate_indi_during_search
from Utils.utils import Utils


class FitnessEvaluate(object):
    def __init__(self, pops, log, params):
        self.pops_individuals = pops
        self.log = log
        self.params = params

    def evaluate(self):
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.pops_individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                indi.acc = float(_acc)
                indi.age
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f,its age:%d' % (
                    indi.id, _key, float(_acc), indi.acc, indi.init_age))

        self.log.info('Total hit %d individuals for fitness' % _count)

        for indi in self.pops_individuals:
            self.log.info('%s' % indi)
            if indi.acc < 0:
                start = time.time()
                indi.acc, indi.params, _ = evaluate_indi_during_search(indi.genotype, C=self.params['C'],
                                                                       num_classes=self.params['num_class'],
                                                                       N=self.params['N'],
                                                                       total_epochs=self.params['epochs'],
                                                                       lr=self.params['lr'],
                                                                       batch_size=self.params['batch_size'])
                print('time used is: {}'.format(time.time() - start))
                Utils.save_fitness_at_once(indi)
            else:
                file_name = indi.id
                self.log.info('%s has inherited the fitness as %.5f, no need to evaluate' % (file_name, indi.acc))
                f = open('./populations/after_%s.txt' % (file_name[4:6]), 'a+')
                f.write('%s=%.5f\n' % (file_name, indi.acc))
                f.flush()
                f.close()

        Utils.save_fitness_to_cache(self.pops_individuals)
