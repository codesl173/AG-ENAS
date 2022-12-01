import copy
import numpy as np


def do_1point_crossover(indi1, indi2, logger):
    parent1, parent2 = copy.deepcopy(indi1), copy.deepcopy(indi2)
    logger.info('crossover before')
    logger.info('parent 1: {}'.format(indi1.genotype))
    logger.info('parent 2: {}'.format(indi2.genotype))
    block_num_list1, block_num_list2 = indi1.each_block_num, indi2.each_block_num
    logger.info('crossover before\n\tthe block of parent 1: {};'
                '\n\tthe block of parent 1: {}'.format(block_num_list1, block_num_list2))

    logger.info('-' * 100)
    genotype1, genotype2 = parent1.genotype, parent2.genotype
    crossover_point = np.random.randint(1, indi2.block_num)
    logger.info('crossover_point:{}'.format(crossover_point))
    part1_1, part2_1 = genotype1[0:crossover_point], genotype2[0:crossover_point]
    part1_2, part2_2 = genotype1[crossover_point:], genotype2[crossover_point:]

    block_num_part1_1, block_num_part2_1 = block_num_list1[0:crossover_point], block_num_list2[0:crossover_point]
    block_num_part1_2, block_num_part2_2 = block_num_list1[crossover_point:], block_num_list2[crossover_point:]

    new_genotype1 = part1_1 + part2_2
    new_genotype2 = part2_1 + part1_2
    new_block_num_list1 = block_num_part1_1 + block_num_part2_2
    new_block_num_list2 = block_num_part2_1 + block_num_part1_2

    logger.info('new gene: {}'.format(new_genotype1))
    logger.info('new gene: {}'.format(new_genotype2))
    indi1.genotype, indi2.genotype = new_genotype1, new_genotype2
    indi1.each_block_num, indi2.each_block_num = new_block_num_list1, new_block_num_list2
    indi1.reset_acc()
    indi2.reset_acc()
    logger.info('-' * 100)
    logger.info('crossover after')
    logger.info('parent 1: {}'.format(indi1.genotype))
    logger.info('parent 2:: {}'.format(indi2.genotype))
    logger.info('crossover after\n\tthe block of parent 1: {};'
                '\n\tthe block of parent 1: {}'.format(indi1.each_block_num, indi2.each_block_num))

    return indi1, indi2
