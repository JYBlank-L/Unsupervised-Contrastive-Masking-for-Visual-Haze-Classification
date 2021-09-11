import random as rand
import numpy as np
import math


def sample(label, Percent_of_testing_samples):
    '''
        :param

        label: data of labels

        :return

        index_all: test samples' index
    '''

    # Compute statistics of classes
    index_of_measure_0 = np.where(label == 0)[1]
    index_of_measure_1 = np.where(label == 1)[1]
    index_of_measure_2 = np.where(label == 2)[1]
    index_of_measure_3 = np.where(label == 3)[1]
    index_of_measure_4 = np.where(label == 4)[1]
    number_of_each_measure = [len(index_of_measure_0), len(index_of_measure_1), len(index_of_measure_2),
                              len(index_of_measure_3), len(index_of_measure_4)]

    # Use Dataset of pytorch
    # Randomly generate index of testing samples evenly from each class
    index_0 = rand.sample(range(number_of_each_measure[0]),
                          math.ceil(number_of_each_measure[0] * Percent_of_testing_samples))
    index_1 = rand.sample(range(number_of_each_measure[1]),
                          math.ceil(number_of_each_measure[1] * Percent_of_testing_samples))
    index_2 = rand.sample(range(number_of_each_measure[2]),
                          math.ceil(number_of_each_measure[2] * Percent_of_testing_samples))
    index_3 = rand.sample(range(number_of_each_measure[3]),
                          math.ceil(number_of_each_measure[3] * Percent_of_testing_samples))
    index_4 = rand.sample(range(number_of_each_measure[4]),
                          math.ceil(number_of_each_measure[4] * Percent_of_testing_samples))

    index_all = measure = np.concatenate((index_of_measure_0[index_0], index_of_measure_1[index_1],
                                          index_of_measure_2[index_2], index_of_measure_3[index_3],
                                          index_of_measure_4[index_4]))

    return index_all
