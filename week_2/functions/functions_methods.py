import numpy as np


def sigmoid(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1 * (x - beta_2)))
    return y

