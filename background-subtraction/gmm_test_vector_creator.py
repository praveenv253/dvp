#!/usr/bin/env python

"""
Creates test vectors based on a mixed gaussian distribution.
"""

import numpy as np
import matplotlib.pyplot as pl

def _choose(cdf):
    x = np.random.random()
    # Find x in the cdf
    i = np.where(x < cdf, np.ones(cdf.shape), np.zeros(cdf.shape))
    return list(i).index(1)

def create_test_vectors(weights, means, variances, num_vectors):
    """
    Creates num_vectors number of vectors. Each vector is picked randomly from
    the nth gaussian with probability weights[n]/sum(weights).
    
    Returns a vector of vectors.
    """
    
    weight_sum = weights.sum()
    # Compute cumulative distribution function from weights
    cdf = np.cumsum(weights / weight_sum)
    
    for j in xrange(num_vectors):
        i = _choose(cdf)
        test_vector = np.random.multivariate_normal(means[i], 
                                                    np.diag(variances[i]))
        yield test_vector

if __name__ == '__main__':
    weights = np.array([1, 1], dtype=float)
    means = np.array([[50, 10],
                      [10, 50],
                    ])
    variances = np.array([[5, 5],
                          [5, 5],
                        ])
    num_vectors = 1000
    test_vectors = np.array(list(create_test_vectors(weights, means, variances, num_vectors)))
    np.save('gmm_test.npy', test_vectors)
    pl.plot(test_vectors[:, 0], test_vectors[:, 1], 'b,')
    pl.show()
