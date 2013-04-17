"""
Code fragment that I tried out. It would be a shame to delete it.
"""
def g(x, m, v):
    """D-dimensional Gaussian function"""
    ret = np.ndarray(m.shape)
    # This should be a relatively small loop, since we don't expect too many
    # gaussians
    for k in xrange(NUM_GAUSSIANS):
        v_inv = np.diag(1 / v[:, k])
        ret[:, k] = np.exp(- np.dot((x - m[:, k]), v_inv.dot(x - m[:, k])))
        ret[:, k] *= np.sqrt(lin.det(v_inv)) / ((2*np.pi) ** (D/2))
    return ret

if __name__ == '__main__':
    ### Expectation-maximization algorithm ###

    # All gaussians have equal weight
    pk = np.ones(NUM_GAUSSIANS, dtype=float) / NUM_GAUSSIANS

    # Gaussians have staggered means
    means = np.ndarray((D, NUM_GAUSSIANS))
    means[0] = np.arange(start = DATA_RANGE / (2*NUM_GAUSSIANS),
                         stop = DATA_RANGE,
                         step = DATA_RANGE / NUM_GAUSSIANS,
                         dtype = float                          )
    means[1] = DATA_RANGE / 2
    #means = means.reshape((1, NUM_GAUSSIANS))
    #means = np.tensordot(np.ones((D, 1)), means, axes=[1, 0])
    
    # Variances of the gaussians are such that the data width is spanned
    variances = np.ndarray((D, NUM_GAUSSIANS))
    variances[0] = np.ones(NUM_GAUSSIANS, dtype=float) * DATA_RANGE / (2*NUM_GAUSSIANS)
    variances[1] = DATA_RANGE / 2
    #variances = variances.reshape((1, NUM_GAUSSIANS))
    #variances = np.tensordot(np.ones((D, 1)), variances, axes=[1, 0])

    # Iterate 100 times
    for i in range(10):
        print variances
        ## Expectation step. Compute p(k|n) ##
        
        pkn = np.ndarray((num_vectors, D, NUM_GAUSSIANS))
        for n in xrange(num_vectors):
            # This entire equation is in 2x3 variables, except for test_vectors
            qkn = pk * g(test_vectors[n], means, variances)
            pkn[n] = qkn / qkn.sum(axis=1).reshape((D, 1))
        
        ## Maximization step ##
        
        # The sum of pkn over num_vectors will be used often, so precalculate.
        sum_pkn = np.sum(pkn, axis=0)
        
        # Compute means
        means = np.sum(
                    (pkn.transpose() * test_vectors.transpose()).transpose(),
                    axis=0
                ) / sum_pkn
        
        # Compute variances
        for k in xrange(NUM_GAUSSIANS):
            # Subtracting a 1x2 matrix from a 500x2 matrix subtracts means from
            # each of the test_vectors. square_term becomes a 500x2 matrix.
            square_term = (test_vectors - means[:, k]) ** 2
            variances[:, k] = (np.sum(pkn[:, :, k] * square_term, axis=0)
                               / (sum_pkn[:, k] * D))
        
        # Compute probabilities of each gaussian
        pk = sum_pkn / num_vectors
    
    # Hopefully the algorithm would have worked by now.
    print means
    print variances
