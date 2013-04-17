#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Finds the parameters of a GMM distribution using the expectation maximization
algorithm.
"""

import numpy as np

NUM_GAUSSIANS = 5
DATA_RANGE = 255

def update_gmm_params(test_vector, pk, means, variances, rho, alpha):
    #num_vectors = test_vector.shape[0]
    D = test_vector.shape               # Number of dimensions
    
    ## Set up initial parameters ##
        
    # Check which gaussian the test vector belongs to.
    # => For "belonging", it must be within some number of std. deviations of 
    # the gaussian's mean.
    x = test_vector
    o = np.ones(means.shape)
    z = np.zeros(means.shape)
    # Threshold parameter v: number of std deviations
    v = 1
    
    # If the intensity is within v deviations of the mean, then it belongs
    # to that gaussian
    # hits says which gaussians the test vector falls into
    hits = np.where(abs((means.transpose() - x.transpose()).transpose()) 
                                              < (v * np.sqrt(variances)), o, z)
    hits = np.prod(hits, axis=2)   # All three colour dimensions should hit
    
    # Find the pixels which hit one or more gaussian
    yes_hits = hits.any(axis=2)
    yes_hit_indices = np.where(yes_hits == 1)
    
    # Check if none of the gaussians was hit
    # Find the pixels at which no gaussian in the present mixture was hit
    no_hits = 1 - yes_hits
    no_hit_indices = np.where(no_hits == 1)
    
    # Find which gaussian has minimum weight at each pixel and construct tuple
    min_gaussians = (np.argmin(pk[no_hit_indices], axis=1), )
    # min_gaussians has only one dimension now - the number of elements not hit
    # Update only those means and variances corresponding to pixels which did
    # not hit any gaussians
    indices = no_hit_indices + min_gaussians
    # Roll the colour dimensions and gaussians axis for convenience
    means = np.rollaxis(means, 3, 2)
    means[indices] = x[no_hit_indices]
    variances = np.rollaxis(variances, 3, 2)
    variances[indices] = DATA_RANGE / NUM_GAUSSIANS
    
    # If the test vector belongs to more than one gaussian, choose the one 
    # with a higher weight. If the heighest weights are equal, pick any.
    gaussian_choices = hits * pk
    max_gaussians = (np.argmax(gaussian_choices[yes_hit_indices], axis=1), )
    
    # Update the mean and variance of the matched gaussian as follows:
    #       mu = (1 - rho) * mu + rho * x
    # Udpate the variance as follows:
    #       var = (1 - rho) * var + rho * (x - mu)^2
    indices = yes_hit_indices + max_gaussians
    means[indices] = (1 - rho) * means[indices] + rho * x[yes_hit_indices]
    variances[indices] = (  (1 - rho) * variances[indices]
                          + rho * (means[indices] - x[yes_hit_indices]) ** 2)
    
    # Roll the axes back
    means = np.rollaxis(means, 2, 4)
    variances = np.rollaxis(variances, 2, 4)
    
    # Update the weights as follows:
    # Mark out the gaussian that was finally selected
    hits = np.zeros(hits.shape)
    hits[indices] = 1
    #num_matches[gaussian_index] += 1
    pk = (1 - alpha) * pk + alpha * hits
    pk = pk.transpose()
    pk /= pk.sum(axis=0)
    pk = pk.transpose()
    
    return pk, means, variances

def estimate_gmm(test_vectors, rho, alpha):    
    num_vectors = test_vectors.shape[0]
    D = test_vectors.shape[1]               # Number of dimensions
    
    ## Set up initial parameters ##
    
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
    
    # Number of test vectors that matched this particular gaussian
    #num_matches = np.zeros(pk.shape)
    
    # Take a test vector. Check which gaussian it belongs to.
    # => For "belonging", it must be within some number of std. deviations of 
    # the gaussian's mean.
    x = test_vectors
    o = np.ones(means.shape)
    z = np.zeros(means.shape)
    for n in xrange(num_vectors):
        # Threshold parameter v: number of std deviations
        v = 1
        
        # If the intensity is within v deviations of the mean, then it belongs
        # to that gaussian
        # hits says which gaussians the test vector falls into
        hits = np.where(abs((means.transpose() - x[n]).transpose()) < (v * np.sqrt(variances)), o, z)
        hits = np.prod(hits, axis=0)
        #print x[n]
        #print hits
        
        if not hits.any():
            # Find the gaussian with the lowest weight so far and replace it
            min_gaussian = np.argmin(pk)
            means[:, min_gaussian] = x[n]
            variances[:, min_gaussian] = DATA_RANGE / NUM_GAUSSIANS
            continue
        
        # If the test vector belongs to more than one gaussian, choose the one 
        # with a higher weight. If the heighest weights are equal, pick any.
        gaussian_choices = hits * pk
        gaussian_index = np.argmax(gaussian_choices)
        #print gaussian_index
        
        # Update the mean and variance of the matched gaussian as follows:
        #       mu = (1 - rho) * mu + rho * x
        # Udpate the variance as follows:
        #       var = (1 - rho) * var + rho * (x - mu)^2
        means[:, gaussian_index] = (  (1 - rho) * means[:, gaussian_index]
                                      + rho * x[n])
        variances[:, gaussian_index] = (  ((1 - rho) * variances[:, gaussian_index])
                                          + rho * ((means[:, gaussian_index] - x[n])) ** 2 )
        
        # Update the weights as follows:
        hits = np.zeros(hits.shape)
        hits[gaussian_index] = 1
        #num_matches[gaussian_index] += 1
        pk = (1 - alpha) * pk + alpha * hits
        pk /= pk.sum(axis=0)
        
    return pk, means, variances
    
if __name__ == '__main__':
    # Get the test vectors
    try:
        test_vectors = np.load('gmm_test.npy', 'r')
    except IOError:
        print 'File not found'
    
    pk, means, variances = estimate_gmm(test_vectors, 0.001, 0.05)
    print pk
    print means
    print variances

