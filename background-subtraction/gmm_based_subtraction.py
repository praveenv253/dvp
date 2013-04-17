#!/usr/bin/env python

"""
Program to implement a background subtraction algorithm on a video using the
Gaussian Mixture method.
"""

import cv
import cv2
import numpy as np

TRAINING_FRAMES = 10 
TRAINING_FRAME_SKIP = 1
FRAME_SKIP = 1
THRESHOLD = 7
RHO = 0.001
ALPHA = 0.1
NUM_GAUSSIANS = 5
DATA_RANGE = 255

def update_gmm_params(test_vector, pk, means, variances, rho, alpha):
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

def background_subtract(frame, pk, means, variances):
    """Function to subtract the background from a frame, pixel by pixel"""
    
    # Threshold parameter v: number of std deviations
    v = THRESHOLD
    o = np.ones(frame.shape)
    z = np.zeros(frame.shape)
    
    # If the intensity is within v deviations of the mean, then it belongs
    # to that gaussian
    # hits says which gaussians the test vector falls into
    hits = np.where(abs((means.transpose() - frame.transpose()))
                                        < (v * np.sqrt(variances.transpose())),
                                        o.transpose(), z.transpose())
    hits = hits.transpose()
    # It should hit along all channels
    hits = np.prod(hits, axis=2)
    # But any one gaussian hit is enough for it to be background, provided that
    # gaussian has reasonable weight
    hits *= pk
    hits = hits.sum(axis=2)
    o = np.ones(hits.shape)
    z = np.zeros(hits.shape)
    hits = np.where(hits > 0.25, o, z).reshape(hits.shape + (1,))
    #hits = np.tensordot(hits, np.ones((1, 3)), axes=[2,0])
    return 255 * (1 - hits)

if __name__ == '__main__':
    print 'Program started'

    # Begin to capture image from camera
    #capture = cv.CaptureFromFile('Training_Background.mp4')
    capture = cv.CaptureFromCAM(0)
    #vectors = np.load('test_vectors.npy', mmap_mode='r')
    
    image = cv.QueryFrame(capture)
    mat = cv.GetMat(image)
    frame = np.asarray(mat)

    height, width, channels = frame.shape
    
    print 'Beginning training'
    
    try:
        pk = np.load('pk.npy')
        means = np.load('means.npy')
        variances = np.load('variances.npy')
    except IOError:
        ## Initialize parameters ##
        pk = np.ones((height, width, NUM_GAUSSIANS))
        means = np.ndarray((height, width, channels, NUM_GAUSSIANS))
        variances = np.ndarray((height, width, channels, NUM_GAUSSIANS))

        # All gaussians have equal weight
        pk /= NUM_GAUSSIANS
        
        # Gaussians have staggered means
        means[:, :, 0] = np.arange(start = DATA_RANGE / (2*NUM_GAUSSIANS),
                                   stop = DATA_RANGE,
                                   step = DATA_RANGE / NUM_GAUSSIANS,
                                   dtype = float                          )
        means[:, :, 1] = DATA_RANGE / 2
        means[:, :, 2] = DATA_RANGE / 2
        
        # Variances of the gaussians are such that the data width is spanned
        variances[:, :, 0] = (  np.ones(NUM_GAUSSIANS, dtype=float)
                              * DATA_RANGE / (2*NUM_GAUSSIANS)      )
        variances[:, :, 1] = DATA_RANGE / 2
        variances[:, :, 2] = DATA_RANGE / 2
        
        frame_number = 0
        while(1):
            print frame_number
            pk, means, variances = update_gmm_params(frame, pk, means,
                                                     variances, RHO, ALPHA)
            frame_number += 1
            if(frame_number > TRAINING_FRAMES):
                break
            for i in xrange(TRAINING_FRAME_SKIP):
                image = cv.QueryFrame(capture)
            mat = cv.GetMat(image)
            frame = np.asarray(mat)
        
        np.save('pk.npy', pk)
        np.save('means.npy', means)
        np.save('variances.npy', variances)
    
    print 'Training complete'
    
    print 'Now subtracting background...'
    # Proceed to subtract background of successive frames
    #capture = cv.CaptureFromFile('Test.mp4')
    #capture = cv.CaptureFromCAM(0)
    while(1):
        # Acquire frame
        #for i in xrange(FRAME_SKIP):
        #    image = cv.QueryFrame(capture)
        image = cv.QueryFrame(capture)
        # Convert frame image into a numpy array
        mat = cv.GetMat(image)
        frame = np.asarray(mat)
        # Send frame for background subtraction
        frame = background_subtract(frame, pk, means, variances).reshape((
                                                                     height,
                                                                     width, 1
                                                                 ))
        # Attempt own median blurring technique
        sum_frame = (  frame[2:, 1:-1] + frame[:-2, 1:-1]
                     + frame[1:-1, 2:] + frame[1:-1, :-2]
                     + frame[2:, 2:] + frame[:-2, :-2]
                     + frame[:-2, 2:] + frame[2:, :-2]    ) / 255
        o = np.ones(sum_frame.shape)
        z = np.zeros(sum_frame.shape)
        #frame[1:-1, 1:-1] = 255 * np.where(sum_frame > 6, o, z)
        # Convert the numpy array back into an image
        #frame2 = frame.copy()
        #cv2.medianBlur(frame, 3, frame2)
        #frame = frame2
        mat = cv.fromarray(frame)
        #cv.Erode(mat, mat)
        #mat2 = cv.CloneMat(mat)
        #cv.Smooth(mat, mat2, smoothtype=3)
        #mat = mat2
        #image2 = cv.CloneImage(image)
        #cv.Smooth(image, image2, cv.CV_MEDIAN, 3)
        #image = image2
        image = cv.GetImage(mat)
        # Write frame to display
        cv.ShowImage('Image_Window', image)
        cv.WaitKey(2)

