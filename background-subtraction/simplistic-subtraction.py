#!/usr/bin/env python

"""
Program to implement a background subtraction algorithm on a video.
"""

import cv
import numpy as np

# Number of frames on which to train. These will be used as a template
# background to subtract.
TRAINING_FRAMES = 20

def train(capture, num_frames, height, width, channels, flatten=False):
    """Function used to set up the environment for background subtraction."""
    
    if flatten:
        channels = 1
    
    count = 0
    frames = np.zeros((num_frames, height, width, channels))
    while count < num_frames:
        # Acquire frame to train upon
        image = cv.QueryFrame(capture)
        # Convert frame image into a numpy array
        mat = cv.GetMat(image)
        frame = np.asarray(mat)
        if flatten:
            # Flatten colours to procure a greyscale image
            frame = np.average(frame, axis=2).reshape((height, width, 1))
        # Collate frames
        frames[count] = frame
        if count == 0:
            print frame.shape
        #cv.WriteFrame(writer, image)
        #cv.ShowImage('Image_Window',image)
        #cv.WaitKey(2)
        count += 1
    
    # means and variances are matrices giving the mean intensity at each pixel
    # and the corresponding intensity variance.
    means = np.average(frames, axis=0)
    stacker = np.ones((num_frames, 1))
    reshaped_means = means.reshape((1, height, width, channels))
    stacked_means = np.tensordot(stacker, reshaped_means, axes=[1,0])
    variances = np.average((frames - stacked_means) ** 2, axis=0)
    print means.shape
    return means, variances

def background_subtract(frame, means, variances):
    """Function to subtract the background from a frame, pixel by pixel"""
    
    # Threshold parameter v
    v = 2

    # If the intensity is within `v` variances of the mean, it is background,
    # otherwise it is foreground
    o = 255 * np.ones(frame.shape)
    z = np.zeros(frame.shape)
    
    frame = np.where(abs(frame-means) < v * variances, z, o)
    return frame

if __name__ == '__main__':
    print 'Program started'

    # Begin to capture image from camera
    capture = cv.CaptureFromCAM(0)
    
    # Discard first 200 frames, as the camera takes time to adjust
    for i in xrange(200):
        temp = cv.QueryFrame(capture)
    
    # First frame is used as a temporary frame to determine frame parameters
    temp = cv.QueryFrame(capture)
    width = temp.width
    height = temp.height
    channels = temp.nChannels
    
    # Open a writer for writing processed output into a file
    #writer = cv.CreateVideoWriter('output.avi', 0, 15, cv.GetSize(temp), 1)
    
    print 'Beginning training'
    # Determine mean and variance of intensity distribution at each pixel
    means, variances = train(capture, TRAINING_FRAMES,
                             height, width, channels, True)
    print 'Training complete'
    
    print 'Now subtracting background...'
    # Proceed to subtract background of successive frames
    while(1):
        # Acquire frame
        image = cv.QueryFrame(capture)
        # Convert frame image into a numpy array
        mat = cv.GetMat(image)
        frame = np.asarray(mat)
        # Flatten the image
        frame = np.average(frame, axis=2).reshape((height, width, 1))
        # Send frame for background subtraction
        frame = background_subtract(frame, means, variances)
        # Convert the numpy array back into an image
        mat = cv.fromarray(frame)
        image = cv.GetImage(mat)
        # Write frame to display
        cv.ShowImage('Image_Window', image)
        cv.WaitKey(2)

