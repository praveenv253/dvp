#!/usr/bin/env python

import sys

NUM_FRAMES = int(sys.argv[1])
FILENAME = 'Training_Background.mp4'
OUTFILE = sys.argv[2]

import numpy as np
import cv

if __name__ == '__main__':
    
    capture = cv.CaptureFromFile(FILENAME)
    f = open(OUTFILE, 'w')
    
    image = cv.QueryFrame(capture)
    mat = cv.GetMat(image)
    frame = np.asarray(mat)
    
    height, width, channels = frame.shape
    
    a = np.ndarray((NUM_FRAMES, height, width, channels))
    for i in xrange(NUM_FRAMES):
        a[i] = frame
        image = cv.QueryFrame(capture)
        mat = cv.GetMat(image)
        frame = np.asarray(mat)
    
    np.save(f, a)
    f.close()
