#!/usr/bin/env python

import cv2.cv as cv
import numpy as np

MAXFRAMES = 250

# Begin to capture image from camera
capture = cv.CaptureFromCAM(0)
print capture

# First frame is used as a temporary frame to determine frame parameters
temp = cv.QueryFrame(capture)
width = temp.width
height = temp.height
channels = temp.nChannels

# Open a writer for writing processed output into a file
writer = cv.CreateVideoWriter('output.avi', 0, 15, cv.GetSize(temp), 1)

count = 0
while count < MAXFRAMES:
    # Acquire frame
    image = cv.QueryFrame(capture)
    # Convert frame image into a numpy array
    #mat = cv.GetMat(image)
    #frame = np.asarray(mat)
    # Operate on the numpy array
    #frame = 255 - frame             # Inverts colour
    #frame = np.average(frame, axis=2).reshape((height, width, 1))
    #frame = np.tensordot(frame, np.ones((1, 3)), axes=[2,0])
    #print frame
    # Convert the numpy array back into an image
    #mat = cv.fromarray(frame)
    #image = cv.GetImage(mat)
    # Write frame to output and display
    #cv.WriteFrame(writer, image)
    cv.ShowImage('Image_Window', image)
    cv.WaitKey(2)
    count += 1

