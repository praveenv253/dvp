#ifndef PROJ_ALIGN_H
#define PROJ_ALIGN_H

#include <opencv/cv.h>			// For type definitions

//#define Matx33f Matx33d

using namespace cv;

Mat subsample(Mat);
Point2f proj_warp(Matx33f, Point2i);
float proj_align(Mat, Mat, Matx33f &, int);

#endif
