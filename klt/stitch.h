#ifndef STITCH_H
#define STITCH_H

#include <opencv/cv.h>

using namespace cv;

Mat crop(Mat);
Mat stitch(Mat &, Mat);

#endif
