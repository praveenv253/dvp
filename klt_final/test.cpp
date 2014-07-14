#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

int main()
{
	// Load image
	//IplImage *img = cvLoadImage("test.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img = imread("test.jpg", CV_LOAD_IMAGE_COLOR);
	
	/*
	// Get image info
	int height    = img->height;
	int width     = img->width;
	int step      = img->widthStep;
	int channels  = img->nChannels;
	uchar *data   = (uchar *)img->imageData;
	*/
	
	//cvShowImage("Test", img);
	//cvWaitKey(0);
	imshow("Image", img);
	waitKey(0);
	
	//cvReleaseImage(&img);
	return 0;
}
