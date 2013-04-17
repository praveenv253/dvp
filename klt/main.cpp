#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "utils.h"
#include "proj_align.h"

using namespace cv;

int main()
{
	Mat img1[5];
	Mat img2[5];
	
	char file1[] = "test_images/Canon A1200 slow pan test..mp40000.jpg";
	char file2[] = "test_images/Canon A1200 slow pan test..mp40010.jpg";
	
	img1[0] = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
	img2[0] = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);
	
	int i, k=10;		  // k is a parameter that determines amount of blurring
	for (i = 1 ; i < 5 ; i++) {
		img1[i] = img1[0].clone();
		GaussianBlur(img1[i], img1[i], Size(k*i + 3, k*i + 3), 0, 0);
		img2[i] = img2[0].clone();
		GaussianBlur(img2[i], img2[i], Size(k*i + 3, k*i + 3), 0, 0);
		imshow("image", img2[i]);
		waitKey(0);
	}
	
	imshow("Background", img1[0]);
	waitKey(0);
	
	imshow("Image 2 before transformation", img2[0]);
	waitKey(0);
	
	log<<"Before matrix initialization"<<std::endl;
		
	// Initilize projective transformation
	Matx33f B( 1.0, 0.0, 0.0,
			   0.0, 1.0, 0.0,
			   0.0, 0.0, 1.0 );
	
	log<<"B initialized"<<std::endl<<B<<std::endl;
	
	// Call the projective transform to compute the exact transform.
	for(i = 0 ; i < 5 ; i++) {
		// Start with most blurred version (top of pyramid)
		proj_align(img1[4-i], img2[4-i], B, 10);
	}
	
	log<<"Projective alignment completed"<<std::endl;
	
	log<<"Transformation:\n"<<B<<"\n";
	
	return 0;
}
