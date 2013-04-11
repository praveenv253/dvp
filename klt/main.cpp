#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "utils.h"
#include "proj_align.h"

using namespace cv;

int main()
{
	//char file1[] = "test_images/trunks_1.png";
	//char file2[] = "test_images/trunks_2.png";
	
	char file1[] = "test_images/Canon A1200 slow pan test..mp40000.jpg";
	char file2[] = "test_images/Canon A1200 slow pan test..mp40010.jpg";
	
	Mat img1 = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);
	
	imshow("Image 2 before transformation", img2);
	waitKey(0);
	
	log<<"Before matrix initialization"<<std::endl;
		
	// Initilize projective transformation
	Matx33f B( 1.0, 0.0, 0.0,
			   0.0, 1.0, 0.0,
			   0.0, 0.0, 1.0 );
	
	log<<"B initialized"<<std::endl<<B<<std::endl;
	
	// Call the projective transform to compute the exact transform.
	proj_align(img1, img2, B, 5);
	
	log<<"Projective alignment completed"<<std::endl;
	
	imshow("Image 1", img1);
	waitKey(0);
	imshow("Image 2", img2);
	waitKey(0);
	
	log<<"Transformation:\n"<<B<<"\n";
	
	return 0;
}
