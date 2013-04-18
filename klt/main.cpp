#include <iostream>
#include <sstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "utils.h"
#include "proj_align.h"
#include "stitch.h"

#define FILESTARTNUM 40150
#define NUMITER 10		   // Number of iterations for each projective transform
using namespace cv;

int main()
{
	Mat img1[5];
	Mat img2[5];
	Mat canvas, tempimage;
	
	char filename[] = "test_images/Canon A1200 slow pan test..mp"; 
	char extension[] = ".jpg";
	std::stringstream s;
	int filenum = FILESTARTNUM;

	int k = 10;			  // k is a parameter that determines amount of blurring

	for( ; ; filenum--) {
		
		s.str("");
		s<<filename<<filenum<<extension;
		
		if(filenum == FILESTARTNUM) {
			img1[0] = imread(s.str(), CV_LOAD_IMAGE_GRAYSCALE);
			filenum--;
			s.str("");
			s<<filename<<filenum<<extension;
		}
		else {
			img1[0] = canvas;
		}
		
		log<<s.str();
		
		img2[0] = imread(s.str(), CV_LOAD_IMAGE_GRAYSCALE);
		// Termination condition: no more images to stitch
		if (img2 == NULL)
			break;
		
		int i;
		for (i = 1 ; i < 5 ; i++) {
			img1[i] = img1[0].clone();
			GaussianBlur(img1[i], img1[i], Size(k*i + 3, k*i + 3), 0, 0);
			img2[i] = img2[0].clone();
			GaussianBlur(img2[i], img2[i], Size(k*i + 3, k*i + 3), 0, 0);
		}
		
		log<<"Before matrix initialization"<<std::endl;
		
		// Initilize projective transformation
		Matx33f B( 1.0, 0.0, 0.0,
				   0.0, 1.0, 0.0,
				   0.0, 0.0, 1.0 );
	
		log<<"B initialized"<<std::endl<<B<<std::endl;
		
		// Call the projective transform to compute the exact transform.
		for(i = 0 ; i < 5 ; i++) {
			// Start with most blurred version (top of pyramid)
			tempimage = proj_align(img1[4-i], img2[4-i], B, NUMITER);
		}
		
		log<<"Beginning cropping"<<std::endl;
		tempimage = crop(tempimage);
		log<<"Cropping completed"<<std::endl;
		
		canvas = tempimage;
		
		log<<"Projective alignment completed"<<std::endl;
		
		log<<"Transformation:\n"<<B<<"\n";
	
	}
	
	return 0;
}
