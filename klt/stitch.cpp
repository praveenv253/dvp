#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "utils.h"
#include "stitch.h"

using namespace cv;

Mat crop(Mat image)
{
	Size size = image.size();
	int type = image.type();
	int left=0, top=0, right=size.width, bottom=size.height;
	
	log<<"Computing border regions"<<std::endl;
	
	Mat row_zeros = Mat::zeros(1, right, type);
	Mat col_zeros = Mat::zeros(bottom, 1, type);
	
	log<<"Row size: "<<row_zeros.size().width<<std::endl;
	log<<"Image row size: "<<image.row(top).size().width<<std::endl;
	log<<"Column size: "<<col_zeros.size().height<<std::endl;
	log<<"Image column size: "<<image.col(left).size().height<<std::endl;
	
	while(countNonZero(image.row(top) != row_zeros) == 0)
		top++;
	
	while(countNonZero(image.col(left) != col_zeros) == 0)
		left++;
	
	while(countNonZero(image.row(bottom-1) != row_zeros) == 0)
		bottom--;
	
	while(countNonZero(image.col(right-1) != col_zeros) == 0)
		right--;
	
	log<<"Done computing border regions"<<std::endl;
	log<<"Crop details: "<<top<<" "<<left<<" "<<bottom<<" "<<right<<std::endl;
	
	log<<"Resizing image"<<std::endl;
	Rect crop_rect(left, top, right - left, bottom - top);
	Mat cropped_image = image(crop_rect);
	log<<"Done resizing image"<<std::endl;
	
	//imshow("Cropped image", cropped_image);
	//waitKey(0);
	
	return cropped_image;
}

Mat stitch(Mat &canvas, Mat newimage)
{

}
