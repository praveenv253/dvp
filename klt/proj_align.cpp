#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "lu.h"
#include "utils.h"
#include "proj_align.h"

// lookup is used to prevent projective-transformed values from referring to
// pixels with negative indices
#define lookup(val, limit) ( ((val) + 20*(limit)) % (limit) )
// These are easy referencing methods for commonly used arrays
// Note the interchange of u and v! u is the x-component and v is the
// y-component while calling the function, however when accessing the matrix,
// we need to first access the row and then the column => y then x!
#define newimage(u, v) newimage.at<uchar>((v), (u))
#define oldimage(u, v) oldimage.at<uchar>((v), (u))
#define tempimage(u, v) tempimage.at<uchar>((v), (u))
#define fnewimage(u, v) fnewimage.at<float>((v), (u))
#define foldimage(u, v) foldimage.at<float>((v), (u))
// Some threshold value, I think
#define IMPROVEMENT 0.02

//#define Matx33f Matx33d

using namespace cv;

/* Function to subsample a given input image.
 * Allocates space for a new image at half the width and height and subsamples.
 * Returns the subsampled image
 */
Mat subsample(Mat input)
{
	int width, height;
	
	Size size = input.size();
	width = size.width;
	height = size.height;
	
	/* Allocating storage for the result image. */
	Mat result(height/2, width/2, CV_8UC3);
	
	/* Main loop. */
	MatIterator_<Vec3b> input_it, result_it, end;
	for(input_it = input.begin<Vec3b>(), end = input.end<Vec3b>(), 
			result_it = result.begin<Vec3b>() ;
			input_it != end ;
			input_it+=2, result_it++) {
		(*result_it)[0] = (*input_it)[0];
		(*result_it)[1] = (*input_it)[1];
		(*result_it)[2] = (*input_it)[2];
	}
	
	return result;
}

/* Function to apply a given projective transform to a given point (x, y).
 * Returns the transformed point (x', y'). Note that the returned point has
   floating point coordinates.
 */
Point2f proj_warp(Matx33f B, Point2i u)
{
	Point2f v;
	
	float x = B(0, 0) * u.x + B(0, 1) * u.y + B(0, 2);
	float y = B(1, 0) * u.x + B(1, 1) * u.y + B(1, 2);
	float z = B(2, 0) * u.x + B(2, 1) * u.y + B(2, 2);
	
	// We do not type cast to int. This way, error is also handed over to the
	// calling function.
	v.x = x / z;
	v.y = y / z;
	
	return v;
}

/* Function to find the projective warp between two given images, starting with
   a given projective transformation. 
 * Starts with the projective transform B and then iterates upto maxiter times
   until there is "alignment".
 */
Mat proj_align(Mat oldimage, Mat newimage, Matx33f &B, int maxiter)
{
	log<<"Beginning proj_align"<<std::endl;
	
	float dIdB[9];
	
	// lhs - 8x8 optimization matrix for recovering projective tranformation,
	// indexing starts from 1, 
	// http://graphics.cs.cmu.edu/courses/15-463/2010_fall/Papers/proj.pdf
	// rhs right hand side of optimization
	float lhs[9][9], rhs[9], d, fracx, fracy;
	int indx[9];
	
	// sumer - sum of errors
	// scaling, I-x image gradient in x, I_y image gradient in y, e is error
	float sumer, presumer, I_x, I_y, e;
	int i, j, nloop, x, y, u, v;
	
	// new transformation
	Matx33f Bnew( 0, 0, 0,
				  0, 0, 0,
				  0, 0, 0 );

	// summse - sum of mean square error
	float premse, summse;

	Size size = newimage.size();
	int bckwidth = size.width;
	int bckheight = size.height;
	size = oldimage.size();
	int imwidth = size.width;
	int imheight = size.height;
	
	log<<"Done defining variables"<<std::endl;
	
	// Canonical homography: which is a fancy word for "normalized".
	for (i = 0 ; i < 3 ; i++)
		for (j = 0 ; j < 3 ; j++)
			B(i, j) = B(i, j) / B(2, 2);
	
	// Type cast the images that we got as input to double type. We expect that
	// they are coming directly from an imread operation, in which case they
	// are likely uchar.
	Mat fnewimage = Mat(bckheight, bckwidth, CV_32F);
	Mat foldimage = Mat(imheight, imwidth, CV_32F);
	Mat tempimage = Mat(imheight+200, imwidth+200, CV_8U, Scalar::all(0));
	
	log<<"newimage = \n"<<newimage.Mat::operator()(Range(0, 5), Range(0, 5));
	log<<"\noldimage = \n"<<oldimage.Mat::operator()(Range(0, 5), Range(0, 5));
	
	// Now to set the type right...
	for(j = 0 ; j < imheight ; j++) {
		for(i = 0 ; i < imwidth ; i++) {
			fnewimage(i, j) = newimage(i, j);
			foldimage(i, j) = oldimage(i, j);
		}
	}
		
	log<<"\nfnewimage = \n"<<fnewimage.Mat::operator()(Range(0, 5), Range(0, 5));
	log<<"\nfoldimage = \n"<<foldimage.Mat::operator()(Range(0, 5), Range(0, 5));
		
	log<<"\nType casting worked"<<std::endl;
	
	int bckwidthcur = bckwidth;
	int bckheightcur = bckheight;
	int imwidthcur = imwidth;
	int imheightcur = imheight;
	
	log<<"Current image height = "<<imheightcur<<std::endl;
	log<<"Current image width = "<<imwidthcur<<std::endl;
	
	nloop = 0;
	
	// Keep improving projective transform.
	while (nloop < maxiter) {
		log<<"Entered while loop"<<std::endl;
		
		presumer = 0.0;
		sumer = 0.0;
		int npts1 = 0;
		int npts2 = 0;
		
		for (i = 1; i < 9; i++) {
			rhs[i] = 0.0;
			for (j = 1; j < 9; j++) {
				lhs[i][j] = 0.0;
			}
		}
		
		// Compute the projective warp of each pixel of the image
		for (y = 1; y < imheightcur - 1; y++) {
			for (x = 1; x < imwidthcur - 1; x++) {
				
				// Projective warp of (x, y)
				Point2f result = proj_warp(B, Point2i(x, y));
				u = int(result.x);
				v = int(result.y);
				
				// Compute the fractional part of the result. This gives you
				// some sort of error estimate on the transform that we just
				// performed.
				fracx = (u > 0) ? (result.x - u) : (u - result.x);
				fracy = (v > 0) ? (result.y - v) : (v - result.y);
				
				// Check if within image boundaries
				if(		u < 0
					||	v < 0
					||	u >= (bckwidthcur - 1)
					||	v >= (bckheightcur - 1)
					||	fnewimage(u, v) == 0.0f
					||	fnewimage(lookup(u+1, bckwidthcur), v) == 0.0
					||	fnewimage(u, lookup(v+1, bckheightcur)) == 0.0
				) {
					continue;
				}
				
				// Compute derivatives
				I_x = (foldimage(x+1, y) - foldimage(x-1, y)) / 2.0;
				I_y = (foldimage(x, y+1) - foldimage(x, y-1)) / 2.0;
				
				// Not too sure what the objective of this is...
				// Bi-linear interpolation and the corresponding error e
				e = (  (1-fracx) * (1-fracy) * fnewimage(u, v)
					 + fracx * (1-fracy)
					   * fnewimage(lookup(u + 1, bckwidthcur), v)
					 + (1-fracx) * fracy
					   * fnewimage(u, lookup(v + 1, bckheightcur))
					 + fracx * fracy
					   * fnewimage(lookup(u + 1, bckwidthcur), lookup(v + 1,
					   											bckheightcur))
					 - foldimage(x, y)
					);
				
				presumer += e * e;					// Sum of squares of errors
				npts1++;

				//Taylor expansion
				dIdB[1] = I_x * x;
				dIdB[2] = I_x * y;
				dIdB[3] = I_x;
				dIdB[4] = I_y * x;
				dIdB[5] = I_y * y;
				dIdB[6] = I_y;
				dIdB[7] = - (x * result.x * I_x) - (x * result.y * I_y);
				dIdB[8] = - (y * result.x * I_x) - (y * result.y * I_y);

				for (i = 1; i < 9; i++) {
					rhs[i] -= dIdB[i] * e;	// Evidently the rhs ought to be
											// zero, but instead, it is holding
											// some kind of error term
					for (j = 1; j < 9; j++) {
						lhs[i][j] += dIdB[i] * dIdB[j];
					}
				}
				//log<<"Done with inner for loop with x = "<<x<<std::endl;
			}
			//log<<"Done with outer for loop with y = "<<y<<std::endl;
		} // end compute projective transform... I think.
		
		log<<"Done with projective transform iteration"<<std::endl;
		log<<"presumer = "<<presumer<<std::endl;
		
		// If the error is not zero, which it obviously will not be, we are
		// going to do something more...
		if (presumer != 0.0) {
			log<<"Entered the error correction loop"<<std::endl;
			
			// LU decomposition for lhs
			ludcmp(lhs, 8, indx, &d);		// When was d set!? What does it do?
			lubksb(lhs, 8, indx, rhs);
			
			// We're somehow going to calculate a new projective transform
			// Note how it depends on the rhs, which is some measure of the
			// error
			for (i = 1 ; i < 9 ; i++) {
				Bnew((i-1) / 3, (i-1) % 3) = B((i-1) / 3, (i-1) % 3) + rhs[i];
			}
			Bnew(2, 2) = 1.0;
			log<<"New transformation"<<std::endl<<Bnew<<std::endl;
			
			// Here we go again. Why are we doing this twice within a single
			// while loop?
			// Either way, this whole thing should just be a function if it's
			// going to be done twice like this
			for (y = 1; y < imheightcur - 1; y++) {
				for (x = 1 ; x < imwidthcur - 1; x++) {
					
					Point2f result = proj_warp(Bnew, Point2i(x, y));
					u = int(result.x);
					v = int(result.y);
					
					// Compute the fractional part of the result. This gives you
					// some sort of error estimate on the transform that we just
					// performed.
					fracx = (u > 0) ? (result.x - u) : (u - result.x);
					fracy = (v > 0) ? (result.y - v) : (v - result.y);
					
					if(		u < 0
						||	v < 0
						||	u >= (bckwidthcur - 1)
						||	v >= (bckheightcur - 1)
						||	fnewimage(u, v) == 0.0f
						||	fnewimage(lookup(u+1, bckwidthcur), v) == 0.0
						||	fnewimage(u, lookup(v+1, bckheightcur)) == 0.0
					) {
						continue;
					}
					
					// Compute derivatives
					I_x = (foldimage(x+1, y) - foldimage(x-1, y)) / 2.0;
					I_y = (foldimage(x, y+1) - foldimage(x, y-1)) / 2.0;
					
					// Not too sure what the objective of this is...
					// Bi-linear interpolation and the corresponding error e
					e = (  (1-fracx) * (1-fracy) * fnewimage(u, v)
						 + fracx * (1-fracy)
						   * fnewimage(lookup(u + 1, bckwidthcur), v)
						 + (1-fracx) * fracy
						   * fnewimage(u, lookup(v + 1, bckheightcur))
						 + fracx * fracy
						   * fnewimage(lookup(u + 1, bckwidthcur),
						   			  lookup(v + 1, bckheightcur))
						 - foldimage(x, y)
						);
					
					sumer += e * e;
					npts2++;
				}
			} // End of the projective transform application again, with Bnew 
			log<<"Done with error correction projective transform"<<std::endl;
			
			if ((sumer/npts2) <= (presumer/npts1)) {
				for (i = 0 ; i < 3 ; i++)
					for (j = 0 ; j < 3 ; j++)
						B(i, j) = Bnew(i, j);
			}
			
			//if (sumer/npts2 + IMPROVEMENT >= presumer/npts1)
			//	nloop = maxiter;
			
			nloop ++;
		
		} // End of if(presumer != 0.0)
		else {
			nloop = maxiter; // This is a complicated way of saying "break, but
							 // do the stuff that follows."
		}

		if (npts1 == 0)
			premse = 0.0;	// Some mean square error estimate
		else
			premse = presumer/npts1;

		if (npts2 == 0)
			summse = 0.0;	// Sum of mean square errors
		else
			summse = sumer/npts2;

		log<<"presumermse = "<<premse<<" sumermse = "<<summse<<"\n";
	
	} // End while
	
	// Actually apply the projective transform to see the progression of
	// results
	for(j = -100 ; j < imheight ; j++) {
		for(i = -100 ; i < imwidth ; i++) {
			Point2f result = proj_warp(B, Point2i(i, j));
			u = result.x;
			v = result.y;
			tempimage(i+100, j+100) = 0;
			if (    u < 0
				 || v < 0
				 || u >= imwidthcur
				 || v >= imheightcur
			) {
			  	continue;
			}
			tempimage(i+100, j+100) = uchar(fnewimage(u, v));
		}
	}
	
	log<<"Done with application of projective transform";
	
	// Add the background to the new image
	for(j = 0 ; j < imheight ; j++) {
		for(i = 0 ; i < imwidth ; i++) {
			tempimage(i+100, j+100) = oldimage(i, j);
		}
	}

	imshow("Transformed background image", tempimage);
	waitKey(0);

	// Return something useful
	return tempimage;
}
