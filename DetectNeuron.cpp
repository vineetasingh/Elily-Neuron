  //Detcts dendtites, developed and under developed. Finds start points of dendrites

#include <iostream>     // std::cout
#include <algorithm>    // std::min_element, std::max_element
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include<conio.h>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<vector>
#include "opencv2/opencv.hpp "



using namespace cv;
using namespace std;
int thresh = 1;
int gauKsize = 11;
int maxGauKsize = 21;
vector<float> complete_eigvec;

Mat createmaskimage(Mat image, Mat dXX, Mat dYY, Mat dXY)
{
	Mat maskimg(image.rows, image.cols, CV_8U);
	maskimg = cv::Scalar(0, 0, 0);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	std::vector<float> eigenvalues(2);

	//----------Inside image

	for (int i = 1; i < image.rows; i++) {
		for (int j = 1; j < image.cols; j++) {
			hessian_matrix.at<float>(Point(0, 0)) = dXX.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 1)) = dYY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(0, 1)) = dXY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 0)) = dXY.at<float>(Point(j, i));

			// find eigen values of hessian matrix /* Larger  eigenvalue  show the  direction  of  intensity change, while  smallest  eigenvalue  show  the  direction  of  vein.*/
			eigen(hessian_matrix, eigenvalues, eigenvec);

			/*Main Condition*/if (abs(eigenvalues[0])<5 && abs(eigenvalues[1])>6)

			{
				/*Condition 1*/		if ((abs(eigenvec.at<float>(Point(0, 0))) > 0) && (abs(eigenvec.at<float>(Point(0, 0)) < 0.4)))
				{
					circle(maskimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 2, 8, 0);//orange
				}
				/*Condition 2*/		if ((abs(eigenvec.at<float>(Point(0, 0))) > 0.5) && (abs(eigenvec.at<float>(Point(0, 0)) < 1.2)))
				{
					circle(maskimg, cv::Point(j, i), 1, cv::Scalar(255, 255,255), 2, 8, 0);//blue
				}
			}
		}
	}
	//imshow("maskimg", maskimg);
	return maskimg;
}

void filterHessian(Mat image)
{
	
	int co = 0;
	Mat org = imread("test.png");
	Mat checkimg(image.rows, image.cols, CV_8U);
	Mat overlapimage(image.rows, image.cols, CV_8U);
	Mat dendritetips(image.rows, image.cols, CV_8U);
	Mat overlapbinimage(image.rows, image.cols, CV_8U);
	cv::Mat dXX, dYY, dXY;
	std::vector<float> eigenvalues(2);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	//std::vector<float> eigenvec(2,2); //Mat eigenvec, eigenvalues;

	//calculte derivatives
	cv::Sobel(image, dXX, CV_32F, 2, 0);
	cv::Sobel(image, dYY, CV_32F, 0, 2);
	cv::Sobel(image, dXY, CV_32F, 1, 1);

	//apply gaussian filtering to the image
	cv::Mat gau = cv::getGaussianKernel(gauKsize, -1, CV_32F);
	cv::sepFilter2D(dXX, dXX, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dYY, dYY, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dXY, dXY, CV_32F, gau.t(), gau);

	Mat maskimage=  createmaskimage(image,dXX, dYY,dXY);// creates thresholded image of all the possible dendrites

	//create high intensity thresholded image to bin dendrites into developed and less developed dendrites
	Mat highIntredthreshimg(image.rows, image.cols, CV_8U);
	cv::inRange(org, cv::Scalar(0, 0, 180), cv::Scalar(30, 30, 255), highIntredthreshimg);
	dilate(highIntredthreshimg, highIntredthreshimg, Mat());
	erode(highIntredthreshimg, highIntredthreshimg, Mat());

	//----------Inside image
	int countofdendrites = 0;
	int developed = 0;
	int lessdeveloped = 0;
	for (int i = 1; i < image.rows; i++) {
		for (int j = 1; j < image.cols; j++) {
			hessian_matrix.at<float>(Point(0, 0)) = dXX.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 1)) = dYY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(0, 1)) = dXY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 0)) = dXY.at<float>(Point(j, i));

			// find eigen values of hessian matrix /* Larger  eigenvalue  show the  direction  of  intensity change, while  smallest  eigenvalue  show  the  direction  of  vein.*/
			eigen(hessian_matrix, eigenvalues, eigenvec);
			//find all sets of dendrites (horizontal an vertical)
/*Main Condition*/if (abs(eigenvalues[0])<5 && abs(eigenvalues[1])>6)
			{		
					checkimg = cv::Scalar(0, 0, 0);
					overlapimage = cv::Scalar(0, 0, 0);
					dendritetips = cv::Scalar(0, 0, 0);
					overlapbinimage = cv::Scalar(0, 0, 0);
					
/*Condition 1*/		if ((abs(eigenvec.at<float>(Point(0, 0))) > 0) && (abs(eigenvec.at<float>(Point(0, 0)) < 0.4)))	// for vertical dendrites
						{
							circle(checkimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 3, 8, 0);//orange
						}
/*Condition 2*/		else if ((abs(eigenvec.at<float>(Point(0, 0))) > 0.5) && (abs(eigenvec.at<float>(Point(0, 0)) < 1.2))) // for horizontal dendrites
						{
							circle(checkimg, cv::Point(j, i), 1, cv::Scalar(255, 255,255), 3, 8, 0);//blue
						}
					else{}

					bitwise_and(checkimg, maskimage, overlapimage);// to detct region of overlap inorder to find dendrite tips/start points 
					///gives number of dendrites
					/*if (countNonZero(overlapimage)>25)
					{
						countofdendrites++;
						circle(org, cv::Point(j, i), 1, cv::Scalar(255, 125, 0), 3, 8, 0);//blue;
					}*/

					// classifies dendrites as developd and under dveloped based on overlap of dendrite tips with high intensity red images
					if (countNonZero(overlapimage)>25)
					{
					countofdendrites++;
					circle(org, cv::Point(j, i), 1, cv::Scalar(255, 125, 0), 3, 8, 0);//blue;
					circle(dendritetips, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 3, 8, 0);//blue;
					bitwise_and(dendritetips, highIntredthreshimg, overlapbinimage);
					if (countNonZero(overlapbinimage) > 5)
					{
						developed++;
						circle(org, cv::Point(j, i), 1, cv::Scalar(255, 255, 0), 3, 8, 0);//blue;
					}
					else
						lessdeveloped++;
					}
			}
		}
	}
	cout << "Dendrite count:  " << countofdendrites << " Developed:  " << developed << " Less Developed:  " << lessdeveloped << endl;
	imshow("show", org);
	imwrite("dendritecount.png", org);
}

int main()
{
	Mat im = imread("test.png");
	cvtColor(im, im, CV_BGR2GRAY);

/*CALL*/filterHessian(im); // finds the dendrites in the image
	waitKey(0);

	return 0;


}
/*//if (abs(eigenvec.at<float>(Point(0, 0))) / abs(eigenvec.at<float>(Point(0, 1)))<0.3)//ratio gives direction of dendrite 
//			cout << eigenvec.at<float>(Point(0, 0)) << "," << eigenvec.at<float>(Point(1, 0)) << " " << eigenvec.at<float>(Point(0, 1)) << " " << eigenvec.at<float>(Point(1,1)) <<endl;
//			cout << "Eigen values: " << abs(eigenvalues[0]) << ", " << abs(eigenvalues[1]) << endl;
/if (abs(eigenvec.at<float>(Point(0, 0))) / abs(eigenvec.at<float>(Point(0, 1)))<0.3)//ratio gives direction of dendrite
*/
