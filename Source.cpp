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



using namespace cv;
using namespace std;


vector <Mat> enhance;

void enhanceImage(vector <Mat> stackim, vector <Mat> & enhance)
 {
	 for (int i = 0; i < stackim.size(); i++)
	 {
		 Mat img = stackim[i]; Mat gry;
		// cvtColor(img, img, CV_BGR2GRAY);
		 // Normalize the image
		 cv::Mat normalized;
		 
		 cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		 cvtColor(normalized, gry, CV_BGR2GRAY);
		 
//-----------------------------------------------------------------------------------------------------------------------------------
		 // Enhance the image using Gaussian blur and thresholding
		 cv::Mat enhanced;
		 if (i == 0 || (i % 3 == 0))// channel blue
		 {
			 // Enhance the blue channel
			 //cv::threshold(normalized, enhanced, 19, 255, cv::THRESH_BINARY);
			 // cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
			 cv::inRange(normalized, cv::Scalar(130, 0, 0), cv::Scalar(255, 20,20), enhanced);
			 enhance.push_back(enhanced);
			 string name = format("enh_%d.jpg", i);
			 imwrite(name, enhanced);
		 }
		 if (i == 1 || (i % 3 == 1))// channel red
		  {
			 // Enhance the red channel
			// cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
			  cv::inRange(normalized, cv::Scalar(0,0,30), cv::Scalar(30,30,255), enhanced);
			 enhance.push_back(enhanced);
			// string name = format("enh_%d.jpg", i);
			// imwrite(name, enhanced);
		  } 

		 if (i == 2 || (i % 3 == 2))// channel green
		 {
			 // Enhance the green channel
			 //cv::threshold(normalized, enhanced, 20, 255, cv::THRESH_BINARY);
			 cv::inRange(normalized, cv::Scalar(0, 100, 0), cv::Scalar(20,255, 20), enhanced);
			 enhance.push_back(enhanced);
			// string name = format("enh_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 
//----------------------------------------------------------------------------------------------------------------------------------

		 if (i == 2 || (i % 3 == 2))// channel green low
		 {
			 // Enhance the green low channel
			 cv::inRange(normalized, cv::Scalar(0, 25, 0), cv::Scalar(10, 80, 10), enhanced);
			 //string name = format("grlow_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 


	     if (i == 2 || (i % 3 == 2))// channel green high
		 {
			 // Enhance the green high channel
			 cv::inRange(normalized, cv::Scalar(0, 90, 0), cv::Scalar(10, 255,10), enhanced);
			// string name = format("grhigh_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 

		 if (i == 1 || (i % 3 == 1))// channel red low
		 {
			 // Enhance the red low channel
			 cv::inRange(normalized, cv::Scalar(0, 0, 25), cv::Scalar(10,10,30), enhanced);
			// string name = format("redl_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 

		 if (i == 1 || (i % 3 == 1))// channel red medium
		 {
			 // Enhance the red medium channel
			 cv::inRange(normalized, cv::Scalar(0,0,50), cv::Scalar(10, 10,80), enhanced);
			 //string name = format("rdmed_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 

		 if (i == 1 || (i % 3 == 1))// channel red high
		 {
			 // Enhance the red high channel
			 cv::inRange(normalized, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), enhanced);
			// string name = format("rdhih_%d.jpg", i);
			// imwrite(name, enhanced);
		 } 

		 
	 }
}


void cellcount(vector <Mat> stackim,vector <Mat> enhance)
{
	
	
	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<vector<Point>> blucontours;
	vector<vector<Point>> nucontours;
	vector<vector<Point>> astcontours;
	int tau;


	// draws and saves blue contours
	for (int i = 0; i < enhance.size(); i++)
	{
		blucontours.clear();
		astcontours.clear();
		nucontours.clear();
		
		if (i == 0 || (i % 3 == 0))// channel blue
		{
			findContours(enhance[i], contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // find all black contours

			// FINDING CELLS: and saving the contours that are above a certain area
			for (int d = 0; d < contours.size(); d++)  
			{
				float chaid = contourArea(contours[d], false);
				if (chaid > 300)
				{
					drawContours(stackim[i], contours, d, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
					blucontours.push_back(contours[d]);
				}
			}
			tau = i + 2; // decides image index

			// FIND NUERAL CELLS: draw bounding rectangle around blue contours
			// Check for presence of dendrites(green) around the blue contours
			vector<Rect> brect(blucontours.size());
			for (int io = 0; io < blucontours.size(); io++)
			{
				brect[io] = boundingRect(Mat(blucontours[io]));

				Mat image = enhance[tau];
				Mat image_roi = image(brect[io]);// creating a new image from roi
				int count = countNonZero(image_roi);
				if (count > 500)
					nucontours.push_back(blucontours[io]);
				
			}


			// FIND ASTROCYTES: large, oval & textured
			// Checking for aspect ratio (Oval), area and presense of large number of black pixels(textured)
			vector<Rect> arect(blucontours.size()); vector< float> asprat(blucontours.size());
			for (int ia = 0; ia < blucontours.size(); ia++)
			{
				cout << "again" << endl;

				float chaid = contourArea(blucontours[ia], false);
				arect[ia] = boundingRect(Mat(blucontours[ia]));// bpunding rect
				asprat[ia] = arect[ia].height / arect[ia].width;// aspect ratio
				Mat imagee = enhance[i];
				Mat image_roi = imagee(arect[ia]);// creating a new image from roi
				int count = (image_roi.rows*image_roi.cols) - countNonZero(image_roi);// counting number of black pixels
				if ((count>500) && (asprat[ia] <= 0.5) && (chaid > 1000))
				{
					cout << "hello" << endl; astcontours.push_back(blucontours[ia]);
				}

			}
			for (int du = 0; du < astcontours.size(); du++)  // finding and saving the contours that are above a certain area
				drawContours(stackim[i], astcontours, du, Scalar(0, 255, 255), 2, 8, vector<Vec4i>(), 0, Point());

			string name = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Test\\cell_%d.jpg", i);
			imwrite(name, stackim[i]);

		}

	}
}




void main()
{
	std::vector<cv::Mat> channel; int valred = 255, valblue = 255, valgreen = 255; Mat im_color;
	vector<Mat> stackim; 
	// reading 16 bit image
	string raw_path = "C:\\Users\\VIneeta\\Downloads\\Eliliy_NewData_May20\\2016-05-10T175424-0400[3087]\\002002-7.tif";  
	cv::imreadmulti(raw_path, channel, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cout << "channels : " << channel.size() << endl; //15
	for (unsigned int i = 0; i < channel.size(); i++) 
	{
		Mat src;
		channel[i].convertTo(src, CV_8U);
		Mat empty_image = Mat::zeros(src.rows, src.cols, CV_8UC1);
		Mat result_blue(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!
		Mat result_green(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!
		Mat result_red(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!
		


		if (i== 0 || (i % 3==0))// channel blue
		{
			/*if I have 8bit gray, and create a new empty 24bit RGB, I can copy the entire 8bit gray into one of the BGR channels (say, R), 
			leaving the others black, and that effectively colorizes the pixels in a range of red. 
			Similar, if the user wants to make it, say, RGB(80,100,120) then I can set each of the RGB channels to the source grayscale 
			intensity multiplied by (R/255) or (G/255) or (B/255) respectively. This seems to work visually. 
			It does need to be a per-pixel operation though cause the color applies only to a user-defined range of grayscale intensities.*/
			Mat in1[] = { src, empty_image, empty_image };
			int from_to1[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in1, 3, &result_blue, 1, from_to1, 3);
			stackim.push_back(result_blue);
			//string name = format("Vin_%d.jpg", i);
			//imwrite(name, result_blue);
		}
		if (i == 2 || (i % 3 == 2))// channel red
		{
			Mat in2[] = { empty_image, src, empty_image };
			int from_to2[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in2, 3, &result_red, 1, from_to2, 3);
			stackim.push_back(result_red);
			//string name = format("Vin_%d.jpg", i);
			//imwrite(name, result_red);
		}
		if (i == 1 || (i % 3 == 1))// channel green
		{
			Mat in3[] = { empty_image, empty_image, src };
			int from_to3[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in3, 3, &result_green, 1, from_to3, 3);
			stackim.push_back(result_green);
			//string name = format("Vin_%d.jpg", i);
			//imwrite(name, result_green);
		}
		
		
	}
	
	enhanceImage(stackim, enhance);
	cellcount(stackim,enhance);
	
	
}
		