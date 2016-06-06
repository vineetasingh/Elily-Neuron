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

ofstream myfile;
void reddetect(Mat im, Mat & redlow, Mat & redmed, Mat & redhigh)
{

	cv::Mat normalizedl; Mat enhancedl;
	cv::normalize(im, normalizedl, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red low channel
	cv::inRange(normalizedl, cv::Scalar(0, 0, 25), cv::Scalar(10, 10, 30), enhancedl);
	redlow = enhancedl.clone();

	cv::Mat normalizedm; Mat enhancedm;
	cv::normalize(im, normalizedm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red medium channel
	cv::inRange(normalizedm, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 80), enhancedm);
	redmed = enhancedm.clone();


	cv::Mat normalizedh; Mat enhancedh;
	cv::normalize(im, normalizedh, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red high channel
	cv::inRange(normalizedh, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), enhancedh);
	redhigh = enhancedh.clone();
	//imshow("redhigh", redhigh);

}
void dendritedetect(string imname,Mat img, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::inRange(img, cv::Scalar(0, 200, 0), cv::Scalar(10, 255, 10), img);
	Mat thresh = img.clone();

	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp(img.size(), CV_8UC1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	bool done;
	do
	{
		cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
		cv::bitwise_not(temp, temp);
		cv::bitwise_and(img, temp, temp);
		//	cout << "img size  " << img.size() << endl << "temp.size  " << temp.size() << endl << " skel size  " << skel.size() << endl;
		cv::bitwise_or(skel, temp, skel);
		cv::erode(img, img, element);

		double max;
		cv::minMaxLoc(img, 0, &max);
		done = (max == 0);
	} while (!done);

	/*for (int i =0; i < 1000; i++)
	{
	cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
	cv::bitwise_not(temp, temp);
	cv::bitwise_and(img, temp, temp);
	//	cout << "img size  " << img.size() << endl << "temp.size  " << temp.size() << endl << " skel size  " << skel.size() << endl;
	cv::bitwise_or(skel, temp, skel);
	}*/

	//dilate(skel, skel, element);
	//Mat blured = cv::pyrMeanShiftFiltering(skel,skel, 3, 9);

	//cv::imwrite("Skeleton.png", skel);
	//Mat src = skel;
	Mat dst, cdst;

	cvtColor(skel, cdst, CV_GRAY2BGR);

#if 0
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
#else
	vector<Vec4i> lines; int totl = 0, totm = 0, toth = 0; float totwid = 0;
	vector<RotatedRect> minRect(lines.size());
	GaussianBlur(skel, skel, Size(1, 1), 2.0, 2.0);
	HoughLinesP(skel, lines, 5, CV_PI / 135, 70, 50, 10);
	int smalwid = 0, midwid = 0, larwid = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
		//cout << "lines" << Point(l[0], l[1]) << "   " << Point(l[2], l[3]);
		//minRect[i] = minAreaRect(Mat(lines[i]));	
		//cv::rectangle(cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 255));
		Point a = (Point(l[0], l[1])); Point b = (Point(l[2], l[3]));
		int h = 10;  double w = cv::norm(a - b);
		cv::Rect bRect(a, b);
		cv::rectangle(cdst,a,b,cv::Scalar(255, 0, 255), 1, 8);
		Mat lroi = redlow(bRect);// creating a new image from roi
		int lcount = countNonZero(lroi);
		Mat mroi = redmed(bRect);// creating a new image from roi
		int mcount = countNonZero(mroi);
		Mat hroi = redhigh(bRect);// creating a new image from roi
		int hcount = countNonZero(hroi);
		Mat rect_roi = thresh(bRect);// creating a new image from roi
		int wcount = countNonZero(rect_roi);
		if (w > 0)
		{
			int widthden = wcount / w; // no of whote pixels in thresholded image/ length of synapse
			if (widthden < 10)
				smalwid++;
			if (widthden >= 10 && widthden < 20)
				midwid++;
			if (widthden >= 20)
				larwid++;
		}
		
		totl = lcount + totl;
		totm = mcount + totm;
		toth = hcount + toth;
		totwid = totwid + w;
	}
#endif
	int Ftotl, Ftotm, Ftoth, Ftotwid;

	if (lines.size() > 0)
	{
		 Ftotl = totl / lines.size();
		 Ftotm = totm / lines.size();
		 Ftoth = toth / lines.size();
		 Ftotwid = totwid / lines.size();
	}
	else
	{
		Ftotl = 0;
		Ftotm = 0;
		Ftoth = 0;
		Ftotwid = 0;
	}
	myfile <<imname<<","<< lines.size() << ", " << Ftotl << "," << Ftotm << "," << Ftoth<<","<< Ftotwid<<","<<smalwid<<","<<midwid<<","<<larwid<< endl;
	//imshow("source", skel);
	//imshow("detected lines", cdst);

}
// for each image (with all z stack layers) - finds the redlow, redmed and redhigh image



void main()
{
	std::vector<cv::Mat> channel; int valred = 255, valblue = 255, valgreen = 255; Mat im_color;
	vector<Mat> stackim; string raw_path;
	myfile.open("DendriteMetrics.csv");
	// reading 16 bit image
	for (int n1 = 3; n1 <= 7; n1++)
	{
		for (int n2 = 2; n2 <= 11; n2++) //11
		{
			for (int n3 = 1; n3 <= 35; n3++)//35
			{
				channel.clear();
				stackim.clear();

				if (n2 < 10)
					raw_path = format("C:\\Users\\VIneeta\\Downloads\\Eliliy_NewData_May20\\2016-05-10T175424-0400[3087]\\00%d00%d-%d.tif", n1, n2, n3);  // 002002-1
				else
					raw_path = format("C:\\Users\\VIneeta\\Downloads\\Eliliy_NewData_May20\\2016-05-10T175424-0400[3087]\\00%d0%d-%d.tif", n1, n2, n3);  // 002010-1

				string imname = format("00%d0%d-%d.tif", n1, n2, n3);

				cout << format("Processing %s", imname.c_str()) << endl;



				cv::imreadmulti(raw_path, channel, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
				if (channel.empty())
					cout << "On no!!" << endl;
				for (unsigned int i = 0; i < channel.size(); i++)
				{

					Mat src;
					channel[i].convertTo(src, CV_8U);
					Mat empty_image = Mat::zeros(src.rows, src.cols, CV_8UC1);
					Mat result_blue(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!
					Mat result_green(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!
					Mat result_red(src.rows, src.cols, CV_8UC3); // notice the 3 channels here!



					if (i == 0 || (i % 3 == 0))// channel blue
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
					if (i == 2 || (i % 3 == 2))// channel green
					{
						Mat in2[] = { empty_image, src, empty_image };
						int from_to2[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in2, 3, &result_red, 1, from_to2, 3);
						stackim.push_back(result_red);
						//string name = format("Vin_%d.jpg", i);
						//imwrite(name, result_red);
					}
					if (i == 1 || (i % 3 == 1))// channel red
					{
						Mat in3[] = { empty_image, empty_image, src };
						int from_to3[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in3, 3, &result_green, 1, from_to3, 3);
						stackim.push_back(result_green);
						//string name = format("Vin_%d.jpg", i);
						//imwrite(name, result_green);
					}


				}
				Mat redlow, redmed, redhigh;


				for (int i = 0; i < stackim.size(); i++)
				{
					if (i == 2 || (i % 3 == 2))// channel green
					{

						reddetect(stackim[i-1], redlow, redmed, redhigh);
						dendritedetect(imname, stackim[i], redlow, redmed, redhigh, myfile);
					}
				}

			}
		}
	}

	waitKey(0);
	myfile.close();
}