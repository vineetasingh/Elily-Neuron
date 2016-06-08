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

vector <Mat> enhance;
// make countlrsum 0 after every z image is processed 

void redcount(vector<vector<Point>> blucontoursr, int j)
{
	int countlrsum = 0, countrmsum = 0, countrhsum = 0;
	int grim = j + 1;
	//  draw bounding rectangle around blue contours
	// Check for presence of reds around the blue contours
	vector<Rect> brect(blucontoursr.size());
	Mat rl, rm, rh;
	for (int io = 0; io < blucontoursr.size(); io++)
	{
		brect[io] = boundingRect(Mat(blucontoursr[io]));

		Mat image = enhance[grim];
		Mat image_roi = image(brect[io]);// creating a new image from roi
		cv::inRange(image_roi, cv::Scalar(0, 0, 25), cv::Scalar(10, 10, 30), rl);// channel red low
		int countlr = countNonZero(rl);
		countlrsum = countlr + countlrsum;
		cv::inRange(image_roi, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 80), rm);// channel red medium
		int countrm = countNonZero(rm);
		countrmsum = countrm + countrmsum;
		cv::inRange(image_roi, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), rh);// channel red high
		int countrh = countNonZero(rh);
		countrhsum = countrh + countrhsum;

	}


	countlrsum = countlrsum / blucontoursr.size();
	countrmsum = countrmsum / blucontoursr.size();
	countrhsum = countrhsum / blucontoursr.size();
}

void calcCellMetrics(string imname, int i, vector < vector<Point>> blucontours, ofstream &myfile)
{
	// average area of blue contours
	float bltotarea = 0, blavgarea; int lowcount = 0, medcount = 0, hihcount = 0;
	for (int m = 0; m < blucontours.size(); m++)
	{
		float chaid = contourArea(blucontours[m], false);
		if (chaid <= 1000)
			lowcount = lowcount + 1;
		if (chaid > 1000 && chaid <= 3500)
			medcount = medcount + 1;
		if (chaid > 3500)
			hihcount = hihcount + 1;

		bltotarea = bltotarea + chaid;
	}
	if (blucontours.size() != 0)
		blavgarea = bltotarea / blucontours.size();
	else
		blavgarea = 0;

	// average aspect ratio of blue contours
	float bltotasp = 0, blavgasp;
	for (int m = 0; m < blucontours.size(); m++)
	{
		vector<Rect> brect(blucontours.size());
		float asprat;
		brect[m] = boundingRect(Mat(blucontours[m]));// bpunding rect
		asprat = brect[m].height / brect[m].width;// aspect ratio
		bltotasp = bltotasp + asprat;
	}
	if (blucontours.size() != 0)
		blavgasp = bltotasp / blucontours.size();
	else
		blavgasp = 0;

	// average diameter of blue contours
	float bltotdia = 0, blavgdia;
	vector<Point2f>center(blucontours.size());
	vector<float>radius(blucontours.size());
	for (int m = 0; m < blucontours.size(); m++)
	{
		minEnclosingCircle((Mat)blucontours[m], center[m], radius[m]);
		bltotdia = bltotdia + (2 * radius[m]);
	}
	if (blucontours.size() != 0)
		blavgdia = bltotdia / blucontours.size();
	else
		blavgdia = 0;

	myfile << "," << "cellmetrics" << " ," << imname << "," << blucontours.size() << "," << hihcount << "," << medcount << "," << lowcount << "," << blavgarea << "," << blavgasp << "," << blavgdia;


}


// cell metrics for every plane in an image (5 z-planes)
void cellmetrics(string imname, int i, vector < vector<Point>> blucontours, vector < vector<Point>> nucontours, vector <vector<Point>> astcontours, ofstream &myfile)
{
	calcCellMetrics(imname, i, blucontours, myfile);
	calcCellMetrics(imname, i, nucontours, myfile);
	calcCellMetrics(imname, i, astcontours, myfile);

}
// finds the number of low/med/high synapses near an astrocyte and neural cell
void synapcalc(string imname, int i, vector <Mat> stackim, vector<vector<Point>> nucontours)
{
	Mat redres;
	Mat img = stackim[i];
	cv::Mat normalized;
	vector <Mat> redlow, redmed, redhigh;
	cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);


	// counting the number of red  intensity pixels around a nuclei cell
	int totrl = 0, avglo;
	if (i == 1 || (i % 3 == 1))// channel red low
	{
		// Enhance the red low channel
		cv::inRange(normalized, cv::Scalar(0, 0, 25), cv::Scalar(10, 10, 30), redres);
		vector<Rect> brect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			brect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(brect[io]);// creating a new image from roi
			int count = countNonZero(image_roi);
			totrl = totrl + count;
		}
		if (nucontours.size() != 0)
			avglo = totrl / nucontours.size();
		else
			avglo = 0;
	}

	// counting the average number of red medium intensity pixels around a nuclei cell
	int totrm = 0, avgme;
	if (i == 1 || (i % 3 == 1))// channel red medium
	{
		// Enhance the red medium channel
		cv::inRange(normalized, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 80), redres);
		vector<Rect> bmrect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			bmrect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(bmrect[io]);// creating a new image from roi
			int countm = countNonZero(image_roi);
			totrm = totrm + countm;
		}
		if (nucontours.size() != 0)
			avgme = totrm / nucontours.size();
		else
			avgme = 0;

	}

	// counting the average number of red high intensity pixels around a nuclei cell
	int totrh = 0, avglh;
	if (i == 1 || (i % 3 == 1))// channel red high
	{
		// Enhance the red high channel
		cv::inRange(normalized, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), redres);
		vector<Rect> bhrect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			bhrect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(bhrect[io]);// creating a new image from roi
			int counth = countNonZero(image_roi);
			totrh = totrh + counth;
		}
		if (nucontours.size() != 0)
			avglh = totrh / nucontours.size();
		else
			avglh = 0;

	}

	myfile << "," << "synapse cal" << " ," << imname << "," << avglo << "," << avgme << "," << avglh;

}

void enhanceImage(string imname, vector <Mat> stackim, vector <Mat> & enhance)
{
	enhance.clear();
	for (int i = 0; i < stackim.size(); i++)
	{
		Mat img = stackim[i]; Mat gry;
		// cvtColor(img, img, CV_BGR2GRAY);
		// Normalize the image
		cv::Mat normalized;
		vector <Mat> redlow, redmed, redhigh;
		cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cvtColor(normalized, gry, CV_BGR2GRAY);

		//-----------------------------------------------------------------------------------------------------------------------------------
		// Enhance the image using Gaussian blur and thresholding
		cv::Mat enhanced;
		if (i == 0 || (i % 3 == 0))// channel blue
		{
			// Enhance the blue channel
			cv::inRange(normalized, cv::Scalar(130, 0, 0), cv::Scalar(255, 20, 20), enhanced);
			enhance.push_back(enhanced);
			//string name = format("enh_%d.jpg", i);
			//imwrite(name, enhanced);
		}
		if (i == 1 || (i % 3 == 1))// channel red
		{
			// Enhance the red channel
			// cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
			cv::inRange(normalized, cv::Scalar(0, 0, 30), cv::Scalar(30, 30, 255), enhanced);
			enhance.push_back(enhanced);
			// string name = format("enh_%d.jpg", i);
			// imwrite(name, enhanced);
		}

		if (i == 2 || (i % 3 == 2))// channel green
		{
			// Enhance the green channel
			//cv::threshold(normalized, enhanced, 20, 255, cv::THRESH_BINARY);
			cv::inRange(normalized, cv::Scalar(0, 100, 0), cv::Scalar(20, 255, 20), enhanced);
			enhance.push_back(enhanced);
			// string name = format("enh_%d.jpg", i);
			// imwrite(name, enhanced);
		}

		//----------------------------------------------------------------------------------------------------------------------------------
		/*
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
		redlow.push_back(enhanced);
		// string name = format("redl_%d.jpg", i);
		// imwrite(name, enhanced);
		}
		if (i == 1 || (i % 3 == 1))// channel red medium
		{
		// Enhance the red medium channel
		cv::inRange(normalized, cv::Scalar(0,0,50), cv::Scalar(10, 10,80), enhanced);
		redmed.push_back(enhanced);
		//string name = format("rdmed_%d.jpg", i);
		// imwrite(name, enhanced);
		}
		if (i == 1 || (i % 3 == 1))// channel red high
		{
		// Enhance the red high channel
		cv::inRange(normalized, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), enhanced);
		redhigh.push_back(enhanced);
		// string name = format("rdhih_%d.jpg", i);
		// imwrite(name, enhanced);
		}
		*/

	}

	cout << "enhance= " << enhance.size() << endl;
}

// counts number of astrocytes, cells and nueral cells
void cellcount(string imname, vector <Mat> stackim, vector <Mat> enhance)
{


	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<vector<Point>> blucontours;
	vector<vector<Point>> nucontours;
	vector<vector<Point>> astcontours;
	int tau;
	Mat drawimg;


	// draws and saves blue contours
	for (int i = 0; i < enhance.size(); i++) // for a particular z layer in an image
	{
		//string imm = format("%s_z%d", imname, i);
		string imm = format("%s_z%d", imname.c_str(), i);
		blucontours.clear();
		astcontours.clear();
		nucontours.clear();

		if (i == 0 || (i % 3 == 0))// channel blue
		{
			findContours(enhance[i], contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // find all blue contours

			// FINDING CELLS: and saving the contours that are above a certain area
			for (int d = 0; d < contours.size(); d++)
			{
				float chaid = contourArea(contours[d], false);
				if (chaid > 300)
				{
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

			for (int nu = 0; nu < nucontours.size(); nu++)
				drawContours(stackim[i], nucontours, nu, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());



			// FIND ASTROCYTES: large, oval & textured
			// Checking for aspect ratio (Oval), area and presense of large number of black pixels(textured)
			vector<Rect> arect(blucontours.size()); vector< float> asprat(blucontours.size());
			for (int ia = 0; ia < blucontours.size(); ia++)
			{
				float chaid = contourArea(blucontours[ia], false);
				arect[ia] = boundingRect(Mat(blucontours[ia]));// bpunding rect
				asprat[ia] = arect[ia].height / arect[ia].width;// aspect ratio
				Mat imagee = enhance[i];
				Mat image_roi = imagee(arect[ia]);// creating a new image from roi
				int count = (image_roi.rows*image_roi.cols) - countNonZero(image_roi);// counting number of black pixels
				if ((count>500) && (asprat[ia] <= 0.5) && (chaid > 1000))
					astcontours.push_back(blucontours[ia]);
			}
			for (int du = 0; du < astcontours.size(); du++)  // finding and saving the contours that are above a certain area
				drawContours(stackim[i], astcontours, du, Scalar(0, 255, 255), 2, 8, vector<Vec4i>(), 0, Point());


			string name = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Test\\%s_z%d_mod.jpg", imname.c_str(), i);
			imwrite(name, stackim[i]);

			cellmetrics(imm, i, blucontours, nucontours, astcontours, myfile); // for the particular z-layer


			int beta = i + 1;
			synapcalc(imm, beta, stackim, nucontours); // calculating presence of red aroung neural cells
			synapcalc(imm, beta, stackim, astcontours);// calculating count of red around astrocytes
			myfile << endl;
		}

	}
}


void main()
{
	std::vector<cv::Mat> channel; int valred = 255, valblue = 255, valgreen = 255; Mat im_color;
	vector<Mat> stackim; string raw_path;
	myfile.open("EliliyMetrics.csv");
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

				enhanceImage(imname, stackim, enhance);
				cellcount(imname, stackim, enhance);
			}
		}
	}


	waitKey(0);
	myfile.close();
}
