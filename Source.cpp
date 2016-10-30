// checked code dendrite last submittes 21oct, 2016

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
#include <vector>	



using namespace cv;
using namespace std;

ofstream myfile;
RNG rng;
vector <Mat> enhance;

const int NeuralTHRESH = 2000; //threshold is the amount of green pixels in the bounding rectangle around blucontours
const float tune = 0.1;
const int houghthresh = 275;




// make countlrsum 0 after every z image is processed 

// changes contrast and brightness of each z slice for each channel

Mat changeimg(Mat image, float alpha, float beta)
{
	alpha = alpha / 10;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	Mat new_image = Mat::zeros(image.size(), CV_16UC3);


	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				//cout << new_image.at<Vec3s>(y,x)[c] << endl;

				new_image.at<Vec3s>(y, x)[c] = (alpha*(image.at<Vec3s>(y, x)[c]) + beta);
			}
		}
	}
	dilate(new_image, new_image, Mat());
	//	imshow("New Image", new_image);
	return new_image;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float vari(int a, int b, int c, int d)
{
	float mean = (a + b + c + d) / 4;
	float var = (((a - mean)*(a - mean)) + ((b - mean)*(b - mean)) + ((c - mean)*(c - mean)) + ((d - mean)*(d - mean))) / 4;
	return var;
}

//----------------------Finds count of L, M, H intensity synapses around // LMH syn or dendrites --------------------------------
void neighboursyn(Mat redlow, Mat redmed, Mat redhigh, vector<int> CoordinatesX, vector<int> CoordinatesY, ofstream & myfile, int w, int h)
{
	unsigned int totl = 0, totm = 0, toth = 0;
	double totlvar = 0, totmvar = 0, tothvar = 0;
	int lcount1; int mcount1; int hcount1; int lcount2; int mcount2; int hcount2; int lcount3; int mcount3; int hcount3; int lcount4; int mcount4; int hcount4;

	for (int i = 0; i < CoordinatesX.size(); i++)
	{

		Point a = Point(CoordinatesX[i], CoordinatesY[i]);
		lcount1 = 0; mcount1 = 0; hcount1 = 0; lcount2 = 0; mcount2 = 0; hcount2 = 0; lcount3 = 0; mcount3 = 0; hcount3 = 0; lcount4 = 0; mcount4 = 0; hcount4 = 0;


		if (((a.x) + w  < redlow.rows) && ((a.x) - w > 0) && ((a.y) + h < redlow.cols) && ((a.y) - h> 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				int px = (a.x + (0.5*w)); int py = (a.y + (h / 2));
				//if (px >= 0 && py >= 0 && myrect.width + px < redlow.cols && myrect.height + py < redlow.rows)
				Mat lroi = redlow(myrect);// creating a new image from roi of redlow
				cv::Rect top_left(cv::Point(0, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(lroi.size().height / 2, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(lroi.size().height / 2, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = lroi(top_right);
				Image2 = lroi(top_left);
				Image3 = lroi(bottom_right);
				Image4 = lroi(bottom_left);

				lcount1 = countNonZero(Image1);
				lcount2 = countNonZero(Image2);
				lcount3 = countNonZero(Image3);
				lcount4 = countNonZero(Image4);
				float lvar = vari(lcount1, lcount2, lcount3, lcount4);
				totlvar = totlvar + lvar;
			}

		}

		//cout <<"totl:  "<< totl<< endl;



		if (((a.x) + w + 20 < redmed.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redmed.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed

				cv::Rect top_left(cv::Point(0, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(mroi.size().height / 2, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(mroi.size().height / 2, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = mroi(top_right);
				Image2 = mroi(top_left);
				Image3 = mroi(bottom_right);
				Image4 = mroi(bottom_left);
				mcount1 = countNonZero(Image1);
				mcount2 = countNonZero(Image2);
				mcount3 = countNonZero(Image3);
				mcount4 = countNonZero(Image4);
				float mvar = vari(mcount1, mcount2, mcount3, mcount4);
				totmvar = totmvar + mvar;
			}
		}


		if (((a.x) + w + 20 < redhigh.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redhigh.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed


				cv::Rect top_left(cv::Point(0, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(hroi.size().height / 2, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(hroi.size().height / 2, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = hroi(top_right);
				Image2 = hroi(top_left);
				Image3 = hroi(bottom_right);
				Image4 = hroi(bottom_left);
				hcount1 = countNonZero(Image1);
				hcount2 = countNonZero(Image2);
				hcount3 = countNonZero(Image3);
				hcount4 = countNonZero(Image4);
				float hvar = vari(hcount1, hcount2, hcount3, hcount4);
				tothvar = tothvar + hvar;

			}
		}


	}
	if (CoordinatesX.size()>0)
		myfile << "neignborsyn" << "," << totlvar / CoordinatesX.size() << "," << totmvar / CoordinatesX.size() << "," << tothvar / CoordinatesX.size() << ",";
	else
		myfile << "neignborsyn" << "," << 0 << "," << 0 << "," << 0 << ",";
}
//calculates avg count of Low, Medium and High int synapses around L/M/H intensity synapse points
void aroundsyncalc(Mat redlow, Mat redmed, Mat redhigh, vector<int> CoordinatesX, vector<int> CoordinatesY, ofstream & myfile)
{
	unsigned int totl = 0, totm = 0, toth = 0; int lcount; Mat dst; int mcount; int hcount; int w = 40, h = 40;

	for (int i = 0; i < CoordinatesX.size(); i++)
	{

		Point a = Point(CoordinatesX[i], CoordinatesY[i]);
		lcount = 0; mcount = 0; hcount = 0;

		if (((a.x) + w + 4 < redlow.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redlow.cols) && ((a.y) - h > 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				Mat lroi = redlow(myrect);// creating a new image from roi of redmed
				lcount = countNonZero(lroi);
				totl = totl + lcount;
				//cout <<"totl:  "<< totl<< endl;

			}
		}


		if (((a.x) + w + 4 < redmed.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redmed.cols) && ((a.y) - h > 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed
				mcount = countNonZero(mroi);
				totm = totm + mcount;
			}
		}


		if (((a.x) + w + 4 < redhigh.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redhigh.cols) && ((a.y) - h > 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed
				hcount = countNonZero(hroi);
				toth = toth + hcount;
			}
		}


	}



	//myfile << "no of low pts" << CoordinatesX.size()<< endl;
	// avg count of Low, Medium and High int synapses around L/M/H intensity synapse points
	if (CoordinatesX.size() != 0)
		myfile << "," << "aroundsyn" << "," << totl / CoordinatesX.size() << "," << totm / CoordinatesX.size() << "," << toth / CoordinatesX.size() << ",";
	else
		myfile << "," << "aroundsyn" << "," << 0 << "," << 0 << "," << 0 << ",";
}

//-- - [7]----
void redinf(string imname, Mat im, ofstream &myfile)
{
	Mat enhancedl; vector< int> CoordinateLX; vector< int> CoordinateLY; vector< int> CoordinateMX; vector< int> CoordinateMY;
	vector< int> CoordinateHX; vector< int> CoordinateHY;

	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), enhancedl);
	erode(enhancedl, enhancedl, Mat());
	Mat redlow = enhancedl.clone();
	Mat nonZeroCoordinatesL;
	findNonZero(enhancedl, nonZeroCoordinatesL);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), enhancedm);
	erode(enhancedm, enhancedm, Mat());
	Mat redmed = enhancedm.clone();
	Mat nonZeroCoordinatesM;
	findNonZero(enhancedm, nonZeroCoordinatesM);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), enhancedh);
	erode(enhancedm, enhancedm, Mat());
	Mat redhigh = enhancedh.clone();
	//imshow("redhigh", redhigh);
	Mat nonZeroCoordinatesH;
	findNonZero(enhancedh, nonZeroCoordinatesH);


	for (int i = 0; i < nonZeroCoordinatesL.total(); i++)
	{
		CoordinateLX.push_back(((nonZeroCoordinatesL.at<Point>(i).x)));
		CoordinateLY.push_back(((nonZeroCoordinatesL.at<Point>(i).y)));
		//cout << "Zero#" << i << ": " << nonZeroCoordinatesL.at<Point>(i).x << ", " << nonZeroCoordinatesL.at<Point>(i).y << endl;
	}
	for (int i = 0; i < nonZeroCoordinatesM.total(); i++)
	{
		CoordinateMX.push_back(((nonZeroCoordinatesM.at<Point>(i).x)));
		CoordinateMY.push_back(((nonZeroCoordinatesM.at<Point>(i).y)));
		//cout << "Zero#" << i << ": " << nonZeroCoordinatesL.at<Point>(i).x << ", " << nonZeroCoordinatesL.at<Point>(i).y << endl;
	}
	for (int i = 0; i < nonZeroCoordinatesH.total(); i++)
	{
		CoordinateHX.push_back(((nonZeroCoordinatesH.at<Point>(i).x)));
		CoordinateHY.push_back(((nonZeroCoordinatesH.at<Point>(i).y)));
		//cout << "Zero#" << i << ": " << nonZeroCoordinatesL.at<Point>(i).x << ", " << nonZeroCoordinatesL.at<Point>(i).y << endl;
	}
	//cout << CoordinateLX.size() << endl;
	//cout << CoordinateLY.size() << endl;

	myfile << " ," << imname << ",";
	// find avg count of redlow, redmed, redhigh synapse around Low, M, H int synpse
	aroundsyncalc(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile);// find avg count of redlow, redmed, redhigh synapse around Low int synpse
	aroundsyncalc(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile);
	aroundsyncalc(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile);
	myfile << "," << "40redlow" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile, 40, 40);
	myfile << "," << "40redmed" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile, 40, 40);
	myfile << "," << "40redhih" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile, 40, 40);
	myfile << "80redlow" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile, 80, 80);
	myfile << "," << "80redmed" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile, 80, 80);
	myfile << "," << "80redhih" << ",";
	neighboursyn(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile, 80, 80);



}
//---[5]-------------
void reddetect(Mat im, Mat & redlow, Mat & redmed, Mat & redhigh)
{

	Mat enhancedl;
	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), enhancedl);
	erode(enhancedl, enhancedl, Mat());
	redlow = enhancedl.clone();
	imwrite("redlow.png", redlow);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), enhancedm);
	erode(enhancedm, enhancedm, Mat());
	erode(enhancedm, enhancedm, Mat());
	dilate(enhancedm, enhancedm, Mat());
	redmed = enhancedm.clone();
	imwrite("redmed.png", redmed);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), enhancedh);
	erode(enhancedh, enhancedh, Mat());
	redhigh = enhancedh.clone();
	imwrite("redhigh.png", redhigh);


}

void dendritecalc(string imname, Mat thresh, vector<Vec4i> lines, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	int smalwid = 0, midwid = 0, larwid = 0; int totl = 0, totm = 0, toth = 0; float totwid = 0;
	int LS = 0, MS = 0, HS = 0, LM = 0, MM = 0, HM = 0, LL = 0, ML = 0, HL = 0;// Lowint_Smallwid, Mediumint_MedWid, Highint_LargeWidth
	int denLS = 0, denMS = 0, denHS = 0, denLM = 0, denMM = 0, denHM = 0, denLL = 0, denML = 0, denHL = 0;// density Lowint_Smallwid, Mediumint_MedWid, Highint_LargeWidth
	int TdenLS = 0, TdenMS = 0, TdenHS = 0, TdenLM = 0, TdenMM = 0, TdenHM = 0, TdenLL = 0, TdenML = 0, TdenHL = 0;
	int totleng = 0, avgleng = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{

		Vec4i l = lines[i];
		Point a = (Point(l[0], l[1])); Point b = (Point(l[2], l[3]));
		double w = cv::norm(a - b);
		cv::Rect bRect(a, b);

		Mat lroi = redlow(bRect);// creating a new image from roi of redlow
		int lcount = countNonZero(lroi);
		Mat mroi = redmed(bRect);// creating a new image from roi of redmed
		int mcount = countNonZero(mroi);
		Mat hroi = redhigh(bRect);// creating a new image from roi of redhigh
		int hcount = countNonZero(hroi);
		Mat rect_roi = thresh(bRect);// creating a new image from roi of org green dendrites thresholded
		int wcount = countNonZero(rect_roi);

		if (w > 0)
		{
			totleng = totleng + w;
			int inLcountS; int inMcountS; int inHcountS; int inLcountM; int inMcountM; int inHcountM; int inLcountL; int inMcountL; int inHcountL;
			int widthden = wcount / w; // no of whote pixels in thresholded image/ length of synapse
			if (widthden < 10) // small width dendrite
			{
				smalwid++;
				Mat inlroiS = redlow(bRect);// creating a new image from roi of redlow
				inLcountS = countNonZero(inlroiS);
				Mat inmroiS = redmed(bRect);// creating a new image from roi of redmed
				inMcountS = countNonZero(inmroiS);
				Mat inhroiS = redhigh(bRect);// creating a new image from roi of redhigh
				inHcountS = countNonZero(inhroiS);
				//denLS = inLcountS / (bRect.width* bRect.height);
				//denMS = inMcountS / (bRect.width* bRect.height);
				//denHS = inHcountS / (bRect.width* bRect.height);
				LS = LS + inLcountS;  ///total synase count for low int sysnapses near small width dendrites
				MS = MS + inMcountS;
				HS = HS + inHcountS;


			}
			if (widthden >= 10 && widthden < 20)
			{
				midwid++;
				Mat inlroiM = redlow(bRect);// creating a new image from roi of redlow
				inLcountM = countNonZero(inlroiM);
				Mat inmroiM = redmed(bRect);// creating a new image from roi of redmed
				inMcountM = countNonZero(inmroiM);
				Mat inhroiM = redhigh(bRect);// creating a new image from roi of redhigh
				inHcountM = countNonZero(inhroiM);
				//denLM = inLcountM / (bRect.width* bRect.height);
				//denMM = inMcountM / (bRect.width* bRect.height);
				//denHM = inHcountM / (bRect.width* bRect.height);
				LM = LM + inLcountM;
				MM = MM + inMcountM;
				HM = HM + inHcountM;
			}
			if (widthden >= 20)
			{
				larwid++;
				Mat inlroiL = redlow(bRect);// creating a new image from roi of redlow
				inLcountL = countNonZero(inlroiL);
				Mat inmroiL = redmed(bRect);// creating a new image from roi of redmed
				inMcountL = countNonZero(inmroiL);
				Mat inhroiL = redhigh(bRect);// creating a new image from roi of redhigh
				inHcountL = countNonZero(inhroiL);
				//denLL = inLcountL / (bRect.width* bRect.height);
				//denML = inMcountL / (bRect.width* bRect.height);
				//denHL = inHcountL / (bRect.width* bRect.height);
				LL = LL + inLcountL;
				ML = ML + inMcountL;
				HL = HL + inHcountL;
			}



			totl = lcount + totl;
			totm = mcount + totm;
			toth = hcount + toth;
			totwid = totwid + w;


			TdenLS = TdenLS + denLS; ///total synase density for low int sysnapses near small width dendrites
			TdenMS = TdenMS + denMS;
			TdenHS = TdenHS + denHS;
			TdenLM = TdenLM + denLM;
			TdenMM = TdenMM + denMM;
			TdenHM = TdenHM + denHM;
			TdenLL = TdenLL + denLL;
			TdenML = TdenML + denML;
			TdenHL = TdenHL + denHL;
		}
		else
		{
			cout << "Oh OH OH!!!" << endl;
			denLS = 0; denMS = 0; denHS = 0; denLM = 0; denMM = 0; denHM = 0; denLL = 0; denML = 0; denHL = 0;
		}

	}
	
	//	cout << midwid << larwid << smalwid << endl;
	int Ftotl, Ftotm, Ftoth, Ftotwid, FavgLS, FavgMS, FavgHS, FavgLM, FavgMM, FavgHM, FavgLL, FavgML, FavgHL;
	//	int avgdenLS, avgdenMS , avgdenHS, avgdenLM , avgdenMM, avgdenHM, avgdenLL, avgdenML, avgdenHL;

	if (lines.size() > 0)
	{
		avgleng = totleng / lines.size();
		Ftotl = totl / lines.size();
		Ftotm = totm / lines.size();
		Ftoth = toth / lines.size();
		Ftotwid = totwid / lines.size();
		if (smalwid > 0)
		{
			FavgLS = LS / smalwid; /// average synase count for low int sysnapses near small width dendrites
			FavgMS = MS / smalwid;
			FavgHS = HS / smalwid;
			/*avgdenLS = TdenLS / smalwid; /// average synase density for low int sysnapses near small width dendrites
			avgdenMS = TdenMS / smalwid;
			avgdenHS = TdenHS / smalwid;*/
		}
		else
		{
			FavgLS = 0; FavgMS = 0; FavgHS = 0;//avgdenLS = 0;avgdenMS = 0;avgdenHS = 0;
		}
		if (midwid > 0)
		{
			FavgLM = LM / midwid;
			FavgMM = MM / midwid;
			FavgHM = HM / midwid;
			/*avgdenLM = TdenLM / midwid;
			avgdenMM = TdenMM / midwid;
			avgdenHM = TdenHM / midwid;*/
		}
		else
		{
			FavgLM = 0; FavgMM = 0; FavgHM = 0; //avgdenLM = 0; avgdenMM = 0; avgdenHM = 0;
		}
		if (larwid > 0)
		{
			FavgLL = LL / larwid;
			FavgML = ML / larwid;
			FavgHL = HL / larwid;
			/*avgdenLL = TdenLL / larwid;
			avgdenML = TdenML / larwid;
			avgdenHL = TdenHL / larwid;*/
		}
		else
		{
			FavgLL = 0; FavgML = 0; FavgHL = 0;// avgdenLL = 0; avgdenML = 0; avgdenHL = 0;
		}


	}
	else
	{
		avgleng = 0; Ftotl = 0; Ftotm = 0; Ftoth = 0; Ftotwid = 0; FavgLS = 0; FavgMS = 0; FavgHS = 0; FavgLM = 0; FavgMM = 0; FavgHM = 0; FavgLL = 0; FavgML = 0; FavgHL = 0; //avgdenLS = 0, avgdenMS = 0, avgdenHS = 0, avgdenLM = 0, avgdenMM = 0, avgdenHM = 0, avgdenLL = 0, avgdenML = 0, avgdenHL = 0;
	}
	myfile << "dendritecalc" << "," << imname << "," << lines.size() << ", " <<avgleng<< ", " << Ftotl + Ftotm + Ftoth << ", " << Ftotl << "," << Ftotm << "," << Ftoth << "," << Ftotwid << "," << smalwid << "," << midwid << "," << larwid << ", " << FavgLS << "," << FavgMS << "," << FavgHS << "," << FavgLM << "," << FavgMM << "," << FavgHM << "," << FavgLL << "," << FavgML << "," << FavgHL << ",";

}

void AxonorDendrite(vector<Vec4i>flines, Mat img1, vector<Vec4i> &finaldendrites, vector<Vec4i> &finalaxons)
{
	cv::inRange(img1, cv::Scalar(0, 25, 0), cv::Scalar(10, 255, 10), img1);
	for (size_t i = 0; i < flines.size(); i++)// no of lines within each cluster
	{
		int w = 20, h = 20;
		Vec4i l = flines[i];
		CvRect myrect1 = cvRect(l[0] - w / 2, l[1] - h / 2, w, h); /// create a 4*4 rectangle around the synapse
		CvRect myrect2 = cvRect(l[2] - w / 2, l[3] - h / 2, w, h); /// create a 4*4 rectangle around the synapse
		//cout << myrect1.x << " " << myrect1.y << " " << myrect2.x << " " << myrect2.y << endl;
		//check point 1
		if (myrect1.x >= 0 && myrect1.y >= 0 && (myrect1.width + myrect1.x) < img1.cols && (myrect1.height + myrect1.y) < img1.rows)
		{
			//cout << myrect1.width + myrect1.x << "  " << img1.rows << " " << (myrect1.height + myrect1.y) << "  " << img1.cols<<endl;
			Mat roi1 = img1(myrect1);
			//cout << countNonZero(roi1) << endl;
			if (countNonZero(roi1) > 150)
			{
				finaldendrites.push_back(flines[i]);
				continue;
			}


		}
		if (myrect2.x >= 0 && myrect2.y >= 0 && (myrect2.width + myrect2.x) < img1.cols && (myrect2.height + myrect2.y) < img1.rows)
		{
			//cout << myrect2.width + myrect2.x << "  " << img1.rows << " " << (myrect2.height + myrect2.y) << "  " << img1.cols << endl;
			Mat roi2 = img1(myrect2);
			if (countNonZero(roi2) > 150)
			{
				finaldendrites.push_back(flines[i]);
				continue;
			}

		}

		finalaxons.push_back(flines[i]);

	}
}
void maxline(vector<vector<Vec4i>> seglines,  string imname, Mat img1,Mat thresh, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{

	//cout << "segline size  "<<  seglines.size() << endl;
	//Mat thresh;
	vector<Vec4i>flines; double maxl = 0; int maxpos = 0; vector<double>maxlength; vector<Vec4i> finaldendrites, finalaxons;
//	cv::inRange(img1, cv::Scalar(0, 25, 0), cv::Scalar(10, 255, 10), thresh);
	for (int i = 0; i < seglines.size(); i++)
	{
		maxl = 0; maxpos = 0;
		if (seglines[i].size() >0)
		{

			for (int j = 0; j < seglines[i].size(); j++)
			{
				Vec4i l = seglines[i][j];
				Point p1 = (l[0], l[1]); Point p2 = (l[2], l[3]);
				double res = cv::norm(p1 - p2);
				if (res > maxl)
				{
					maxpos = j;
					maxl = res;
				}

			}
			flines.push_back(seglines[i][maxpos]);
			maxlength.push_back(maxl);
		}

	}
	RNG rng;

	/*for (size_t i = 0; i < flines.size(); i++)// no of lines within each cluster
	{
		Vec4i l = flines[i];
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		line(img1, Point(l[0], l[1]), Point(l[2], l[3]), color, 3, CV_AA);
	}*/
	finaldendrites.clear();
	finalaxons.clear();
	AxonorDendrite(flines, img1, finaldendrites, finalaxons);

	myfile << "axon" << ",";
	dendritecalc(imname, thresh, finalaxons, redlow, redmed, redhigh, myfile);


	myfile << "dendrite" << ",";
    dendritecalc(imname, thresh, finaldendrites, redlow, redmed, redhigh, myfile);

}
void segments(vector<double>ang, vector<Vec4i> hough_lines, string imname, Mat img,Mat thresh,  Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	int randNum0, randNum1;
	Mat imguntouched = img.clone();
	vector<double> cluster_center; //cluster center value
	int cluster_no[1000]; // maintains cluster number for all lines
	vector<vector<Vec4i>>finalcluster;
	vector<vector<double>> finalangle;
	RNG rng(12345);
	vector<Vec4i> tempor;
	//if there are just 2 lines
	if (hough_lines.size() == 1)
	{
		tempor.push_back(hough_lines[0]);
		finalcluster.push_back(tempor);
	}
	else if ((hough_lines.size() == 2) && (abs(ang[0] - ang[1]) < 10))
	{
		Point a0 = (hough_lines[0][0], hough_lines[0][1]);
		Point b0 = (hough_lines[0][2], hough_lines[0][3]);
		Point a1 = (hough_lines[1][0], hough_lines[1][1]);
		Point b1 = (hough_lines[1][2], hough_lines[1][3]);
		if (norm(a0 - b0) > norm(a1 - b1))//Euclidian distance
		{
			tempor.push_back(hough_lines[0]);//only one line exists
			finalcluster.push_back(tempor);
		}

		else
		{
			tempor.push_back(hough_lines[1]);//only one line exists
			finalcluster.push_back(tempor);
		}
	}
	else//if greater than 2 lines exist
	{
		randNum0 = rand() % (hough_lines.size() - 0) + 0;
		randNum1 = rand() % (hough_lines.size() - 0) + 0;
		cluster_center.push_back(ang[randNum0]);
		cluster_center.push_back(ang[randNum1]);
		cluster_no[randNum0] = 0;
		cluster_no[randNum1] = 1;

		int cencalc[1000];

		for (int co = 0; co < 100; co++)//no of runs of k-means
		{
			for (int i = 0; i < ang.size(); i++)
			{
				double min = 100000; int k = 0; double sum_angleDiff = 0;

				for (int j = 0; j < cluster_center.size(); j++)//check dist with all cluster centers
				{
					double diff = abs(cluster_center[j] - ang[i]);
					sum_angleDiff = sum_angleDiff + diff;
					if (diff < min) //if (difference < smallest difference ) then (min=diff)
					{
						min = diff;
						k = j;
					}
				}
				if (min < (tune*(sum_angleDiff / cluster_center.size()))) // if (min< given threshold) then (put into cluster) 
					cluster_no[i] = k;

				else //else(create new cluster)
				{

					cluster_no[i] = cluster_center.size();
					cluster_center.push_back(ang[i]);
				}

			}
			for (int i1 = 0; i1 < cluster_center.size(); i1++)//update centers of clusters
			{//new cluster center= sum of all angles in a cluster / total no of lines in a cluster
				double summ = 0; int count = 0;
				for (int j1 = 0; j1 < hough_lines.size(); j1++)
				{
					if (cluster_no[j1] == i1)
					{
						summ = summ + ang[j1];
						count = count + 1;
					}
				}
				cluster_center[i1] = summ / count;
			}

		}


		//puttting into vector<vector> based on the indexes(clusters) of the lines
		for (int n = 0; n < cluster_center.size(); n++)
		{
			vector<Vec4i> temp;
			vector<double> tempang;
			for (int m = 0; m < hough_lines.size(); m++)
			{


				if (cluster_no[m] == n)
				{
					tempang.push_back(ang[m]);
					temp.push_back(hough_lines[m]);
				}
			}
			finalcluster.push_back(temp);
			temp.clear();
			finalangle.push_back(tempang);
			tempang.clear();

		}
	}// else ends (run for greater than two lines)
	
	maxline(finalcluster, imname, imguntouched, thresh, redlow, redmed, redhigh, myfile);

}
//----[6]----------detects dedrite,classifies as dendrite/axon, calc metrics-
void dendritedetect(string imname, Mat img, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	imwrite("Vorgdend.tif", img);
	img = imread("Vorgdend.tif");
	//==========================================================================================================
	Mat imgclone = img.clone();
	cv::inRange(img, cv::Scalar(0, 25, 0), cv::Scalar(10, 255, 10), img);
	vector<double>angle;
	vector<Vec4i> lines4;
	HoughLinesP(img, lines4, 1, CV_PI / 180, houghthresh, 65, 25);//50 50 10

	for (size_t i = 0; i < lines4.size(); i++)
	{
		Vec4i l = lines4[i];
		double ang = (atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI);
		angle.push_back(ang);
	}
	if (lines4.size() > 0)
		segments(angle, lines4, imname, imgclone,img, redlow,redmed,redhigh,myfile);
	//else if(lines4.size() == 1)  finallines=lines4[0];
	else
	{
		//myfile << "No dendrites" << ",";
		myfile << "NOdendritecalc" << "," <<"dendritecalc"<<","<< imname << "," << 0 << ", " << 0 << ", " << 0 << ", " << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ", " << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 ;
		myfile << "NOdendritecalc" << "," <<"dendritecalc" << "," << imname << "," << 0 << ", " << 0 << ", " << 0 << ", " << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ", " << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0;

	}

}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
		cv::inRange(image_roi, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), rl);// channel red low
		erode(rl, rl, Mat());
		int countlr = countNonZero(rl);
		countlrsum = countlr + countlrsum;

		cv::inRange(image_roi, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), rm);// channel red medium
		erode(rm, rm, Mat());
		int countrm = countNonZero(rm);
		countrmsum = countrm + countrmsum;

		cv::inRange(image_roi, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), rh);// channel red high
		erode(rh, rh, Mat());
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


// -----[3]------cell metrics for every plane in an image (5 z-planes)
void cellmetrics(string imname, int i, vector < vector<Point>> blucontours, vector < vector<Point>> nucontours, vector <vector<Point>> astcontours, ofstream &myfile)
{
	myfile << "," << "bluecontours" << ",";
	calcCellMetrics(imname, i, blucontours, myfile);
	myfile << "," << "neural contours" << ",";
	calcCellMetrics(imname, i, nucontours, myfile);
	myfile << "," << "astroyte contours" << ",";
	calcCellMetrics(imname, i, astcontours, myfile);


}
//----[4]----- finds the number of low/med/high synapses near an astrocyte and neural cell
void synapcalc(string imname, int i, vector <Mat> stackim, vector<vector<Point>> nucontours)
{
	Mat redres;
	Mat img = stackim[i];
	//cv::Mat normalized;
	vector <Mat> redlow, redmed, redhigh;
	//cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_16UC1);


	// counting the number of red  intensity pixels around a nuclei cell
	int totrl = 0, avglo;
	if (i == 1 || (i % 3 == 1))// channel red low
	{
		// Enhance the red low channel
		cv::inRange(img, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), redres);
		erode(redres, redres, Mat());
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
		cv::inRange(img, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), redres);
		erode(redres, redres, Mat());

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
		cv::inRange(img, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), redres);
		erode(redres, redres, Mat());
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
	// total no of synapses around, avg low intensity around , avg medium intensity synapses around, avg high intensity synapses around
	myfile << "," << "synapse cal" << " ," << imname << "," << avglo+avgme+avglh << ","<< avglo << "," << avgme << "," << avglh;

}
// thresholds all three channels for all z planes of one image and writes it

//----------[1]
void enhanceImage(string imname, vector <Mat> stackim, vector <Mat> & enhance)
{

	enhance.clear();
	for (int i = 0; i < stackim.size(); i++)//breaks 16 bit image into different z planes
	{
		Mat img = stackim[i]; Mat gry;
		// cvtColor(img, img, CV_BGR2GRAY);
		// Normalize the image
		//cv::Mat normalized;
		vector <Mat> redlow, redmed, redhigh;
		//cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_16UC1);
		//cvtColor(img, gry, CV_BGR2GRAY);

		//-----------------------------------------------------------------------------------------------------------------------------------
		// Enhance the image using Gaussian blur and thresholding
		cv::Mat enhanced;
		if (i == 0 || (i % 3 == 0))// channel blue
		{
			// Enhance the blue channel
			cv::inRange(img, cv::Scalar(500, 0, 0), cv::Scalar(50000, 50, 20), enhanced);// blue threshold
			bitwise_not(enhanced, enhanced);
			dilate(enhanced, enhanced, Mat());
			enhance.push_back(enhanced);
			string name = format("enh_%d.tif", i);
			imwrite(name, enhanced);
		}
		if (i == 1 || (i % 3 == 1))// channel red
		{
			// Enhance the red channel
			// cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
			cv::inRange(img, cv::Scalar(0, 0, 7000), cv::Scalar(10, 10, 65535), enhanced);
			enhance.push_back(enhanced);
			string name = format("enh_%d.tif", i);
			imwrite(name, enhanced);
		}

		if (i == 2 || (i % 3 == 2))// channel green
		{
			// Enhance the green channel
			//cv::threshold(normalized, enhanced, 20, 255, cv::THRESH_BINARY);
			cv::inRange(img, cv::Scalar(0, 100, 0), cv::Scalar(10, 9500, 10), enhanced);
			bitwise_not(enhanced, enhanced);
			enhance.push_back(enhanced);
			string name = format("enh_%d.tif", i);
			imwrite(name, enhanced);
		}

	}

	cout << "enhance= " << enhance.size() << endl;
}
// displays combined image with blue thresholds
void drawthreshold(vector <Mat> stackim, int i, string imname)
{
	Mat added = stackim[i] + stackim[i + 1] + stackim[i + 2];
	string name = format("C:\\Users\\VIneeta\\Desktop\\CCHMC Projects\\Eli Lilly\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[2221]\\2016-07-13T162902-0400[3133]\\Segmented\\%s_z%d_mod.tif", imname.c_str(), i);
	imwrite(name, added);
}
//----------[2]
// counts number of astrocytes, cells and nueral cells

void cellcount(string imname, vector <Mat> stackim, vector <Mat> enhance)
{
	myfile << endl;


	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<vector<Point>> blucontours;
	vector<vector<Point>> nucontours;
	vector<vector<Point>> astcontours;
	vector<vector<Point>> othcontours;
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
			blucontours.clear();
			// FINDING CELLS: and saving the contours that are above a certain area
			for (int d = 0; d < contours.size(); d++)
			{
				float chaid = contourArea(contours[d], false);
				if (chaid > 200)
					blucontours.push_back(contours[d]);
			}
			tau = i + 2; // decides image index for green channel



			// FIND NUERAL CELLS: draw bounding rectangle around blue contours
			// Check for presence of dendrites(green) around the blue contours
			vector<Rect> brect(blucontours.size());
			vector<Rect> arect(blucontours.size());
			vector< float> asprat(blucontours.size());
			for (int io = 0; io < blucontours.size(); io++)
			{
				brect[io] = boundingRect(Mat(blucontours[io]));
				float chaid = contourArea(blucontours[io], false);
				Mat image = enhance[tau];
				Mat image_roi = image(brect[io]);// creating a new image from roi
				int count = countNonZero(image_roi);
				if (chaid > 1000)
				{
					if (count > NeuralTHRESH) //neural contours (threshold is the amount of green pixels in the bounding rectangle around blucontours) 
					{
						nucontours.push_back(blucontours[io]);
					}
					else
					{
						// FIND ASTROCYTES: large, oval & textured
						// Checking for aspect ratio (Oval), area and presense of large number of black pixels(textured)

						asprat[io] = brect[io].height / brect[io].width;// aspect ratio
						Mat imagee = enhance[i];
						Mat image_roi = imagee(arect[io]);// creating a new image from roi
						int countbl = (image_roi.rows*image_roi.cols) - countNonZero(image_roi);// counting number of black pixels
						if ((countbl < 500) && (asprat[io] >= 0.5 && asprat[io] <= 1.2))// if not very stippled and aspect ratio closer to 1 (more circular)-not astrocyte
							othcontours.push_back(blucontours[io]);// other contours
						else
							astcontours.push_back(blucontours[io]);//astrocytes
					}

				}
			}


			for (int nu = 0; nu < nucontours.size(); nu++)
				drawContours(stackim[i], nucontours, nu, Scalar(0, 66335, 0), 2, 8, vector<Vec4i>(), 0, Point());//green
			for (int du = 0; du < astcontours.size(); du++)  // finding and saving the contours that are above a certain area
				drawContours(stackim[i], astcontours, du, Scalar(0, 65535, 65535), 2, 8, vector<Vec4i>(), 0, Point());//yellow
			for (int ou = 0; ou < othcontours.size(); ou++)  // finding and saving the contours that are above a certain area
				drawContours(stackim[i], othcontours, ou, Scalar(35000, 0, 65535), 2, 8, vector<Vec4i>(), 0, Point());//pink


			/*~~~~~~*/drawthreshold(stackim,i, imname);


			cellmetrics(imm, i, blucontours, nucontours, astcontours, myfile); // for the particular z-layer


			int beta = i + 1;// red channel
			myfile << "," << "neural contours" << ",";
			synapcalc(imm, beta, stackim, nucontours); // calculating presence of red aroung neural cells
			myfile << "," << "astr contours" << ",";
			synapcalc(imm, beta, stackim, astcontours);// calculating count of red around astrocytes
		}

		/*//---------------- dendrite stuff------------------------------------
		Mat redlow, redmed, redhigh;
		if (i == 2 || (i % 3 == 2))// channel green

		{
			//reddetect(stackim[i - 1], redlow, redmed, redhigh);//saves LMH red thresh images
			//dendritedetect(imname, stackim[i], redlow, redmed, redhigh, myfile);//detects dedrite,classifies as dendrite/axon, calc metrics-
		}
		if (i == 1 || (i % 3 == 1))//channel red
		{
			redinf(imname, stackim[i], myfile);// calls 
			//neighborsyn[find avg count of redlow, redmed, redhigh synapse around Low, M, H int synpse]
			//and aroundsyn[finds avg count of redlow, redmed, redhigh synapse around Low, M, H int synpse at 40 & 80 dist]
		}
		//myfile << endl;*/
	}


}


void main()
{
	std::vector<cv::Mat> channel; int valred = 255, valblue = 255, valgreen = 255; Mat im_color;
	vector<Mat> stackim; string raw_path;
	myfile.open("elilyfiles.csv");
	// reading 16 bit image
	for (int n1 = 2; n1 <= 7; n1++)
	{
		for (int n2 = 2; n2 <= 11; n2++) //11
		{
			for (int n3 = 1; n3 <= 72; n3++)//35
			{
				channel.clear();
				stackim.clear();

				if (n2 < 10)
					raw_path = format("C:\\Users\\VIneeta\\Desktop\\CCHMC Projects\\Eli Lilly\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[2221]\\2016-07-13T162902-0400[3133]\\00%d00%d-%d.tif", n1, n2, n3);  // 002002-1
				else
					raw_path = format("C:\\Users\\VIneeta\\Desktop\\CCHMC Projects\\Eli Lilly\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[1316]\\07132016synaptophysin-neuronal marke[2221]\\2016-07-13T162902-0400[3133]\\00%d0%d-%d.tif", n1, n2, n3);  // 002010-1

				string imname = format("00%d0%d-%d.tif", n1, n2, n3);

				cout << format("Processing %s", imname.c_str()) << endl;

				cv::imreadmulti(raw_path, channel, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

				if (channel.empty())
					cout << "On no!!" << endl;

				for (unsigned int i = 0; i < channel.size(); i++)
				{

					Mat src = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1);;
					Mat empty_image = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1);
					Mat result_blue(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!
					Mat result_green(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!
					Mat result_red(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!

					src = channel[i];///stores values of channel [i] temporarily

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
						result_blue = changeimg(result_blue, 2000, 0);
						//dilate(result_blue, result_blue, Mat());
						//dilate(result_blue, result_blue, Mat());
						string iname = format("Vin_%d.tif", i);
						imwrite(iname, result_blue);
						stackim.push_back(result_blue);
					}
					if (i == 2 || (i % 3 == 2))// channel red
					{
						Mat in2[] = { empty_image, src, empty_image };
						int from_to2[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in2, 3, &result_red, 1, from_to2, 3);
						result_red = changeimg(result_red, 200, 0);
						string iname = format("Vin_%d.tif", i);
						imwrite(iname, result_red);
						stackim.push_back(result_red);
					}
					if (i == 1 || (i % 3 == 1))// channel green
					{
						Mat in3[] = { empty_image, empty_image, src };
						int from_to3[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in3, 3, &result_green, 1, from_to3, 3);
						result_green = changeimg(result_green, 200, 0);
						string iname = format("Vin_%d.tif", i);
						imwrite(iname, result_green);
						stackim.push_back(result_green);

					}


				}

				enhanceImage(imname, stackim, enhance);// thresholds all three channels for all z planes of one image and writes it
				cellcount(imname, stackim, enhance);// finds blue contours, astrocytes, nueral cells and other cells and calculates the metrics for these
				//--> CALLS [cellmetrics-metrics of astrocyte and neural cells]
				//[synapcalc- calculates count of red around astrocytes and neural cells]
				//[reddetect-finds red around dendrites]
				//[dendritedetect][redinf]

				cout << "Entering Dendrite Stuff" << endl;
				//---------dendrite stuff-----------------
				Mat redlow, redmed, redhigh;


				for (int i = 0; i < stackim.size(); i++)
				{

				if (i == 2 || (i % 3 == 2))// channel green
				{

				reddetect(stackim[i - 1], redlow, redmed, redhigh);
				dendritedetect(imname, stackim[i], redlow, redmed, redhigh, myfile);
				}
				if (i == 1 || (i % 3 == 1))
				redinf(imname,stackim[i], myfile);

				}


			}
		}
	}


	waitKey(0);
	myfile.close();
}






