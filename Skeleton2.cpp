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
// for each image (with all z stack layers) - finds the redlow, redmed and redhigh image
void neighboursyn(Mat redlow, Mat redmed, Mat redhigh, vector<int> CoordinatesX, vector<int> CoordinatesY, ofstream & myfile)
{
	unsigned int totl = 0, totm = 0, toth = 0; int w = 40, h = 40;
	int lcount1; int mcount1; int hcount1; int lcount2; int mcount2; int hcount2; int lcount3; int mcount3; int hcount3; int lcount4; int mcount4; int hcount4;

	for (int i = 0; i < CoordinatesX.size(); i++)
	{

		Point a = Point(CoordinatesX[i], CoordinatesY[i]);
		lcount1 = 0; mcount1 = 0; hcount1 = 0; lcount2 = 0; mcount2 = 0; hcount2 = 0; lcount3 = 0; mcount3 = 0; hcount3 = 0; lcount4 = 0; mcount4 = 0; hcount4 = 0;
		int chaosl = 4; int chaosm = 4; int chaosh = 4;

		if (((a.x) + w + 20 < redlow.rows) && ((a.x) - w-20 > 0) && ((a.y) + h + 20 < redlow.cols) && ((a.y) - h -20> 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				Mat lroi = redlow(myrect);// creating a new image from roi of redlow
				Mat Image1 = lroi(Rect(a.x, a.y, w / 2, h / 2));
				Mat Image2 = lroi(Rect(a.x + w / 2, a.y, w / 2, w / 2));
				Mat Image3 = lroi(Rect(a.x, a.y + w / 2, h / 2, h / 2));
				Mat Image4 = lroi(Rect(a.x + w / 2, a.y + h / 2, w / 2, h / 2));
				lcount1 = countNonZero(Image1);
				lcount2 = countNonZero(Image2);
				lcount3 = countNonZero(Image3);
				lcount4 = countNonZero(Image4);
				chaosl = 4;
				if ((lcount1 <= lcount2 + 1000) && (lcount1 >= lcount2 - 1000))
				{
					--chaosl;
				}
				if ((lcount2 <= lcount3 + 1000) && (lcount2 >= lcount3 - 1000))
				{
					--chaosl;
				}
				if ((lcount3 <= lcount4 + 1000) && (lcount3 >= lcount4 - 1000))
				{
					--chaosl;
				}
				if ((lcount4 <= lcount1 + 1000) && (lcount4 >= lcount1 - 1000))
				{
					--chaosl;
				}
				cout << chaosl << endl;
				totl = totl + chaosl;
			}
			else
				cout << "hop" << endl;
		}
				
				//cout <<"totl:  "<< totl<< endl;



		if (((a.x) + w + 20 < redmed.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redmed.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed

				Mat Image1 = mroi(Rect(a.x, a.y, w/2, h/2));
				Mat Image2 = mroi(Rect(a.x + w/2, a.y, w/2, w/2));
				Mat Image3 = mroi(Rect(a.x, a.y + w/2, h/2, h/2));
				Mat Image4 = mroi(Rect(a.x + w/2, a.y + h/2, w/2, h/2));
				mcount1 = countNonZero(Image1);
				mcount2 = countNonZero(Image2);
				mcount3 = countNonZero(Image3);
				mcount4 = countNonZero(Image4);
				chaosm = 4;
				if ((mcount1 <= mcount2 + 1000) && (mcount1 >= mcount2 - 1000))
				{
					--chaosm;
				}
				if ((mcount2 <= mcount3 + 1000) && (mcount2 >= mcount3 - 1000))
				{
					--chaosm;
				}
				if ((mcount3 <= mcount4 + 1000) && (mcount3 >= mcount4 - 1000))
				{
					--chaosm;
				}
				if ((lcount4 <= lcount1 + 1000) && (lcount4 >= lcount1 - 1000))
				{
					--chaosm;
				}
				cout << chaosm << endl;
				totm = totm + chaosm;
			}
		}


		if (((a.x) + w + 20 < redhigh.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redhigh.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed
				

				Mat Image1 = hroi(Rect(a.x, a.y, w / 2, h / 2));
				Mat Image2 = hroi(Rect(a.x + w / 2, a.y, w / 2, w / 2));
				Mat Image3 = hroi(Rect(a.x, a.y + w / 2, h / 2, h / 2));
				Mat Image4 = hroi(Rect(a.x + w / 2, a.y + h / 2, w / 2, h / 2));
				hcount1 = countNonZero(Image1);
				hcount2 = countNonZero(Image2);
				hcount3 = countNonZero(Image3);
				hcount4 = countNonZero(Image4);
				chaosh = 4;
				if ((hcount1 <= hcount2 + 1000) && (hcount1 >= hcount2 - 1000))
				{
					--chaosh;
				}
				if ((hcount2 <= hcount3 + 1000) && (hcount2 >= hcount3 - 1000))
				{
					--chaosh;
				}
				if ((hcount3 <= hcount4 + 1000) && (hcount3 >= hcount4 - 1000))
				{
					--chaosh;
				}
				if ((hcount4 <= hcount1 + 1000) && (hcount4 >= hcount1 - 1000))
				{
					--chaosh;
				}
				cout << chaosh << endl;
				toth = toth + chaosh;

			}
		}


	}
	//cout << totl << "," << totm << "," << toth << endl;
}

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
		if (CoordinatesX.size()!=0)
		 myfile << "," << totl / CoordinatesX.size() << "," << totm / CoordinatesX.size() << "," << toth / CoordinatesX.size()<<",";
		else
			myfile << "," << 0 << "," << 0 << "," << 0 << ",";
	}
	
	
void redinf(Mat im, ofstream &myfile)
{
	cv::Mat normalizedl; Mat enhancedl; vector< int> CoordinateLX; vector< int> CoordinateLY; vector< int> CoordinateMX; vector< int> CoordinateMY;
	vector< int> CoordinateHX; vector< int> CoordinateHY;

	cv::normalize(im, normalizedl, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red low channel
	cv::inRange(normalizedl, cv::Scalar(0, 0, 25), cv::Scalar(10, 10, 30), enhancedl);
	Mat redlow = enhancedl.clone();
	Mat nonZeroCoordinatesL;
	findNonZero(enhancedl, nonZeroCoordinatesL);

	cv::Mat normalizedm; Mat enhancedm;
	cv::normalize(im, normalizedm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red medium channel
	cv::inRange(normalizedm, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 80), enhancedm);
	Mat redmed = enhancedm.clone();
	Mat nonZeroCoordinatesM;
	findNonZero(enhancedm, nonZeroCoordinatesM);

	cv::Mat normalizedh; Mat enhancedh;
	cv::normalize(im, normalizedh, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// Enhance the red high channel
	cv::inRange(normalizedh, cv::Scalar(0, 0, 90), cv::Scalar(10, 10, 255), enhancedh);
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


	/*aroundsyncalc(redlow, redmed, redhigh, CoordinateLX, CoordinateLY,myfile);// find avg count of redlow, redmed, redhigh synapse around Low int synpse
	aroundsyncalc(redlow, redmed, redhigh, CoordinateMX, CoordinateMY,myfile);
	aroundsyncalc(redlow, redmed, redhigh, CoordinateHX, CoordinateHY,myfile);*/
	neighboursyn(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile);
}

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

void dendritecalc(string imname, Mat thresh, vector<Vec4i> lines, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	int smalwid = 0, midwid = 0, larwid = 0; int totl = 0, totm = 0, toth = 0; float totwid = 0;
	int LS = 0, MS = 0, HS = 0, LM = 0, MM = 0, HM = 0, LL = 0, ML = 0, HL = 0;// Lowint_Smallwid, Mediumint_MedWid, Highint_LargeWidth
	int denLS = 0, denMS = 0, denHS = 0, denLM = 0, denMM = 0, denHM = 0, denLL = 0, denML = 0, denHL = 0;// density Lowint_Smallwid, Mediumint_MedWid, Highint_LargeWidth
	int TdenLS = 0, TdenMS = 0, TdenHS = 0, TdenLM = 0, TdenMM = 0, TdenHM = 0, TdenLL = 0, TdenML = 0, TdenHL = 0;
	
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
			FavgLS = 0; FavgMS =0;FavgHS =0;//avgdenLS = 0;avgdenMS = 0;avgdenHS = 0;
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
		Ftotl = 0; Ftotm = 0; Ftoth = 0; Ftotwid = 0; FavgLS = 0; FavgMS = 0; FavgHS = 0; FavgLM = 0; FavgMM = 0; FavgHM = 0; FavgLL = 0; FavgML = 0; FavgHL = 0; //avgdenLS = 0, avgdenMS = 0, avgdenHS = 0, avgdenLM = 0, avgdenMM = 0, avgdenHM = 0, avgdenLL = 0, avgdenML = 0, avgdenHL = 0;
	}
	myfile << imname << "," << lines.size() << ", " << Ftotl << "," << Ftotm << "," << Ftoth << "," << Ftotwid << "," << smalwid << "," << midwid << "," << larwid << FavgLS << "," << FavgMS << "," << FavgHS << "," << FavgLM << "," << FavgMM << "," << FavgHM << "," << FavgLL << "," << FavgML << "," << FavgHL <<endl;
	
}


void dendritedetect(string imname, Mat img, Mat  redlow, Mat redmed, Mat redhigh, ofstream & myfile)
{
	cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::inRange(img, cv::Scalar(0, 200, 0), cv::Scalar(10, 255, 10), img);
	Mat thresh = img.clone(); // original thresholded image of greeen dendrites

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

	Mat cdst;
	cvtColor(skel, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines; 
	vector<RotatedRect> minRect(lines.size());
	GaussianBlur(skel, skel, Size(1, 1), 2.0, 2.0);
	HoughLinesP(skel, lines, 5, CV_PI / 135, 70, 50, 10);
	 // to dispay the dendrite lines
	/*for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
		Point a = (Point(l[0], l[1])); Point b = (Point(l[2], l[3]));
		cv::rectangle(cdst, a, b, cv::Scalar(255, 0, 255), 1, 8);
	}
	imshow("dendrites", cdst);
	//imshow("source", skel);
	//imshow("detected lines", cdst);*/
	dendritecalc(imname, thresh, lines, redlow, redmed, redhigh, myfile);

}




void main()
{
	std::vector<cv::Mat> channel; int valred = 255, valblue = 255, valgreen = 255; Mat im_color;
	vector<Mat> stackim; string raw_path;
	myfile.open("DendriteMetrics2.csv");
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

						reddetect(stackim[i - 1], redlow, redmed, redhigh);
						dendritedetect(imname, stackim[i], redlow, redmed, redhigh, myfile);
					}
					if (i==1 || (i%3==1))
						redinf(stackim[i],myfile);
					
				}

			}
		}
	}

	waitKey(0);
	myfile.close();
}