#include<iostream>
#include<string>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Mat img;
	const string trackbar_name = "trackbar";

	//read image
	img = imread(argv[1]);
	
	string str;
	int mode = 0;
	int height = img.rows / 2;
	int width = img.cols / 2;
	int Rotation = 0;

	cout << "Input 1 to rotate whole image, 2 to rotate circle, otherwise escape.\nPress enter to continue!\n";
	cin >> mode;

	//create trackbar
	namedWindow("NEWWINDOW", WINDOW_AUTOSIZE);
	createTrackbar(trackbar_name, "NEWWINDOW", &Rotation, 360);


	while (mode==1) {
		Mat src;
		img.convertTo(src, -1);
		Mat for_Rotation = getRotationMatrix2D(Point(width, height), 360 - Rotation, 1);//affine transformation matrix for 2D rotation
		Mat for_Rotated;//declaring a matrix for rotated image
		
		warpAffine(src, for_Rotated, for_Rotation, src.size());//applying affine transformation
		imshow("NEWWINDOW", for_Rotated);

		int key = waitKey(50);
		if (key == 13)//press enter to close window
			break;
	}

	cout << "Continue?(Y/N)\n";
	cin >> str;
	if (str == "Y")mode = 2;
	else return 0;
	cout << "Continue!\n";

	setTrackbarPos(trackbar_name, "NEWWINDOW", 50);
	while (mode==2) {
		Vec3f circ(width, height, height);

		// Draw the mask: white circle on black background
		Mat mask = Mat::zeros(img.size(), CV_8UC1);
		circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), -1);

		Mat res;

		// Copy only the image under the white circle to black image
		img.copyTo(res, mask);

		Point2f center(img.cols / 2, img.rows / 2);
		Mat rotation = getRotationMatrix2D(center, 360 - Rotation, 1.0);
		warpAffine(res, res, rotation, img.size());
		Mat clone = img.clone();

		// Copy again
		res.copyTo(clone, mask);

		imshow("NEWWINDOW", clone);

		int key = waitKey(50);
		if (key == 13)//press enter to close window
			break;
	}
	return 0;
}