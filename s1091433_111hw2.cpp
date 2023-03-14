#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	namedWindow("Sobel", 1);
	namedWindow("Sketch", 1);
	Mat img = imread(argv[1], 1);
	Mat res, blur, sketch;
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad, skt1, skt2;
	Size2i blur_size(3, 3);
	int scale = 3, ksize = 1, delta = 1;
	// Gaussianblur the image to reduce noise
	GaussianBlur(img, blur, blur_size, 0, 0, 4);
	// convert to grayscale image
	cvtColor(blur, res, COLOR_BGR2GRAY);

	Mat mask = Mat::zeros(img.size(), CV_8UC1);
	//addWeighted(img, -1, mask, 0, 255, res);
	/*cv2.Sobel(img, res, ddepth, dx, dy, ksize, scale)
		# img 來源影像
		# dx 針對 x 軸抓取邊緣
		# dy 針對 y 軸抓取邊緣	
		# ddepth 影像深度，設定 - 1 表示使用圖片原本影像深度
		# ksize 運算區域大小，預設 1 (必須是正奇數)
		# scale 縮放比例常數，預設 1 (必須是正奇數)*/

	// calculate derivative in x and y direction (scale = 3, ksize = 1, delta = 0)
	Sobel(res, grad_x, CV_64F, 1, 0, scale, ksize, delta, BORDER_DEFAULT);
	Sobel(res, grad_y, CV_64F, 0, 1, scale, ksize, delta, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	// reverse to white background image
	addWeighted(grad, -1, mask, 0, 255, sketch);

	while (true) {
		imshow("Sobel", grad);
		imshow("Sketch", sketch);
		int key = waitKey(50);
		if (key == 13) break;
	}
}