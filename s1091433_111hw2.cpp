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
	GaussianBlur(img, blur, Size(5, 5), 0, 0, 4);
	cvtColor(blur, res, COLOR_BGR2GRAY);

	Mat mask = Mat::zeros(img.size(), CV_8UC1);
	//addWeighted(img, -1, mask, 0, 255, res);
	/*cv2.Sobel(img, res, ddepth, dx, dy, ksize, scale)
		# img �ӷ��v��
		# dx �w�� x �b�����t
		# dy �w�� y �b�����t	
		# ddepth �v���`�סA�]�w - 1 ��ܨϥιϤ��쥻�v���`��
		# ksize �B��ϰ�j�p�A�w�] 1 (�����O���_��)
		# scale �Y���ұ`�ơA�w�] 1 (�����O���_��)*/

	Sobel(res, grad_x, -1, 1, 0, 1, 3, 0, BORDER_DEFAULT);
	Sobel(res, grad_y, -1, 0, 1, 1, 3, 0, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x, 1, 0);
	convertScaleAbs(grad_y, abs_grad_y, 1, 0);
	addWeighted(abs_grad_x, 1.5, abs_grad_y, 1.5, 10, grad);


	Mat mask2 = Mat::ones(img.size(), CV_8UC1);
	addWeighted(grad, -1, mask, 0, 255, sketch);

	while (true) {
		imshow("Sobel", grad);
		imshow("Sketch", sketch);
		int key = waitKey(50);
		if (key == 13) break;
	}
}