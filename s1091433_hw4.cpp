#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius);
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag = 0);
Mat showspec(Mat I);
Mat shift(Mat img);

int main(int argc, char**argv)
{
    Mat imgIn = imread(argv[1], IMREAD_GRAYSCALE);
    if (imgIn.empty())
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }
    resize(imgIn, imgIn, Size(imgIn.cols / 2, imgIn.rows / 2));
    imshow("input", imgIn);
    imgIn.convertTo(imgIn, CV_32F);
    
    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
    imgIn = imgIn(roi);
    imshow("spec", showspec(imgIn));
    // PSD calculation
    Mat imgPSD;
    calcPSD(imgIn, imgPSD);
    fftshift(imgPSD, imgPSD);
    normalize(imgPSD, imgPSD, 0, 255, NORM_MINMAX);

    Mat H = Mat(roi.size(), CV_32F, Scalar(1));
    int r = 11;
    // 製作雜訊的遮罩
    synthesizeFilterH(H, Point(400, 270), r);
    synthesizeFilterH(H, Point(400, 304), r);
    synthesizeFilterH(H, Point(400, 334), r);

    Mat imgOut;
    // 將遮罩 H 與原圖相乘把雜訊部分去除
    fftshift(H, H);
    filter2DFreq(imgIn, imgOut, H);

    imgOut.convertTo(imgOut, CV_8U);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    imshow("result", imgOut);
    fftshift(H, H);
    normalize(H, H, 0, 255, NORM_MINMAX);
    //imshow("filter", H);
    imshow("resultspec", showspec(imgOut));
    waitKey(0);
    return 0;
}

void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    // 原圖的傅立葉轉換
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    // 跟遮罩相乘去除雜訊
    mulSpectrums(complexI, complexH, complexIH, 0);
    // 做 idft 取實部轉回一般圖片
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}

void synthesizeFilterH(Mat& input, Point center, int radius)
{
    Point c2 = center, c3 = center, c4 = center;
    c2.y = input.rows - center.y;
    c3.x = input.cols - center.x;
    c4 = Point(c3.x,c2.y);
    circle(input, center, radius, 0, -1, 8);
    circle(input, c2, radius, 0, -1, 8);
    circle(input, c3, radius, 0, -1, 8);
    circle(input, c4, radius, 0, -1, 8);
    line(input, center, Point(0, center.y), 0, 4);
    line(input, c4, Point(0, c4.y), 0, 4);
}

void calcPSD(const Mat& inputImg, Mat& outputImg, int flag)
{
    // 取 PSD(power spectrum density) 跟相位, 然後做 idft 轉換回來
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);

    planes[0].at<float>(0) = 0;
    planes[1].at<float>(0) = 0;

    Mat imgPSD;
    magnitude(planes[0], planes[1], imgPSD);
    pow(imgPSD, 2, imgPSD);
    outputImg = imgPSD;

    if (flag)
    {
        Mat imglogPSD;
        imglogPSD = imgPSD + Scalar::all(1);
        log(imglogPSD, imglogPSD);
        outputImg = imglogPSD;
    }
}
Mat showspec(Mat I) {
    Mat padded;
    // 轉換成易計算的大小
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols);
    // 多出來的補0
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    // 把兩個通道合併做dft
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    split(complexI, planes);
    Mat transform_image_real = planes[0];
    Mat transform_image_imag = planes[1];
    Mat magI;
    // 取幅值大小
    magnitude(planes[0], planes[1], magI);

    // 取對數讓值的範圍小一點
    magI += Scalar::all(1);
    log(magI, magI);
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // 做位移然後 normalize 到正常範圍內
    shift(magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX);
    return magI;
}

Mat shift(Mat img) {
    int row = img.rows / 2;
    int col = img.cols / 2;
    // 做象限對調(1->4, 2->3)
    Mat q1(img, Rect(0, 0, col, row));
    Mat q2(img, Rect(col, 0, col, row));
    Mat q3(img, Rect(0, row, col, row));
    Mat q4(img, Rect(col, row, col, row));
    Mat tmp;
    q1.copyTo(tmp);
    q4.copyTo(q1);
    tmp.copyTo(q4);

    q3.copyTo(tmp);
    q2.copyTo(q3);
    tmp.copyTo(q2);
    return img;
}
