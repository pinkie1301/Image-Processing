#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void fftshift(const Mat& inputImg, Mat& outputImg);
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag = 0);// calculates power spectrum density of an image
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

    Mat imgPSD;
    calcPSD(imgIn, imgPSD);
    fftshift(imgPSD, imgPSD);
    normalize(imgPSD, imgPSD, 0, 255, NORM_MINMAX);
    Mat spec = showspec(imgIn);
    imshow("spec", spec);

    waitKey(0);
    return 0;
}

void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;

    // 做象限對調(1->4, 2->3)
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

    Mat imgPSD, imgPhase;
    magnitude(planes[0], planes[1], imgPSD);
    phase(planes[0], planes[1], imgPhase);
    pow(imgPSD, 2, imgPSD);
    outputImg = imgPSD;
    log(imgPhase, imgPhase);
    imshow("phase", imgPhase);

    Mat ifft, out;
    Mat planesidft[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    planesidft[0].at<float>(0) = 0;
    planesidft[1].at<float>(0) = 0;
    idft(complexI, ifft);
    split(ifft, planesidft);
    out = planesidft[0];
    normalize(out, out, 0, 1, NORM_MINMAX);
    imshow("idft", out);
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
