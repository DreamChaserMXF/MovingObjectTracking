#include "ShadowDetector.h"

const double ShadowDetector::kShadowDetetionAlpha_ = 0.4;
const double ShadowDetector::kShadowDetetionBeta_ = 0.8;
const int ShadowDetector::kShadowDetectionPI_S_ = 75;
const int ShadowDetector::kShadowDetectionPI_H_ = 25;

void ShadowDetector::MarkShadowOnly(const cv::Mat& hsv_background, const cv::Mat& hsv_img, cv::Mat& mask, uchar shadow_color) const
{
	double ratio_V = 0.0;
	int diff_S = 0;
	int distance_H1 = 0;
	int distance_H2 = 0;
	int min_distance_H = 0;

	int width = hsv_img.size().width;
	int height = hsv_img.size().height;

	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < width; ++j)
		{
			ratio_V = (double)hsv_img.at<cv::Vec3b>(i, j).val[2] / (double)hsv_background.at<cv::Vec3b>(i, j).val[2];
			diff_S = abs((int)hsv_img.at<cv::Vec3b>(i, j).val[1] - (int)hsv_background.at<cv::Vec3b>(i, j).val[1]);
			distance_H1 = abs((int)hsv_img.at<cv::Vec3b>(i, j).val[0] - (int)hsv_background.at<cv::Vec3b>(i, j).val[0]);
			distance_H2 = 180 - distance_H1;
			min_distance_H = distance_H1 < distance_H2 ? distance_H1 : distance_H2;

			if(ratio_V >= kShadowDetetionAlpha_ && ratio_V <= kShadowDetetionBeta_
				&& diff_S <= kShadowDetectionPI_S_ 
				&& min_distance_H <= kShadowDetectionPI_H_)
			{
				mask.at<uchar>(i, j) = shadow_color;
			}
		}
	}
}