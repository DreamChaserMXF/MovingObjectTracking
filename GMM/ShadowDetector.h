#ifndef SHADOWDETECTOR_H
#define SHADOWDETECTOR_H

#include "global_variable.h"
#include <opencv.hpp>
class ShadowDetector
{
public:
	void MarkShadowOnly(const cv::Mat& hsv_background, const cv::Mat& hsv_img, cv::Mat& mask, uchar shadow_color) const;
private:
	static const double kShadowDetetionAlpha_;
	static const double kShadowDetetionBeta_;
	static const int kShadowDetectionPI_S_;
	static const int kShadowDetectionPI_H_;
};
//void ShadowDetection(const *p_hsv_background, const IplImage *p_hsv_img, IplImage *p_binary_img);

#endif