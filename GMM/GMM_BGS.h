#pragma once
#include <iostream>
#include <opencv.hpp>

const int		INITIAL_STD		= 20;
const double	MATCH_THRESHOLD = 2.5;
const double	T				= 0.75;
const double	ALPHA			= 0.003;
const double	RHO				= 0.25;
const double	STD_FLOOR		= 5.0;
const double	MAX_WEIGHT		= 0.9;

//单高斯模型结构体
typedef struct GaussianBackgroundModel
{
	cv::Vec3d mean;
	cv::Vec3d std;
	double weight;
} GM;

//混合高斯模型结构体，包括K个单高斯模型和对应位置的像素、匹配模型的下标和是否为前景
class GMM
{
public:
	enum{K = 3};
	GMM();
	bool Training(const  cv::Vec3b& pixel, double learning_rate = ALPHA);
	bool Detecting(const cv::Vec3b& pixel);
	void Updating(const  cv::Vec3b& pixel, double learning_rate = ALPHA);

	GM gaussian_[K];			// array of single Gaussian Model, public for easily accessible
private:
	// 找到最匹配的那个高斯模型，返回其是否属于背景匹配
	bool FindMatchingIndex(const cv::Vec3b& pixel);
	void CreateNewGaussian(const cv::Vec3b& pixel);
	void AdjustParameters(const cv::Vec3b& pixel, double learning_rate);

	int matching_index_;		// 匹配的模型下标,未匹配则为K。用于判断是否该替换新的模型、为更新高斯模型参数时提供信息
};


class GMM_BGS
{
public:
	enum{BACKGROUND = 0, FOREGROUND = 255};
	GMM_BGS(int img_area, int min_training_count, int merge_framecount);
	~GMM_BGS(void);
	bool isTrainedReady() const;
	void Training(const cv::Mat& img, cv::Mat& mask);		// 训练，返回前景像素变换个数
	void Detecting(const cv::Mat& img, cv::Mat& mask);		// 检测
	void Updating(const cv::Mat& img, const cv::Mat& mask);	// 更新
	void getBackground(cv::Mat& img, int index = 0);		// 获取（第index层的）背景图像（要求img已经初始化）
	void getBackgroundWeight(cv::Mat& img, int index = 0);	// 获取（第index层的）背景图像（要求img已经初始化）的权重，并放缩至0~255
	int getChangeCount(){return change_count_;}
	int getMaxStableChangeCount(){return kMaxStableChangeCount;}
private:
	const int kMaxStableChangeCount;
	int change_count_;
	int lastframe_foreground_pixel_count_;

	const int min_training_count_;
	int training_framecounter_;
	double training_learning_rate_;

	int img_area_;

	GMM* background_model_;

	const int merge_framecount_;
	int* foreground_counter_;
};