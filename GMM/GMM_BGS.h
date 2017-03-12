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

//����˹ģ�ͽṹ��
typedef struct GaussianBackgroundModel
{
	cv::Vec3d mean;
	cv::Vec3d std;
	double weight;
} GM;

//��ϸ�˹ģ�ͽṹ�壬����K������˹ģ�ͺͶ�Ӧλ�õ����ء�ƥ��ģ�͵��±���Ƿ�Ϊǰ��
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
	// �ҵ���ƥ����Ǹ���˹ģ�ͣ��������Ƿ����ڱ���ƥ��
	bool FindMatchingIndex(const cv::Vec3b& pixel);
	void CreateNewGaussian(const cv::Vec3b& pixel);
	void AdjustParameters(const cv::Vec3b& pixel, double learning_rate);

	int matching_index_;		// ƥ���ģ���±�,δƥ����ΪK�������ж��Ƿ���滻�µ�ģ�͡�Ϊ���¸�˹ģ�Ͳ���ʱ�ṩ��Ϣ
};


class GMM_BGS
{
public:
	enum{BACKGROUND = 0, FOREGROUND = 255};
	GMM_BGS(int img_area, int min_training_count, int merge_framecount);
	~GMM_BGS(void);
	bool isTrainedReady() const;
	void Training(const cv::Mat& img, cv::Mat& mask);		// ѵ��������ǰ�����ر任����
	void Detecting(const cv::Mat& img, cv::Mat& mask);		// ���
	void Updating(const cv::Mat& img, const cv::Mat& mask);	// ����
	void getBackground(cv::Mat& img, int index = 0);		// ��ȡ����index��ģ�����ͼ��Ҫ��img�Ѿ���ʼ����
	void getBackgroundWeight(cv::Mat& img, int index = 0);	// ��ȡ����index��ģ�����ͼ��Ҫ��img�Ѿ���ʼ������Ȩ�أ���������0~255
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