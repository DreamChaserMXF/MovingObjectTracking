#include "GMM_BGS.h"

GMM::GMM() : matching_index_(K)
{
	for(int i = 0; i < K; ++i)
	{
		gaussian_[i].mean = cv::Vec3d(cv::RNG().uniform(0, 255), cv::RNG().uniform(0, 255), cv::RNG().uniform(0, 255));
		gaussian_[i].std = cv::Vec3d(INITIAL_STD, INITIAL_STD, INITIAL_STD);
		gaussian_[i].weight = 1.0 / K;
	}
}

bool GMM::Training(const cv::Vec3b& pixel, double learning_rate)
{
	bool is_background = FindMatchingIndex(pixel);
	Updating(pixel, learning_rate);
	return is_background;
}

bool GMM::Detecting(const cv::Vec3b& pixel)
{
	return FindMatchingIndex(pixel);
}

void GMM::Updating(const cv::Vec3b& pixel, double learning_rate)
{
	if(K == matching_index_)
	{
		CreateNewGaussian(pixel);
	}
	else
	{
		AdjustParameters(pixel, learning_rate);
	}
}

bool GMM::FindMatchingIndex(const cv::Vec3b& pixel)
{
	double accumulate_weight(0.0f);
	int i;
	for(i = 0; i < K; ++i)
	{
		// �����ضԵ�i����˹ģ�͵�������
		if(		abs(pixel[0] - gaussian_[i].mean[0]) / gaussian_[i].std[0] <= MATCH_THRESHOLD
			&&  abs(pixel[1] - gaussian_[i].mean[1]) / gaussian_[i].std[1] <= MATCH_THRESHOLD
			&&  abs(pixel[2] - gaussian_[i].mean[2]) / gaussian_[i].std[2] <= MATCH_THRESHOLD)
		{
			break;
		}
		accumulate_weight += gaussian_[i].weight;
	}
	matching_index_ = i;	// ��ͨ���ж�matching_index�Ƿ����K�����ж��Ƿ���ĳ��ƥ��
	if(accumulate_weight > T)
	{
		return false;
	}
	else
	{
		return true;
	}
}

void GMM::CreateNewGaussian(const cv::Vec3b& pixel)
{
	gaussian_[K-1].mean[0] = pixel[0];
	gaussian_[K-1].mean[1] = pixel[1];
	gaussian_[K-1].mean[2] = pixel[2];
	gaussian_[K-1].std[0] = INITIAL_STD;
	gaussian_[K-1].std[1] = INITIAL_STD;
	gaussian_[K-1].std[2] = INITIAL_STD;
}

void GMM::AdjustParameters(const cv::Vec3b& pixel, double learning_rate)
{
	// ���¾�ֵ
	gaussian_[matching_index_].mean *= 1 - RHO;
	gaussian_[matching_index_].mean += cv::Vec3d(pixel) * RHO;
	//gaussian_[matching_index_].mean[0] += RHO * pixel[0];
	//gaussian_[matching_index_].mean[1] += RHO * pixel[1];
	//gaussian_[matching_index_].mean[2] += RHO * pixel[2];
	
	//���±�׼����ɱ�׼��������ޣ����׼����º����ʱ�Ÿ��£�
	double std_b = abs(pixel[0] - gaussian_[matching_index_].mean[0]);
	double std_g = abs(pixel[1] - gaussian_[matching_index_].mean[1]);
	double std_r = abs(pixel[2] - gaussian_[matching_index_].mean[2]);
	if(gaussian_[matching_index_].std[0] > STD_FLOOR
		|| std_b > STD_FLOOR)
	{
		gaussian_[matching_index_].std[0] = sqrt(
			(1 - RHO) * gaussian_[matching_index_].std[0] * gaussian_[matching_index_].std[0]
			+ RHO * std_b * std_b );
	}
	if(gaussian_[matching_index_].std[1] > STD_FLOOR
		|| std_g > STD_FLOOR)
	{
		gaussian_[matching_index_].std[1] = sqrt(
			(1 - RHO) * gaussian_[matching_index_].std[1] * gaussian_[matching_index_].std[1]
			+ RHO * std_g * std_g );
	}
	if(gaussian_[matching_index_].std[2] > STD_FLOOR
		|| std_r > STD_FLOOR)
	{
		gaussian_[matching_index_].std[2] = sqrt(
			(1 - RHO) * gaussian_[matching_index_].std[2] * gaussian_[matching_index_].std[2]
			+ RHO * std_r * std_r );
	}

	// ����Ȩ�ؼ���˹ģ�͵�˳��
	// ��ƥ�䵽��Ȩ�طǳ���ʱ�Ͳ�ȥ�����ˣ���Ϊ��ʱ������СȨ��(10�ĸ���ʮ�η�)�ļ���ǳ�����ʱ�䣬����Ҳû��̫���Ҫȥ����
	if(gaussian_[matching_index_].weight < MAX_WEIGHT)	
	{
		// ����Ȩ��
		for(int i = 0 ;i < K; ++i)
		{
			gaussian_[i].weight *= 1 - learning_rate;		//�������ֲ���Ȩ�س���ϵ��
		}
		gaussian_[matching_index_].weight += learning_rate;
		// ����˳��
		GM matching_model = gaussian_[matching_index_];
		int i = matching_index_;
		while(i > 0 && matching_model.weight > gaussian_[i - 1].weight)
		{
			gaussian_[i] = gaussian_[i - 1];
			--i;
		}
		if(i != matching_index_)
		{
			gaussian_[i] = matching_model;
		}
	}
}

GMM_BGS::GMM_BGS(int img_area, int min_training_count, int merge_framecount) : 
	kMaxStableChangeCount((int)(img_area * 0.0005)), change_count_(0), lastframe_foreground_pixel_count_(0),
	min_training_count_(min_training_count), training_framecounter_(0), training_learning_rate_(1 - pow(1 - T, 1.5 / min_training_count)),/*�����ָ���ķ���ԭ����1.0�����ڱ��1.5��Ϊ���Լ�������*/
	img_area_(img_area), 
	background_model_((GMM*)calloc(img_area, sizeof(GMM))), 
	merge_framecount_(merge_framecount), foreground_counter_((int*)calloc(img_area, sizeof(int)))
{ }
GMM_BGS::~GMM_BGS(void)
{ 
	free(background_model_);
	free(foreground_counter_);
}

bool GMM_BGS::isTrainedReady() const
{
	//return (training_framecounter_ >= min_training_count_ && change_count_ <= kMaxStableChangeCount);
	return (training_framecounter_ >= min_training_count_);
}

void GMM_BGS::Training(const cv::Mat& img, cv::Mat& mask)		// ѵ��
{
	assert(img_area_ == img.size().area());
	if(mask.empty())
	{
		mask = cv::Mat::zeros(img.size(), CV_8UC1);
	}
	int i = 0;
	int foreground_pixel_count = 0;
	assert(img.isContinuous());
	for(i = 0; i < img_area_; ++i)
	{
		if(background_model_[i].Training(img.at<cv::Vec3b>(i), training_learning_rate_))
		{
			mask.at<uchar>(i) = BACKGROUND;
		}
		else
		{
			mask.at<uchar>(i) = FOREGROUND;
			++foreground_pixel_count;
		}
	}
	// ǰ�����ر任����
	int new_change_count = abs(foreground_pixel_count - lastframe_foreground_pixel_count_);
	change_count_ = (int)(change_count_ * 0.9 + new_change_count * 0.1);	// ָ��ƽ��
	lastframe_foreground_pixel_count_ = foreground_pixel_count;
	// ѵ��֡��
	++training_framecounter_;
}

void GMM_BGS::Detecting(const cv::Mat& img, cv::Mat& mask)			// ���
{
	assert(img_area_ == img.size().area());
	int i;
	assert(img.isContinuous());
	
	for(i = 0; i < img_area_; ++i)
	{
		if(background_model_[i].Detecting(img.at<cv::Vec3b>(i)))
		{
			mask.at<uchar>(i) = BACKGROUND;
			foreground_counter_[i] = 0;
		}
		else
		{
			mask.at<uchar>(i) = FOREGROUND;
			++foreground_counter_[i];
		}
	}
}

void GMM_BGS::Updating(const cv::Mat& img, const cv::Mat& mask)		// ����
{
	assert(img_area_ == img.size().area());
	int i;
	assert(img.isContinuous());
	for(i = 0; i < img_area_; ++i)
	{
		if(foreground_counter_[i] > merge_framecount_
			|| BACKGROUND == mask.at<uchar>(i))
		{
			background_model_[i].Updating(img.at<cv::Vec3b>(i), ALPHA);
		}
	}
}

void GMM_BGS::getBackground(cv::Mat& img, int index)
{
	assert(!img.empty());
	assert(img.isContinuous());
	for(int i = 0; i < img_area_; ++i)
	{
		img.at<cv::Vec3b>(i) = background_model_[i].gaussian_[index].mean;
	}
}

void GMM_BGS::getBackgroundWeight(cv::Mat& img, int index)
{
	assert(!img.empty());
	assert(img.isContinuous());
	if(1 == img.channels())
	{
		for(int i = 0; i < img_area_; ++i)
		{
			img.at<uchar>(i) = (uchar)(background_model_[i].gaussian_[index].weight * 255.0);
		}
	}
	else 
	{
		img.setTo(0);
		for(int i = 0; i < img_area_; ++i)
		{
			img.at<cv::Vec3b>(i)[0] = (uchar)(background_model_[i].gaussian_[index].weight * 255.0);
			img.at<cv::Vec3b>(i)[1] = img.at<cv::Vec3b>(i)[0];
			img.at<cv::Vec3b>(i)[2] = img.at<cv::Vec3b>(i)[0];
		}
	}
}