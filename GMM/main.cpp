#include <iostream>
#include <string>

#include "global_variable.h"
#include "GMM_BGS.h"
#include "CCFilter.h"
#include "tracker.h"
#include "TrackerController.h"
#include "callback_function.h"
#include "ImageSubplot.h"
#include "ShadowDetector.h"

#ifdef NDEBUG
#pragma comment(lib, "opencv_core2411")
#pragma comment(lib, "opencv_imgproc2411")
#pragma comment(lib, "opencv_highgui2411")
#pragma comment(lib, "opencv_contrib2411")
#pragma comment(lib, "opencv_video2411")
#else
#pragma comment(lib, "opencv_core2411d")
#pragma comment(lib, "opencv_imgproc2411d")
#pragma comment(lib, "opencv_highgui2411d")
#pragma comment(lib, "opencv_contrib2411d")
#pragma comment(lib, "opencv_video2411d")
#endif

//extern int g_GlobalFrameCount = 0;

const std::string VIDEO_PATH = "D:/MXFXF/北航资料/我看的论文/Tracking/TestDataFromCAUC/00003.MTS";
//***************************************************************
//**** 窗口的参数***********************************************
const std::string WINDOW_NAME = "GMM";
const std::string TRACKBAR_NAME = "Frames";
const int WINDOW_HEIGHT = 980;
const int WINDOW_WIDTH = 1730;

// 局部非嵌套计时器
int64			__t1;
int64			__t2;
#define	TIC	(__t1 = cv::getTickCount())
#define TOC (__t2 = cv::getTickCount(),\
	std::cout << '\t' << (__t2 - __t1) * 1000 / cv::getTickFrequency() << "ms")

int main()
{
	//******************************************************
	//**** 变量定义 ****************************************
	// video input & output
	cv::VideoCapture	capture(VIDEO_PATH);
	assert(capture.isOpened());
#ifdef FIXED_WIDTH_HEIGHT
	const int kImageWidth		= 960;
	const int kImageHeight		= 544;
#else
	const int kImageWidth		= (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	const int kImageHeight		= (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
#endif
	int kFrameCount				= (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	double kFPS					= capture.get(CV_CAP_PROP_FPS);
	// 测试一下kFrameCount准不准
	capture.set(CV_CAP_PROP_POS_FRAMES, kFrameCount - 1);
	if(!capture.read(cv::Mat()) || capture.read(cv::Mat()))	// 如果读不到最后一帧，说明实际帧数比这个少，要重测
	{
		double secs = (double)kFrameCount / kFPS;
		kFrameCount = 0;
		capture.set(CV_CAP_PROP_POS_FRAMES, 0.0);
		while(capture.read(cv::Mat()))
		{
			++kFrameCount;
		}
		kFPS = (kFrameCount / secs);
		capture.set(CV_CAP_PROP_POS_FRAMES, 0.0);
	}
	// 最大对应距离。预测的object center与之前的object center距离不能超过这个值
	const int kMaxCorrespondDistance = (kImageHeight < kImageWidth ? kImageHeight : kImageWidth) / 10;
	// 跟踪丢失帧数上限，更大时放弃跟踪
	const int kMaxLostCount		= (int)(2 * kFPS);	// 最多丢失两秒

	cv::VideoWriter	writer("Result.avi", CV_FOURCC_DEFAULT, kFPS, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT) ,true);
	assert(writer.isOpened());

#ifdef FIXED_WIDTH_HEIGHT
	cv::Mat			raw_img;
#endif
	// image variable
	cv::Mat			current_img;
	cv::Mat			raw_binary_img;
	cv::Mat			final_binary_img;
	cv::Mat			foreground_img;
	cv::Mat			buffer_img(kImageHeight, kImageWidth, CV_8UC3);
	cv::Mat			big_img(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
	big_img.setTo(127);
	// background model, shadow detector & filter
#ifdef NDEBUG
	GMM_BGS			gmm_model(kImageWidth * kImageHeight, (int)(kFPS * 5), (int)(kFPS * 5));	// 背景模型
#else
	GMM_BGS			gmm_model(kImageWidth * kImageHeight, (int)(kFPS), (int)(kFPS * 5));	// 背景模型
#endif

#ifdef SHADOW_DETECTION
	ShadowDetector shadow_detector;
	cv::Mat			background_img(kImageHeight, kImageWidth, CV_8UC3);
	cv::Mat			hsv_img;
#endif

	CCFilter		cc_filter(kImageWidth, kImageHeight);						// 连通域滤波器
	int				foreground_area_threshold	= (int)(kImageWidth * kImageHeight * 0.0005);	// 前景连通域面积的下限，小于此面积则并入周边连通域
	int				background_area_threshold	= (int)(kImageWidth * kImageHeight * 0.001);	// 背景连通域面积的下限，小于此面积则并入周边连通域
	// tracker
	std::vector<BoundingBox> bdboxes;						// 包围盒
	TrackerController tracker_controller(kMaxLostCount, kMaxCorrespondDistance);
	BoundingBox		tracked_boundingbox;
	// time counter
	int64			tcurrent = cv::getTickCount();
	int64			tlast = tcurrent;
	double			t_frequency = cv::getTickFrequency();

	
	//********************************************************
	//**** 窗口、进度条与响应函数的注册 ***********************************
	cv::namedWindow(WINDOW_NAME);
	cv::moveWindow(WINDOW_NAME, 0, 0);
	char text_buffer[64] = {'\0'};
	MouseClickEvent	click_event;							// 鼠标左键点击时，记录要跟踪的点的位置
	//cv::setMouseCallback(WINDOW_NAME, on_mouse, &click_event);
	bool trackbar_changed = false;
	cv::createTrackbar(TRACKBAR_NAME, WINDOW_NAME, NULL, kFrameCount, on_trackbar, &trackbar_changed);	
	char key_event = 0;
	std::cout << std::fixed << std::setprecision(1);	// 设置输出格式 

	// 训练
	int i = 0;
	//capture.set(CV_CAP_PROP_POS_FRAMES, i);
	while(!gmm_model.isTrainedReady() && i < kFrameCount)
	{
		//g_GlobalFrameCount = i;
		// 调整trackbar
		if(trackbar_changed)	
		{
			i = cv::getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME);
			capture.set(CV_CAP_PROP_POS_FRAMES, i);
		}
		else
		{
			cv::setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, i);
		}
		trackbar_changed = false;
		// 获取图像
#ifdef FIXED_WIDTH_HEIGHT
		capture >> raw_img;
		cv::resize(raw_img, current_img, cv::Size(kImageWidth, kImageHeight));
#else
		capture >> current_img;
#endif
		// 训练
		TIC;
		gmm_model.Training(current_img, raw_binary_img);
		final_binary_img = raw_binary_img;
		TOC;
		// 前景
		foreground_img.setTo(0);
		current_img.copyTo(foreground_img, final_binary_img);
		// 显示
		ImageSubplot::subplot(big_img, current_img, 2, 2, 1);																	// 1号位
		cv::cvtColor(raw_binary_img, buffer_img, CV_GRAY2BGR),		ImageSubplot::subplot(big_img, buffer_img, 2, 2, 2);		// 2号位
		cv::cvtColor(final_binary_img, buffer_img, CV_GRAY2BGR),	ImageSubplot::subplot(big_img, buffer_img, 2, 2, 3);		// 3号位
		ImageSubplot::subplot(big_img, foreground_img, 2, 2, 4);																// 4号位
		sprintf(text_buffer, "change count: %d (max stable count: %d)", gmm_model.getChangeCount(), gmm_model.getMaxStableChangeCount());
		cv::putText(big_img, text_buffer, cv::Point(30, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK);
		cv::imshow(WINDOW_NAME, big_img);
#ifdef BACKGROUND_SHOW
		gmm_model.getBackground(foreground_img);
		cv::imshow("bakcground1", foreground_img);
		gmm_model.getBackground(foreground_img, 1);
		cv::imshow("bakcground2", foreground_img);
		gmm_model.getBackground(foreground_img, 2);
		cv::imshow("bakcground3", foreground_img);
		gmm_model.getBackgroundWeight(foreground_img);	// 试验一下，证明三通道的图像也可以正常显示
		cv::imshow("bgweight1", foreground_img);
		gmm_model.getBackgroundWeight(raw_binary_img, 1);
		cv::imshow("bgweight2", raw_binary_img);
		gmm_model.getBackgroundWeight(raw_binary_img, 2);
		cv::imshow("bgweight3", raw_binary_img);
#endif
		key_event = cv::waitKey(1);
		if(' ' == key_event)
		{
			cv::waitKey();
		}
		else if(27 == key_event)
		{
			break;
		}
		// 计时
		tcurrent = cv::getTickCount();
		std::cout << '\t' << (tcurrent - tlast) * 1000 / t_frequency << "ms" << std::endl;
		tlast = tcurrent;

		++i;
	}
	final_binary_img.release();	// 解除其与raw_binary_img的绑定
	int trained_framecount = i;
	std::cout << std::endl << "训练完毕！ 共训练" << i << "帧。" << std::endl;

	// 检测
	while(i < kFrameCount)
	{
		//g_GlobalFrameCount = i;
		std::cout << " " << i << " ";
		// 调整trackbar
		if(trackbar_changed)	
		{
			i = cv::getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME);
			capture.set(CV_CAP_PROP_POS_FRAMES, i);
		}
		else
		{
			cv::setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, i);
		}
		trackbar_changed = false;
		// 获取图像
#ifdef FIXED_WIDTH_HEIGHT
		capture >> raw_img;
		cv::resize(raw_img, current_img, cv::Size(kImageWidth, kImageHeight));
#else
		capture >> current_img;
#endif
		// 前景检测		12ms
		TIC;
		fflush(stdout);
		gmm_model.Detecting(current_img, raw_binary_img);
		TOC;
#ifdef SHADOW_DETECTION		
		// 阴影检测
		gmm_model.getBackground(background_img);
		//cv::imshow("bkgrd", background_img);
		//gmm_model.getBackground(foreground_img);
		//cv::imshow("bakcground1", foreground_img);
		cv::cvtColor(background_img, background_img, CV_BGR2HSV);
		cv::cvtColor(current_img, hsv_img, CV_BGR2HSV);
		shadow_detector.MarkShadowOnly(background_img, hsv_img, raw_binary_img, COLOR_BLACK.val[0]);
#endif
		// 滤波		4ms
		TIC;
		cc_filter.filter(raw_binary_img, final_binary_img, background_area_threshold, foreground_area_threshold);
		TOC;

		// 前景
		foreground_img.setTo(0);
		current_img.copyTo(foreground_img, final_binary_img);

		// 检测包围盒并画在前景图像上	4ms
		TIC;		
		cc_filter.detect(final_binary_img, bdboxes);
		//BoundingBox::DrawingBoundingBoxes(foreground_img, bdboxes, COLOR_WHITE);
		//BoundingBox::DrawingBoundingBoxes(foreground_img, bdboxes, cv::Scalar::all(-1));	// 没什么卵用
		TOC;
		// 根据包围盒(observation)，更新tracker，并绘制轨迹
		tracker_controller.Update(current_img, bdboxes);
		tracker_controller.DrawTrajectories(foreground_img);
		tracker_controller.DrawBoundingBoxes(foreground_img);
		//tracker_controller.DrawTrajectories(foreground_img, COLOR_RED);
		// 显示		10ms
		TIC;
		ImageSubplot::subplot(big_img, current_img, 2, 2, 1);																	// 1号位
		cv::cvtColor(raw_binary_img, buffer_img, CV_GRAY2BGR),		ImageSubplot::subplot(big_img, buffer_img, 2, 2, 2);		// 2号位
		cv::cvtColor(final_binary_img, buffer_img, CV_GRAY2BGR),	ImageSubplot::subplot(big_img, buffer_img, 2, 2, 3);		// 3号位
		ImageSubplot::subplot(big_img, foreground_img, 2, 2, 4);																// 4号位
		sprintf(text_buffer, "detecting");
		cv::putText(big_img, text_buffer, cv::Point(30, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK);
		cv::imshow(WINDOW_NAME, big_img);
		writer.write(big_img);
#ifdef BACKGROUND_SHOW
		gmm_model.getBackground(foreground_img);
		cv::imshow("bakcground1", foreground_img);
		gmm_model.getBackground(foreground_img, 1);
		cv::imshow("bakcground2", foreground_img);
		gmm_model.getBackground(foreground_img, 2);
		cv::imshow("bakcground3", foreground_img);
		gmm_model.getBackgroundWeight(raw_binary_img);
		cv::imshow("bgweight1", raw_binary_img);
		gmm_model.getBackgroundWeight(raw_binary_img, 1);
		cv::imshow("bgweight2", raw_binary_img);
		gmm_model.getBackgroundWeight(raw_binary_img, 2);
		cv::imshow("bgweight3", raw_binary_img);
#endif
		TOC;
		//printf(" tracker_num: %d, ob_num: %d ", tracker_controller.trackers_.size(), bdboxes.size());
		key_event = cv::waitKey(1);		// 10ms
		if(' ' == key_event)
		{
			cv::waitKey();
		}
		else if(27 == key_event)
		{
			break;
		}
		// 更新		17ms
		TIC;
		if(i - trained_framecount > (int)(2 * kFPS))
		{
			BoundingBox::DrawingBoundingBoxes(final_binary_img, bdboxes, COLOR_WHITE, CV_FILLED);
		}
		gmm_model.Updating(current_img, final_binary_img);
		TOC;
		// 计时
		tcurrent = cv::getTickCount();
		std::cout << '\t' << (tcurrent - tlast) * 1000 / t_frequency << "ms" << std::endl;
		tlast = tcurrent;

		++i;
	}
	cv::setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, i);
	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}