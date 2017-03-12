#include "tracker.h"

Tracker::Tracker(const BoundingBox& object_bdbox, const cv::MatND& object_hist, cv::Scalar color) : state_(TRACKING), missing_count_(0), hidden_trackers_(), trajectory_(), color_(color), kalman_(), estimated_point_(object_bdbox.object_center_), predicted_point_(), hist_(), bdbox_(object_bdbox)
{
	// 初始化卡尔曼滤波器
	init_kalman();	
	// 设置其估计点的坐标并进行一次预测
	kalman_.statePost = cv::Mat(cv::Vec4f((float)object_bdbox.object_center_.x, (float)object_bdbox.object_center_.y, 0.0f, 0.0f));
	cv::Mat predicted_mat = kalman_.predict();
	predicted_point_.x = (int)predicted_mat.at<float>(0);
	predicted_point_.y = (int)predicted_mat.at<float>(1);
	// 添加轨迹中的第一个点
	trajectory_.push_back(estimated_point_);
	// 拷贝直方图
	hist_ = object_hist;
}


// 一个由多个tracker合并起来的tracker
Tracker::Tracker(std::list<Tracker>& origin_trackers, const BoundingBox& object_bdbox, const cv::MatND& object_hist, cv::Scalar color) 
	: state_(TRACKING), missing_count_(0), hidden_trackers_(), trajectory_(), color_(color), kalman_(), estimated_point_(object_bdbox.object_center_), predicted_point_(), hist_(), bdbox_(object_bdbox)
{
	// 初始化卡尔曼滤波器
	init_kalman();		
	// 设置其估计点的坐标并进行一次预测
	kalman_.statePost = cv::Mat(cv::Vec4f((float)object_bdbox.object_center_.x, (float)object_bdbox.object_center_.y, 0.0f, 0.0f));	
	cv::Mat predicted_mat = kalman_.predict();
	predicted_point_.x = (int)predicted_mat.at<float>(0);
	predicted_point_.y = (int)predicted_mat.at<float>(1);
	
	// 处理hidden tracker
	for(std::list<Tracker>::iterator iter = origin_trackers.begin();
		iter != origin_trackers.end(); ++iter)
	{
		iter->trajectory_.push_back(estimated_point_);	// 将当前点加入之前的轨迹，以保证轨迹连续
		if(iter->hidden_trackers_.size() > 0)	// 被合并的tracker还带有hidden tracker的情况
		{
			// 把fake tracker的路径添加到hidden tracker中
			for(std::list<Tracker>::iterator hidden_tracker_iter = iter->hidden_trackers_.begin();
				hidden_tracker_iter != iter->hidden_trackers_.end(); ++hidden_tracker_iter)
			{
				hidden_tracker_iter->trajectory_.insert(hidden_tracker_iter->trajectory_.end(), iter->trajectory_.begin(), iter->trajectory_.end());
			}
			// 把hidden tracker添加到当前对象的hidden_tracker中
			hidden_trackers_.splice(hidden_trackers_.end(), iter->hidden_trackers_);
		}
		else
		{
			hidden_trackers_.push_back(*iter);
		}
	}
	// 添加轨迹中的第一个点
	trajectory_.push_back(estimated_point_);
	// 拷贝直方图
	hist_ = object_hist;
}

void Tracker::Update()
{
	// 因没有测量值，故用预测点代替估计点
	// 但不显式记入路径
	cv::Mat predicted_mat = kalman_.predict();
	predicted_point_.x = (int)predicted_mat.at<float>(0);
	predicted_point_.y = (int)predicted_mat.at<float>(1);
	//trajectory_.front().push_back(predicted_point_);

	// 预测bdbox_的下一个位置
	bdbox_.top_left_.x += predicted_point_.x - estimated_point_.x;
	bdbox_.bottom_right_.x += predicted_point_.x - estimated_point_.x;
	bdbox_.top_left_.y += predicted_point_.y - estimated_point_.y;
	bdbox_.bottom_right_.y += predicted_point_.y - estimated_point_.y;

	// 用预测点更新估计点
	estimated_point_ = predicted_point_;

	// 更新状态
	++missing_count_;
	state_ = MISSING;
}

void Tracker::Update(const BoundingBox& observation, const cv::Mat& measured_ht)
{
	// 状态更新
	missing_count_ = 0;
	state_ = TRACKING;
	// 位置估计
	cv::Mat estimated_mat = kalman_.correct(cv::Mat(cv::Vec2f((float)observation.object_center_.x, (float)observation.object_center_.y)));
	estimated_point_.x = (int)estimated_mat.at<float>(0);
	estimated_point_.y = (int)estimated_mat.at<float>(1);
	trajectory_.push_back(estimated_point_);
	// 下一个位置的预测
	cv::Mat predicted_mat = kalman_.predict();
	predicted_point_.x = (int)predicted_mat.at<float>(0);
	predicted_point_.y = (int)predicted_mat.at<float>(1);
	// 直方图更新
	hist_ = measured_ht;
	// 包围盒更新
	bdbox_ = observation;
}

void Tracker::HiddenUpdate(cv::Point point)
{
	// 只更新位置信息
	cv::Mat estimated_mat = kalman_.correct(cv::Mat(cv::Vec2f((float)point.x, (float)point.y)));
	estimated_point_.x = (int)estimated_mat.at<float>(0);
	estimated_point_.y = (int)estimated_mat.at<float>(1);
	trajectory_.push_back(estimated_point_);
	// 下一个位置的预测
	cv::Mat predicted_mat = kalman_.predict();
	predicted_point_.x = (int)predicted_mat.at<float>(0);
	predicted_point_.y = (int)predicted_mat.at<float>(1);
}

void DrawCross(cv::Mat& img, cv::Point point, int corss_radius, cv::Scalar color)
{
	cv::line(img, cv::Point(point.x - corss_radius, point.y - corss_radius), cv::Point(point.x + corss_radius, point.y + corss_radius), color, 2);
	cv::line(img, cv::Point(point.x - corss_radius, point.y + corss_radius), cv::Point(point.x + corss_radius, point.y - corss_radius), color, 2);
}

void Tracker::DrawTrajectory(cv::Mat& img, bool draw_cross) const
{
	// 当前tracker的轨迹
	const cv::Point *point_trajectory = trajectory_.data();
	int npts = trajectory_.size();
	cv::polylines(img, &point_trajectory, &npts, 1, false, color_);
	// hidden tracker的轨迹
	for(std::list<Tracker>::const_iterator hidden_tracker_iter = hidden_trackers_.begin();
		hidden_tracker_iter != hidden_trackers_.end(); ++hidden_tracker_iter)
	{
		hidden_tracker_iter->DrawTrajectory(img, color_, false);
	}
	// 当前点
	if(draw_cross)
	{
		DrawCross(img, trajectory_.back(), 4, color_);
	}
}


void Tracker::DrawTrajectory(cv::Mat& img, cv::Scalar color, bool draw_cross) const
{
	// 当前tracker的轨迹
	const cv::Point *point_trajectory = trajectory_.data();
	int npts = trajectory_.size();
	cv::polylines(img, &point_trajectory, &npts, 1, false, color);
	// hidden tracker的轨迹
	for(std::list<Tracker>::const_iterator hidden_tracker_iter = hidden_trackers_.begin();
		hidden_tracker_iter != hidden_trackers_.end(); ++hidden_tracker_iter)
	{
		hidden_tracker_iter->DrawTrajectory(img, color, false);
	}
	// 当前点
	if(draw_cross)
	{
		DrawCross(img, trajectory_.back(), 4, color);
	}
}

void Tracker::DrawBoundingBox(cv::Mat& img) const
{
	cv::rectangle(img, bdbox_.top_left_, bdbox_.bottom_right_, color_);
}


void Tracker::init_kalman()
{
	float transition_mat[] = {
	1, 0, 1, 0,
	0, 1, 0, 1,
	0, 0, 1, 0,
	0, 0, 0, 1,
	};
	kalman_.init(4, 2);
	memcpy(kalman_.transitionMatrix.data, transition_mat, sizeof(transition_mat));
	//kalman_.transitionMatrix = cv::Mat(4, 4, CV_32F, transition_mat);
	cv::setIdentity(kalman_.measurementMatrix);
	cv::setIdentity(kalman_.processNoiseCov, 1e-5);
	cv::setIdentity(kalman_.measurementNoiseCov, 1e-1);
	cv::setIdentity(kalman_.errorCovPost);
}