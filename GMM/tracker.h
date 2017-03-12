#ifndef TRACKER_H
#define TRACKER_H

#include "global_variable.h"
#include "BoundingBox.h"
#include "CCFilter.h"
#include <opencv.hpp>
#include <vector>

class Tracker
{
public:
	friend class TrackerController;

	enum TrackerState{TRACKING = 0x01, MISSING = 0x02};

	Tracker(const BoundingBox& object_bdbox, const cv::MatND& object_hist, cv::Scalar color);	// 一个新的独立的tracker
	//Tracker(const Tracker& origin_tracker, const BoundingBox& object_bdbox, const cv::MatND& object_hist);	// 一个分裂出来的tracker
	Tracker(std::list<Tracker>& origin_trackers, const BoundingBox& object_bdbox, const cv::MatND& object_hist, cv::Scalar color);	// 一个由多个tracker合并起来的tracker
	
	void Update();
	void Update(const BoundingBox& observation, const cv::Mat& measured_ht);
	void HiddenUpdate(cv::Point point);
	void DrawTrajectory(cv::Mat& img, bool draw_cross) const;
	void DrawTrajectory(cv::Mat& img, cv::Scalar color, bool draw_cross) const;
	void DrawBoundingBox(cv::Mat& img) const;
	TrackerState getState() const { return state_; }
	cv::Point getPredictedPoint() const { return predicted_point_; }
	//cv::Point getEstimatedPoint() const { return estimated_point_; }
	int getMissingCount() const { return missing_count_; }
	const cv::Mat& getHistogram() const { return hist_; }
	const BoundingBox& bdbox() const { return bdbox_;}

private:
	void init_kalman();

	TrackerState state_;
	int missing_count_;
	std::list<Tracker> hidden_trackers_;
	std::vector<cv::Point> trajectory_;
	cv::Scalar color_;
	//std::vector<cv::Point> trajectory_;
	cv::KalmanFilter kalman_;
	cv::Point estimated_point_;	// 记录估计的点的位置，仅用于预测missing后的bdbox_的新位置
	cv::Point predicted_point_;
	cv::MatND hist_;	// histogram of the tracked object
	BoundingBox bdbox_;
};

#endif