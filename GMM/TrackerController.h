#ifndef TRACKERCONTROLLER_H
#define TRACKERCONTROLLER_H
#include "tracker.h"
#include "BoundingBox.h"
#include <vector>

const double BHATTACHARYA_DISTANCE_THRESHOLD = 0.5;

// according to 《2009_ICRA_Real-Time Object Tracking and Classiﬁcation Using a Static Camera【A priori matching】》
class TrackerController
{
public:
	TrackerController(int max_missing_count, int max_corresponding_distance);	// 最大丢失数，最大对应距离
	~TrackerController();
	void Update(const cv::Mat& img, const std::vector<BoundingBox>& observations);
	void DrawTrajectories(cv::Mat& img);
	//void DrawTrajectories(cv::Mat& img, cv::Scalar color);
	void DrawBoundingBoxes(cv::Mat& img);
	const std::list<Tracker>& getTrackers() const {return trackers_;}

private:
	const static cv::Scalar kColorPalette_[];
	const static int kPaletteSize_;
	static cv::RNG rng;

	void FillDistanceMatrix(cv::Mat& dist_mat, const std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations) const;
	void FillDecisionMatrix(cv::Mat& decision_mat, const cv::Mat& distance_mat) const;
	void GetObservationHistogram(const cv::Mat& img, const std::vector<BoundingBox>& observations, std::vector<cv::MatND>& ob_hists) const;
	double BhattacharyaDistance(const cv::Mat& hist1, const cv::Mat& hist2) const;
	void MakeDistanceCorrespondence(std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, cv::Mat& distance_mat) const;
	void FilterCorrespondenceByArea(const std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations_, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded) const;
	void FilterCorrespondenceByHistogram(std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::list<Tracker>& trackers, const std::vector<cv::MatND>& ob_hists) const;
	void UpdateMatchedTracker(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, const std::vector<int>& corresponding_ob_idx, const std::vector<cv::MatND>& ob_hists);
	void SplittingHandling(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers);
	void MergingHandling(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers);
	void CreateNewTrackerForUnmatchedObservation(const std::vector<BoundingBox>& observations, const std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers);
	void UpdateUnmatchedTracker(const std::vector<int>& corresponding_ob_idx);
	void FilterSplittedMergedTracker(const std::vector<int>& corresponding_ob_idx);
	void FilterLostTracker();

	int getDistanceSquare(cv::Point p1, cv::Point p2) const;

	std::list<Tracker> trackers_;
	const int kMaxMissingCount_;
	const int kMaxCorrespondingDistance_;
	
};

#endif