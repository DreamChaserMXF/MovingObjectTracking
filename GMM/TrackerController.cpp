#include "TrackerController.h"

TrackerController::TrackerController(int max_lostcount, int max_corresponding_distance)
	: trackers_(),
	kMaxMissingCount_(max_lostcount), kMaxCorrespondingDistance_(max_corresponding_distance * max_corresponding_distance) /*��Ϊ���Ǽ���ʱʹ�õ���ŷ�Ͼ����ƽ������������Ҫ��һ��*/
{ }

TrackerController::~TrackerController()
{}
const cv::Scalar TrackerController::kColorPalette_[] = 
{
	CV_RGB(64, 64, 64), CV_RGB(64, 64, 128), CV_RGB(64, 64, 192), CV_RGB(64, 64, 255), CV_RGB(64, 128, 64), CV_RGB(64, 128, 128), CV_RGB(64, 128, 192), CV_RGB(64, 128, 255), CV_RGB(64, 192, 64), CV_RGB(64, 192, 128), CV_RGB(64, 192, 192), CV_RGB(64, 192, 255), CV_RGB(64, 255, 64), CV_RGB(64, 255, 128), CV_RGB(64, 255, 192), CV_RGB(64, 255, 255), 
	CV_RGB(128, 64, 64), CV_RGB(128, 64, 128), CV_RGB(128, 64, 192), CV_RGB(128, 64, 255), CV_RGB(128, 128, 64), CV_RGB(128, 128, 128), CV_RGB(128, 128, 192), CV_RGB(128, 128, 255), CV_RGB(128, 192, 64), CV_RGB(128, 192, 128), CV_RGB(128, 192, 192), CV_RGB(128, 192, 255), CV_RGB(128, 255, 64), CV_RGB(128, 255, 128), CV_RGB(128, 255, 192), CV_RGB(128, 255, 255), 
	CV_RGB(192, 64, 64), CV_RGB(192, 64, 128), CV_RGB(192, 64, 192), CV_RGB(192, 64, 255), CV_RGB(192, 128, 64), CV_RGB(192, 128, 128), CV_RGB(192, 128, 192), CV_RGB(192, 128, 255), CV_RGB(192, 192, 64), CV_RGB(192, 192, 128), CV_RGB(192, 192, 192), CV_RGB(192, 192, 255), CV_RGB(192, 255, 64), CV_RGB(192, 255, 128), CV_RGB(192, 255, 192), CV_RGB(192, 255, 255), 
	CV_RGB(255, 64, 64), CV_RGB(255, 64, 128), CV_RGB(255, 64, 192), CV_RGB(255, 64, 255), CV_RGB(255, 128, 64), CV_RGB(255, 128, 128), CV_RGB(255, 128, 192), CV_RGB(255, 128, 255), CV_RGB(255, 192, 64), CV_RGB(255, 192, 128), CV_RGB(255, 192, 192), CV_RGB(255, 192, 255), CV_RGB(255, 255, 64), CV_RGB(255, 255, 128), CV_RGB(255, 255, 192),/* CV_RGB(255, 255, 255), */
};
const int TrackerController::kPaletteSize_ = sizeof(kColorPalette_) / sizeof(kColorPalette_[0]);
cv::RNG TrackerController::rng;


void TrackerController::Update(const cv::Mat& img, const std::vector<BoundingBox>& observations)
{
	int tracker_num = trackers_.size();
	int ob_num = observations.size();
	std::vector<int> corresponding_ob_idx(tracker_num, -1);	// tracker��Ӧ��observation�±�
	std::vector<bool> is_ob_corresponded(ob_num, false);	// observation�Ƿ��й�����tracker
	std::vector<cv::MatND> ob_hists(ob_num);				// observation��Ӧ��ֱ��ͼ
	// 1. �ҵ����е�tracker��observation�Ķ�Ӧ��ϵ
	cv::Mat distance_mat(tracker_num, ob_num, CV_32S);	// ����D
	FillDistanceMatrix(distance_mat, trackers_, observations);
	MakeDistanceCorrespondence(corresponding_ob_idx, is_ob_corresponded, distance_mat);	 // ���ݾ���ȷ����ʼ��ƥ��
	GetObservationHistogram(img, observations, ob_hists);	// �õ�observation��ֱ��ͼ
	FilterCorrespondenceByArea(trackers_, observations, corresponding_ob_idx, is_ob_corresponded);	// ������ͨ��������˵���һ�µ�ƥ��
	FilterCorrespondenceByHistogram(corresponding_ob_idx, is_ob_corresponded, trackers_, ob_hists);	// ����ֱ��ͼ���˵���һ�µ�ƥ��
	// 2. ����splitting��merging
	std::list<Tracker> new_trackers;
#ifdef SPLITTING_MERGING	// (����Ҫ�ѵ�7��������#ifdef��Ҳ��)
	SplittingHandling(trackers_, observations, corresponding_ob_idx, is_ob_corresponded, ob_hists, new_trackers);
	MergingHandling(trackers_, observations, corresponding_ob_idx, is_ob_corresponded, ob_hists, new_trackers);
#endif
	// 3. Ϊ��û��ƥ���observation�����µ�tracker
	CreateNewTrackerForUnmatchedObservation(observations, is_ob_corresponded, ob_hists, new_trackers);
	// 4. ����һһ��Ӧ��tracker
	UpdateMatchedTracker(trackers_, observations, corresponding_ob_idx, ob_hists);
	// 5. ����δƥ���tracker
	UpdateUnmatchedTracker(corresponding_ob_idx);
	// 6. ȥ��������ϲ���tracker
	FilterSplittedMergedTracker(corresponding_ob_idx);
	/**************** ���ˣ���corresponding_ob_idx��is_ob_correspondedλ����صĹ����������꣬���Բ���ԭtrackers_�� *******************/
	// 7. ���˵�MissingCount������ֵ��tracker
	FilterLostTracker();
	// 8. �Ѵ�����new_trackers׷�ӵ���ǰtracker�����ĩβ
	trackers_.splice(trackers_.end(), new_trackers);
}

void TrackerController::DrawTrajectories(cv::Mat& img)
{
	int i = 0;
	for(std::list<Tracker>::const_iterator c_iter = trackers_.begin();
		c_iter != trackers_.end(); ++c_iter, ++i)
	{
		if(Tracker::TRACKING == c_iter->getState())
		{
			c_iter->DrawTrajectory(img, true);
		}
	}
}

//void TrackerController::DrawTrajectories(cv::Mat& img, cv::Scalar color)
//{
//	int i = 0;
//	for(std::list<Tracker>::const_iterator c_iter = trackers_.begin();
//		c_iter != trackers_.end(); ++c_iter, ++i)
//	{
//		if(Tracker::TRACKING == c_iter->getState())
//		{
//			c_iter->DrawTrajectory(img, color);
//		}
//	}
//}

void TrackerController::DrawBoundingBoxes(cv::Mat& img)
{
	int i = 0;
	for(std::list<Tracker>::const_iterator c_iter = trackers_.begin();
		c_iter != trackers_.end(); ++c_iter, ++i)
	{
		if(Tracker::TRACKING == c_iter->getState())
		{
			c_iter->DrawBoundingBox(img);
		}
	}
}

// ÿ�ж�Ӧһ��tracker
// ÿ�ж�Ӧһ��observation(boundingbox)
// ����ľ�����ŷ�Ͼ����ƽ��
void TrackerController::FillDistanceMatrix(cv::Mat& dist_mat, const std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations) const
{
	int i = 0, j = 0;
	cv::Point predicted_point, observed_point;
	for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	{
		j = 0;
		predicted_point = tracker_iter->getPredictedPoint();
		for(std::vector<BoundingBox>::const_iterator ob_iter = observations.begin();
			ob_iter != observations.end(); ++ob_iter, ++j)
		{
			observed_point = ob_iter->object_center_;
			dist_mat.at<int>(i, j) = getDistanceSquare(predicted_point, observed_point);
			if(dist_mat.at<int>(i, j) > kMaxCorrespondingDistance_)
			{
				dist_mat.at<int>(i, j) = INT_MAX;
			}
		}
	}
	//if(963 <= g_GlobalFrameCount)
	//{
	//	printf("\nFillDistanceMatrix: ");
	//	printf("\nTracker center: ");
	//	i = 0;
	//	for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
	//		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	//	{
	//		std::cout << " " << tracker_iter->bdbox_.object_center_ << " ";
	//	}
	//	printf("\nTracker predicted center: ");
	//	i = 0;
	//	for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
	//		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	//	{
	//		std::cout << " " << tracker_iter->getPredictedPoint() << " ";
	//	}
	//	printf("\nob center: ");
	//	for(std::vector<BoundingBox>::const_iterator ob_iter = observations.begin();
	//		ob_iter != observations.end(); ++ob_iter)
	//	{
	//		std::cout << " " << ob_iter->object_center_ << " ";
	//	}
	//}
}

void TrackerController::FillDecisionMatrix(cv::Mat& decision_mat, const cv::Mat& distance_mat) const
{
	int width = decision_mat.size().width;
	int height = decision_mat.size().height;
	int min_val;
	int min_idx;

	decision_mat.setTo(0);
	// ÿ������Сֵ
	for(int i = 0; i < height; ++i)
	{
		min_val = INT_MAX;
		for(int j = 0; j < width; ++j)
		{
			if(distance_mat.at<int>(i, j) < min_val)
			{
				min_val = distance_mat.at<int>(i, j);
				min_idx = j;
			}
		}
		if(min_val < INT_MAX)
		{
			++decision_mat.at<int>(i, min_idx);
		}
	}
	// ÿ������Сֵ
	for(int j = 0; j < width; ++j)
	{
		min_val = INT_MAX;
		for(int i = 0; i < height; ++i)
		{
			if(distance_mat.at<int>(i, j) < min_val)
			{
				min_val = distance_mat.at<int>(i, j);
				min_idx = i;
			}
		}
		if(min_val < INT_MAX)
		{
			++decision_mat.at<int>(min_idx, j);
		}
	}
}

// ����ÿ��ob��Ӧ��ֱ��ͼ
void TrackerController::GetObservationHistogram(const cv::Mat& img, const std::vector<BoundingBox>& observations, std::vector<cv::MatND>& ob_hists) const
{
	int ob_num = observations.size();
	//if(ob_num != ob_hists.size())
	//{
	//	printf(" ERROR "); // NOTICE
	//	exit(-1);
	//}
	//if(ob_num != ob_hists.size())	// ������β���Ҫ
	//{
	//	ob_hists.resize(ob_num);
	//}

	int channels[] = {0, 1, 2};
	int hist_size[] = {32, 32, 32};
	float rgb_range[] = {0, 256};
	const float *ranges[] = {rgb_range, rgb_range, rgb_range};
	for(int i = 0; i < ob_num; ++i)
	{
		cv::Mat object_region = img(cv::Rect(observations[i].top_left_, observations[i].bottom_right_));
		cv::calcHist(&object_region, 1, channels, cv::Mat(), ob_hists[i], 3, hist_size, ranges);
		cv::normalize(ob_hists[i], ob_hists[i]);
		//double norm2 = cv::norm(ob_hists[i], cv::NORM_L2);
		//std::cout << "norm = " << norm2 << std::endl;
		//cv::Scalar count = cv::sum(ob_hists[i]);
		//std::cout << "\n count = " << count;
		//std::cout << "\t object area = " << (observations[i].bottom_right_.x - observations[i].top_left_.x) * (observations[i].bottom_right_.y - observations[i].top_left_.y) << std::endl;
		//system("pause");
	}
}

double TrackerController::BhattacharyaDistance(const cv::Mat& hist1, const cv::Mat& hist2) const
{
	cv::Scalar sum = 0.0;
	cv::MatND dist_mat;
	cv::multiply(hist1, hist2, dist_mat);
	sum = cv::sum(dist_mat);
	sum[0] = sqrt(sum[0]);
	//if(sum[0] > 1.0)
	//{
	//	printf("!!!");
	//	system("pause");
	//}
	//cv::Scalar sum = 0.0;
	//cv::MatND dist_mat;
	//cv::multiply(hist1, hist2, dist_mat);
	//cv::sqrt(dist_mat, dist_mat);
	//sum = cv::sum(dist_mat);
	//printf(" norm of hist1, hist2 = (%g, %g), sum = %g ", cv::norm(hist1), cv::norm(hist2), sum[0]);
	return sum[0];
}

void TrackerController::MakeDistanceCorrespondence(std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, cv::Mat& distance_mat) const
{
	int width = distance_mat.size().width;
	int height = distance_mat.size().height;
	cv::Mat decision_mat(height, width, CV_32S);	// ����J
	bool new_correspondence_occuring = false;
	int i, j, k;
	//if(963 <= g_GlobalFrameCount)
	//{
	//	std::cout << std::endl << distance_mat << std::endl;
	//	system("pause");
	//}
	do
	{
		new_correspondence_occuring = false;
		FillDecisionMatrix(decision_mat, distance_mat);
		
		for(i = 0; i < height; ++i)
		{
			for(j = 0; j < width; ++j)
			{
				if(2 == decision_mat.at<int>(i, j))
				{
					corresponding_ob_idx[i] = j;
					is_ob_corresponded[j] = true;
					for(k = 0; k < width; ++k)
					{
						distance_mat.at<int>(i, k) = INT_MAX;
					}
					for(k = 0; k < height; ++k)
					{
						distance_mat.at<int>(k, j) = INT_MAX;
					}
					new_correspondence_occuring = true;
				}
			}
		}
		
	} while(new_correspondence_occuring);
}

void TrackerController::FilterCorrespondenceByArea(const std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations_, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded) const
{
	int ob_idx;
	int i = 0;
	for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	{
		ob_idx = corresponding_ob_idx[i];
		if(ob_idx >= 0)
		{
			double area_ratio = (double)tracker_iter->bdbox().object_area_ / (double)observations_[ob_idx].object_area_;
			if(area_ratio > 1.5 || area_ratio < 0.67)
			{
				is_ob_corresponded[ob_idx] = false;
				corresponding_ob_idx[i] = -1;
				printf(" filter tracker by area idx: %d ", std::distance(trackers.begin(), tracker_iter));
			}
		}
	}
}


void TrackerController::FilterCorrespondenceByHistogram(std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::list<Tracker>& trackers, const std::vector<cv::MatND>& ob_hists) const
{
	int ob_idx;
	int i = 0;
	for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	{
		ob_idx = corresponding_ob_idx[i];
		if(ob_idx >= 0 
			&& BhattacharyaDistance(tracker_iter->getHistogram(), ob_hists[ob_idx]) < BHATTACHARYA_DISTANCE_THRESHOLD)
		{
			is_ob_corresponded[ob_idx] = false;
			corresponding_ob_idx[i] = -1;
			printf(" filter tracker by hist idx: %d ", std::distance(trackers.begin(), tracker_iter));
		}
	}
}

void TrackerController::UpdateMatchedTracker(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, const std::vector<int>& corresponding_ob_idx, const std::vector<cv::MatND>& ob_hists)
{
	int ob_idx;
	int i = 0;
	for(std::list<Tracker>::iterator tracker_iter = trackers.begin();
		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	{
		ob_idx = corresponding_ob_idx[i];
		if(ob_idx >= 0 && ob_idx < INT_MAX)
		{
			tracker_iter->Update(observations[ob_idx], ob_hists[ob_idx]);
		}
	}
}

// ��û��ƥ�䵽observation��fake tracker��Ѱ������bdbox_���ص��ģ�δ��trackerƥ��ģ����������bdbox_С��observation����Ϊ����tracker���ѳ���observations
// ���޸Ĵ�����trackers, corresponding_ob_idx��ob_hists
// ���ѵ�tracker�ݲ��ᱻ�Ƴ����µ�tracker�ᱻ���ӵ�new_trackers�У����ı�ԭ����trackers����ΪҪ��֤trackers��corresponding_ob_idx��λ��һ���ԣ�
// corresponding_ob_idx��Ӧ���ֵ�ᱻ��ΪINT_MAX
// ���ѳ���ob��Ӧ��is_ob_corresponded�е���Ҳ�ᱻ��Ϊtrue
void TrackerController::SplittingHandling(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers)
{
	int ob_idx;
	int i = 0, j = 0, k = 0;
	int ob_num = observations.size();
	std::vector<int> splitted_ob_idxes;
	int splitted_ob_num = 0;
	for(std::list<Tracker>::iterator tracker_iter = trackers.begin();
		tracker_iter != trackers.end(); ++tracker_iter, ++i)
	{
		if(tracker_iter->hidden_trackers_.size() > 0 && -1 == (ob_idx = corresponding_ob_idx[i]))
		{
			printf(" a ");
			// Ѱ�Һ��ʵ�ob
			splitted_ob_idxes.clear();
			//splitted_obs.clear();
			for(j = 0; j < ob_num; ++j)
			{
				// û�б�ƥ�䡢�����trackerС������tracker���غ�
				if(!is_ob_corresponded[j]
					&& observations[j].object_area_		< tracker_iter->bdbox().object_area_
					&& observations[j].bottom_right_.x	> tracker_iter->bdbox().top_left_.x
					&& observations[j].top_left_.x		< tracker_iter->bdbox().bottom_right_.x
					&& observations[j].bottom_right_.y	> tracker_iter->bdbox().top_left_.y
					&& observations[j].top_left_.y		< tracker_iter->bdbox().bottom_right_.y)
				{
					splitted_ob_idxes.push_back(j);
					//splitted_obs.push_back(observations[j]);
					//corresponding_ob_idx[i] = INT_MAX;
					//is_ob_corresponded[j] = true;
					//new_trackers.push_back(Tracker(*tracker_iter, observations[j], ob_hists[j]));
				}
			}
			// Ѱ��tracker_iter->hidden_trackers_��splitted_ob_idxes��Ӧ��ƥ��
			if((splitted_ob_num = splitted_ob_idxes.size()) > 0)
			{
				for(std::list<Tracker>::iterator hidden_tracker_iter = tracker_iter->hidden_trackers_.begin();
					hidden_tracker_iter != tracker_iter->hidden_trackers_.end(); ++hidden_tracker_iter)
				{
					double bhattacharya_corr = 0.0;
					double max_corr = 0.0;
					int max_corr_idx = -1;
					for(k = 0; k < splitted_ob_num; ++k)
					{
						if(!is_ob_corresponded[splitted_ob_idxes[k]])
						{
							double area_ratio = (double)hidden_tracker_iter->bdbox().object_area_ / (double)observations[splitted_ob_idxes[k]].object_area_;
							if(area_ratio > 1.5 || area_ratio < 0.67)
							{
								continue;
							}
							bhattacharya_corr = BhattacharyaDistance(hidden_tracker_iter->hist_, ob_hists[splitted_ob_idxes[k]]);
							assert(bhattacharya_corr >= 0.0 && bhattacharya_corr <= 1.0);
							if(bhattacharya_corr > max_corr)
							{
								max_corr = bhattacharya_corr ;
								max_corr_idx = k;
							}
						}
					}
					//if(max_corr > 1.0)
					//{
					//	printf(" max_corr = %g ", max_corr);
					//}
					if(max_corr > BHATTACHARYA_DISTANCE_THRESHOLD)	// ƥ�䣬hidden_trackerҪ������
					{
						corresponding_ob_idx[i] = INT_MAX;
						is_ob_corresponded[splitted_ob_idxes[max_corr_idx]] = true;
						// ��Ϊhidden_tracker���һ�������trajectory_�ĵ�һ���㣬���Բ���ʱΪ�˱����ظ�����ʹ��tracker_iter->trajectory_.begin() + 1
						//hidden_tracker_iter->trajectory_.insert(hidden_tracker_iter->trajectory_.end(), tracker_iter->trajectory_.begin() + 1, tracker_iter->trajectory_.end());
						// ��fake tracker�Ĺ켣������hidden_tracker��
						for(std::vector<cv::Point>::const_iterator point_iter = tracker_iter->trajectory_.begin() + 1;
							point_iter != tracker_iter->trajectory_.end(); ++point_iter)
						{
							hidden_tracker_iter->HiddenUpdate(*point_iter);
						}
						hidden_tracker_iter->Update(observations[splitted_ob_idxes[max_corr_idx]], ob_hists[splitted_ob_idxes[max_corr_idx]]);
						// ������һ������Ҫ�ĵط������������Ѻ�tracker��predicted_center��object_center�����Զ���ܿ��ܵ��µ�һ֡���ѣ��ڶ�֡��tracker��ʧ�����
						// ����Ҫ����¼���
						while(getDistanceSquare(hidden_tracker_iter->getPredictedPoint(), hidden_tracker_iter->bdbox_.object_center_) >= kMaxCorrespondingDistance_ / 2)
						{
							hidden_tracker_iter->HiddenUpdate(observations[splitted_ob_idxes[max_corr_idx]].object_center_);
						}
						new_trackers.push_back(*hidden_tracker_iter);
						//printf(" push ");
					}
				}

			}
		}
	}

}

// ��û��tracker��ƥ���observation��Ѱ���������ص��ģ�δƥ��ob�ģ���������ȸ�obС��tracker����Ϊ���Ƕ��tracker�ϲ��Ľ��
// ���޸Ĵ�����trackers, corresponding_ob_idx��ob_hists
// �ϲ���һ����ԭtracker�ݲ��ᱻ�Ƴ����µ�tracker�ᱻ���ӵ�new_trackers�У����ı�ԭ����trackers����ΪҪ��֤trackers��corresponding_ob_idx��λ��һ���ԣ�
// corresponding_ob_idx��Ӧ���ֵ�ᱻ��ΪINT_MAX
// �ϲ����ob��Ӧ��is_ob_corresponded�е���Ҳ�ᱻ��Ϊtrue
void TrackerController::MergingHandling(std::list<Tracker>& trackers, const std::vector<BoundingBox>& observations, std::vector<int>& corresponding_ob_idx, std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers)
{
	int ob_num = observations.size();
	std::list<Tracker> origin_trackers;
	std::list<int> origin_tracker_idx;
	int j = 0;
	for(int i = 0; i < ob_num; ++i)
	{
		if(!is_ob_corresponded[i])	// ��û��trackerƥ���observation
		{
			j = 0;
			origin_trackers.clear();
			origin_tracker_idx.clear();
			// Ѱ��û��ƥ��ob�ģ��������obС�ģ�����ob�ص���tracker
			for(std::list<Tracker>::const_iterator tracker_iter = trackers.begin();
				tracker_iter != trackers.end(); ++tracker_iter, ++j)
			{
				if(-1 == corresponding_ob_idx[j]
				&& tracker_iter->bdbox().object_area_	 < observations[i].object_area_
				&& tracker_iter->bdbox().bottom_right_.x > observations[i].top_left_.x
				&& tracker_iter->bdbox().top_left_.x	 < observations[i].bottom_right_.x
				&& tracker_iter->bdbox().bottom_right_.y > observations[i].top_left_.y
				&& tracker_iter->bdbox().top_left_.y	 < observations[i].bottom_right_.y
				)
				{
					origin_trackers.push_back(*tracker_iter);
					origin_tracker_idx.push_back(j);
					//corresponding_ob_idx[j] = INT_MAX;
				}
			}
			if(2 == origin_trackers.size() || 3 == origin_trackers.size())		// ֻ����2��3���ϲ���һ��
			{
				//printf(" merging ");
				is_ob_corresponded[i] = true;
				new_trackers.push_back(Tracker(origin_trackers, observations[i], ob_hists[i], kColorPalette_[rng.uniform(0, kPaletteSize_)]));

				for(std::list<int>::const_iterator tracker_idx_iter = origin_tracker_idx.begin();
					tracker_idx_iter != origin_tracker_idx.end(); ++tracker_idx_iter)
				{
					corresponding_ob_idx[*tracker_idx_iter] = INT_MAX;
				}
			}
		}
	}
}

void TrackerController::CreateNewTrackerForUnmatchedObservation(const std::vector<BoundingBox>& observations, const std::vector<bool>& is_ob_corresponded, const std::vector<cv::MatND>& ob_hists, std::list<Tracker>& new_trackers)
{
	
	int ob_num = observations.size();
	for(int i = 0; i < ob_num; ++i)
	{
		if(!is_ob_corresponded[i])	// ��û��trackerƥ���observation
		{
			new_trackers.push_back(Tracker(observations[i], ob_hists[i], kColorPalette_[rng.uniform(0, kPaletteSize_)]));
		}
	}
}

void TrackerController::UpdateUnmatchedTracker(const std::vector<int>& corresponding_ob_idx)
{
	int i = 0;
	for(std::list<Tracker>::iterator tracker_iter = trackers_.begin();
		tracker_iter != trackers_.end(); ++tracker_iter, ++i)
	{
		if(-1 == corresponding_ob_idx[i])
		{
			tracker_iter->Update();
		}
	}
}

void TrackerController::FilterSplittedMergedTracker(const std::vector<int>& corresponding_ob_idx)
{
	int i = 0;
	for(std::list<Tracker>::iterator tracker_iter = trackers_.begin();
		tracker_iter != trackers_.end(); ++i)
	{
		if(INT_MAX == corresponding_ob_idx[i])
		{
			tracker_iter = trackers_.erase(tracker_iter);
		}
		else
		{
			++tracker_iter;
		}
	}
}


void TrackerController::FilterLostTracker()
{
	for(std::list<Tracker>::iterator tracker_iter = trackers_.begin();
		tracker_iter != trackers_.end(); )
	{
		if(tracker_iter->getMissingCount() > kMaxMissingCount_)
		{
			tracker_iter = trackers_.erase(tracker_iter);
		}
		else
		{
			++tracker_iter;
		}
	}
}

int TrackerController::getDistanceSquare(cv::Point p1, cv::Point p2) const
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}