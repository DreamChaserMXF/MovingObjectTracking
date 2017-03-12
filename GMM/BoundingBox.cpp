#include "BoundingBox.h"

void BoundingBox::DrawingBoundingBoxes(cv::Mat& output_img, const std::vector<BoundingBox> &bdboxes, cv::Scalar color, int thickness, int line_type)
{
	for(std::vector<BoundingBox>::const_iterator iter = bdboxes.begin();
		iter != bdboxes.end(); ++iter)
	{
		cv::rectangle(output_img, iter->top_left_, iter->bottom_right_, color, thickness, line_type);
	}
}

bool BoundingBox::GetSelectedBoundingBox(const std::vector<BoundingBox>& boundingboxes, const cv::Point& selected_point, BoundingBox& selected_boundingbox)
{
	for(std::vector<BoundingBox>::const_iterator c_iter = boundingboxes.begin();
		c_iter != boundingboxes.end(); ++c_iter)
	{
		if(    c_iter->top_left_.x < selected_point.x
			&& c_iter->top_left_.y < selected_point.y
			&& c_iter->bottom_right_.x > selected_point.x
			&& c_iter->bottom_right_.y > selected_point.y)
		{
			selected_boundingbox = *c_iter;
			return true;
		}
	}
	return false;
}

bool BoundingBox::GetClosestBoundingBox(const std::vector<BoundingBox>& boundingboxes, const BoundingBox& original_boundingbox, BoundingBox& closest_boundingbox, int& min_distance)
{
	int distance = 0;
	min_distance = INT_MAX;
	std::vector<BoundingBox>::const_iterator closest_citer = boundingboxes.end();
	for(std::vector<BoundingBox>::const_iterator c_iter = boundingboxes.begin();
		c_iter != boundingboxes.end(); ++c_iter)
	{
		distance = (original_boundingbox.object_center_.x - c_iter->object_center_.x) * (original_boundingbox.object_center_.x - c_iter->object_center_.x)
				 + (original_boundingbox.object_center_.y - c_iter->object_center_.y) * (original_boundingbox.object_center_.y - c_iter->object_center_.y);
		if(distance < min_distance)
		{
			min_distance = distance;
			closest_citer = c_iter;
		}
	}
	if(closest_citer != boundingboxes.end())
	{
		closest_boundingbox = *closest_citer;
		return true;
	}
	else
	{
		return false;
	}
}