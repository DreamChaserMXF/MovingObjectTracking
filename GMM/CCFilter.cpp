#include "global_variable.h"
#include "CCFilter.h"



CCFilter::CCFilter(int width, int height) : width_(width), height_(height), cc_point_buffer_((cv::Point*)calloc(width * height, sizeof(cv::Point))), cc_visited_mask_((bool*)calloc(width * height, sizeof(bool)))
{ }

CCFilter::~CCFilter()
{
	free(cc_point_buffer_);
	free(cc_visited_mask_);
}

void CCFilter::filter(const cv::Mat& input_img, cv::Mat& output_img, int background_threashold, int foreground_threshold)
{
	assert(input_img.size() == cv::Size(width_, height_));
	if(output_img.data != input_img.data)
	{
		input_img.copyTo(output_img);
	}

	uchar color;
	int x, y;		// 图像扫描时的下标
	int cc_length;	// 连通域的像素个数
	//cc_visited_mask_.assign(input_img.size().area(), false);	// 初始化访问数组
	memset(cc_visited_mask_, 0, width_ * height_);
	// 开始连通域分析
	for(y = 0; y < height_; ++y)		// y表示高, 即行数
	{
		for(x = 0; x < width_; ++x)	// x表示宽, 即列数
		{
			if(false == cc_visited_mask_[y * width_ + x])
			{
				// 检测到新的连通域
				// 该像素点入队
				color = input_img.at<uchar>(y, x);
				cc_point_buffer_[0].y = y;
				cc_point_buffer_[0].x = x;
				cc_visited_mask_[y * width_ + x] = true;
				// 开始连通域增长
				cc_length = CCGrowing(input_img, 1, color);	// cc_length即为连通域的面积
				// 滤波：若连通域面积过小，就将其抹掉
				if(FOREGROUND == color && cc_length < foreground_threshold
					|| BACKGROUND == color && cc_length < background_threashold)
				{
					color = color == BACKGROUND ? FOREGROUND : BACKGROUND;
					for(int i = 0; i < cc_length; ++i)
					{
						output_img.at<uchar>(cc_point_buffer_[i].y, cc_point_buffer_[i].x) = color;	// 改变颜色
					}
					// 将一个连通域是环形的，那么在它改变颜色后，会把内部的洞和外部的点连接起来
					// 如果内部的洞的面积也很小，则内部的连通域也会改变颜色，成为孤点
					// 所以要再扫描一遍，把该连通域与内部的洞连接起来
					// 由于此时内部的洞被连到了外面这圈，不可能再改变颜色，所以该函数不需要递归调用
					CCGrowing(input_img, cc_length, color);
				}	
			} // if(false == cc_visited_mask_[y * g_ImageWidth + x])
		} // x: 0->width_
	} // y: 0->height_
}

void CCFilter::detect(const cv::Mat& input_img, std::vector<BoundingBox>& bdboxes)
{
	assert(input_img.size() == cv::Size(width_, height_));
	uchar color;
	int x, y;			// 图像扫描时的下标
	int cc_x, cc_y;		// 连通域的像素坐标
	int accu_x, accu_y;	// 求和用的x和y坐标
	cv::Point top_left, bottom_right;
	BoundingBox box;
	int cc_length;		// 连通域的像素个数
	//cc_visited_mask_.assign(width_ * height_, false);
	memset(cc_visited_mask_, 0, width_ * height_);
	bdboxes.clear();
	for(y = 0; y < height_; ++y)		// y表示高, 即行数
	{
		for(x = 0; x < width_; ++x)	// x表示宽, 即列数
		{
			if(false == cc_visited_mask_[y * width_ + x])	// 检测到新的连通域
			{
				// 第一个像素点入队
				color = input_img.at<uchar>(y, x);
				cc_point_buffer_[0].y = y;
				cc_point_buffer_[0].x = x;
				cc_visited_mask_[y * width_ + x] = true;
				if(FOREGROUND == color)	
				{
					top_left = cc_point_buffer_[0];
					bottom_right = cc_point_buffer_[0];
				}
				// 开始连通域增长
				cc_length = CCGrowing(input_img, 1, color);
				// 把前景连通域的矩形边界以及目标中心记录下来
				if(FOREGROUND == color) 
				{
					accu_x = 0;
					accu_y = 0;
					for(int i = 0; i < cc_length; ++i)
					{
						cc_x = cc_point_buffer_[i].x;
						cc_y = cc_point_buffer_[i].y;
						// 记录边界
						if(cc_y < top_left.y)		// 若纵坐标更偏上
						{
							top_left.y = cc_y;
						}
						if(cc_y > bottom_right.y)	// 若纵坐标更偏下
						{
							bottom_right.y = cc_y ;
						}
						if(cc_x < top_left.x)		// 若横坐标更偏左
						{
							top_left.x = cc_x;
						}
						if(cc_x > bottom_right.x)	// 若横坐标更偏右
						{
							bottom_right.x = cc_x;
						}
						// 记录坐标
						accu_x += cc_x;
						accu_y += cc_y;
					}
					box.top_left_ = top_left;
					box.bottom_right_ = bottom_right;
					box.object_center_.x = accu_x / cc_length;
					box.object_center_.y = accu_y / cc_length;
					box.object_area_ = cc_length;
					bdboxes.push_back(box);
				}
			} // if(false == cc_visited_mask_[y * g_ImageWidth + x])
		} // x: 0->width_
	} // y: 0->height_
}


int CCFilter::CCGrowing(const cv::Mat& input_img, int queue_length, uchar color)
{
	// head指向队首元素，rear指向队尾元素的下一个位置
	int head = 0;
	int rear = queue_length;
	int cc_x, cc_y;
	while(head != rear)	// 连通域尚未增长完毕
	{
		//出队
		cc_y = cc_point_buffer_[head].y;
		cc_x = cc_point_buffer_[head].x;
		++head;
		//四连通入队
		if(cc_y > 0 
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x] 
			&& color == input_img.at<uchar>(cc_y-1, cc_x))	//上
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x;
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x] = true;
		}
		if(cc_y < height_ - 1 
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x] 
			&& color == input_img.at<uchar>(cc_y+1, cc_x))	//下
		{
			cc_point_buffer_[rear].y = cc_y + 1;
			cc_point_buffer_[rear].x = cc_x;
			++rear;
			cc_visited_mask_[(cc_y+1) * width_ + cc_x] = true;						
		}
		if(cc_x > 0
			&& false == cc_visited_mask_[cc_y * width_ + cc_x - 1]
			&& color == input_img.at<uchar>(cc_y, cc_x - 1))	//左
		{
			cc_point_buffer_[rear].y = cc_y;
			cc_point_buffer_[rear].x = cc_x - 1;	
			++rear;
			cc_visited_mask_[cc_y * width_ + cc_x - 1] = true;		
		}
		if(cc_x < width_ - 1
			&& false == cc_visited_mask_[cc_y * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y, cc_x + 1))	//右
		{
			cc_point_buffer_[rear].y = cc_y;
			cc_point_buffer_[rear].x = cc_x + 1;	
			++rear;
			cc_visited_mask_[cc_y * width_ + cc_x + 1] = true;						
		}
#ifdef EIGHT_CONNECTED
		// 八连通(后四斜连通)入队
		if(cc_x > 0 && cc_y > 0
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x - 1] 
			&& color == input_img.at<uchar>(cc_y-1, cc_x - 1))	//左上 y-1, x-1
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x - 1;
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x - 1] = true;						
		}
		if(cc_x > 0 && cc_y < height_ - 1 
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x - 1] 
			&& color == input_img.at<uchar>(cc_y+1, cc_x - 1))	//左下 y+1, x-1
		{
			cc_point_buffer_[rear].y = cc_y + 1;
			cc_point_buffer_[rear].x = cc_x - 1;
			++rear;
			cc_visited_mask_[(cc_y+1) * width_ + cc_x - 1] = true;
		}
		if(cc_x < width_ - 1 && cc_y > 0
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y-1, cc_x + 1))	//右上  y-1, x+1
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x + 1;	
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x + 1] = true;
		}
		if(cc_x < width_ - 1 && cc_y < height_ - 1
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y, cc_x + 1))	//右下 y+1, x+1
		{
			cc_point_buffer_[rear].y = cc_y + 1;
			cc_point_buffer_[rear].x = cc_x + 1;	
			++rear;
			cc_visited_mask_[(cc_y+1) * width_ + cc_x + 1] = true;
		}
#endif
	} // while(head != rear)
	return rear;
}
