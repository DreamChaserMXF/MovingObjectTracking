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
	int x, y;		// ͼ��ɨ��ʱ���±�
	int cc_length;	// ��ͨ������ظ���
	//cc_visited_mask_.assign(input_img.size().area(), false);	// ��ʼ����������
	memset(cc_visited_mask_, 0, width_ * height_);
	// ��ʼ��ͨ�����
	for(y = 0; y < height_; ++y)		// y��ʾ��, ������
	{
		for(x = 0; x < width_; ++x)	// x��ʾ��, ������
		{
			if(false == cc_visited_mask_[y * width_ + x])
			{
				// ��⵽�µ���ͨ��
				// �����ص����
				color = input_img.at<uchar>(y, x);
				cc_point_buffer_[0].y = y;
				cc_point_buffer_[0].x = x;
				cc_visited_mask_[y * width_ + x] = true;
				// ��ʼ��ͨ������
				cc_length = CCGrowing(input_img, 1, color);	// cc_length��Ϊ��ͨ������
				// �˲�������ͨ�������С���ͽ���Ĩ��
				if(FOREGROUND == color && cc_length < foreground_threshold
					|| BACKGROUND == color && cc_length < background_threashold)
				{
					color = color == BACKGROUND ? FOREGROUND : BACKGROUND;
					for(int i = 0; i < cc_length; ++i)
					{
						output_img.at<uchar>(cc_point_buffer_[i].y, cc_point_buffer_[i].x) = color;	// �ı���ɫ
					}
					// ��һ����ͨ���ǻ��εģ���ô�����ı���ɫ�󣬻���ڲ��Ķ����ⲿ�ĵ���������
					// ����ڲ��Ķ������Ҳ��С�����ڲ�����ͨ��Ҳ��ı���ɫ����Ϊ�µ�
					// ����Ҫ��ɨ��һ�飬�Ѹ���ͨ�����ڲ��Ķ���������
					// ���ڴ�ʱ�ڲ��Ķ���������������Ȧ���������ٸı���ɫ�����Ըú�������Ҫ�ݹ����
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
	int x, y;			// ͼ��ɨ��ʱ���±�
	int cc_x, cc_y;		// ��ͨ�����������
	int accu_x, accu_y;	// ����õ�x��y����
	cv::Point top_left, bottom_right;
	BoundingBox box;
	int cc_length;		// ��ͨ������ظ���
	//cc_visited_mask_.assign(width_ * height_, false);
	memset(cc_visited_mask_, 0, width_ * height_);
	bdboxes.clear();
	for(y = 0; y < height_; ++y)		// y��ʾ��, ������
	{
		for(x = 0; x < width_; ++x)	// x��ʾ��, ������
		{
			if(false == cc_visited_mask_[y * width_ + x])	// ��⵽�µ���ͨ��
			{
				// ��һ�����ص����
				color = input_img.at<uchar>(y, x);
				cc_point_buffer_[0].y = y;
				cc_point_buffer_[0].x = x;
				cc_visited_mask_[y * width_ + x] = true;
				if(FOREGROUND == color)	
				{
					top_left = cc_point_buffer_[0];
					bottom_right = cc_point_buffer_[0];
				}
				// ��ʼ��ͨ������
				cc_length = CCGrowing(input_img, 1, color);
				// ��ǰ����ͨ��ľ��α߽��Լ�Ŀ�����ļ�¼����
				if(FOREGROUND == color) 
				{
					accu_x = 0;
					accu_y = 0;
					for(int i = 0; i < cc_length; ++i)
					{
						cc_x = cc_point_buffer_[i].x;
						cc_y = cc_point_buffer_[i].y;
						// ��¼�߽�
						if(cc_y < top_left.y)		// ���������ƫ��
						{
							top_left.y = cc_y;
						}
						if(cc_y > bottom_right.y)	// ���������ƫ��
						{
							bottom_right.y = cc_y ;
						}
						if(cc_x < top_left.x)		// ���������ƫ��
						{
							top_left.x = cc_x;
						}
						if(cc_x > bottom_right.x)	// ���������ƫ��
						{
							bottom_right.x = cc_x;
						}
						// ��¼����
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
	// headָ�����Ԫ�أ�rearָ���βԪ�ص���һ��λ��
	int head = 0;
	int rear = queue_length;
	int cc_x, cc_y;
	while(head != rear)	// ��ͨ����δ�������
	{
		//����
		cc_y = cc_point_buffer_[head].y;
		cc_x = cc_point_buffer_[head].x;
		++head;
		//����ͨ���
		if(cc_y > 0 
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x] 
			&& color == input_img.at<uchar>(cc_y-1, cc_x))	//��
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x;
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x] = true;
		}
		if(cc_y < height_ - 1 
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x] 
			&& color == input_img.at<uchar>(cc_y+1, cc_x))	//��
		{
			cc_point_buffer_[rear].y = cc_y + 1;
			cc_point_buffer_[rear].x = cc_x;
			++rear;
			cc_visited_mask_[(cc_y+1) * width_ + cc_x] = true;						
		}
		if(cc_x > 0
			&& false == cc_visited_mask_[cc_y * width_ + cc_x - 1]
			&& color == input_img.at<uchar>(cc_y, cc_x - 1))	//��
		{
			cc_point_buffer_[rear].y = cc_y;
			cc_point_buffer_[rear].x = cc_x - 1;	
			++rear;
			cc_visited_mask_[cc_y * width_ + cc_x - 1] = true;		
		}
		if(cc_x < width_ - 1
			&& false == cc_visited_mask_[cc_y * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y, cc_x + 1))	//��
		{
			cc_point_buffer_[rear].y = cc_y;
			cc_point_buffer_[rear].x = cc_x + 1;	
			++rear;
			cc_visited_mask_[cc_y * width_ + cc_x + 1] = true;						
		}
#ifdef EIGHT_CONNECTED
		// ����ͨ(����б��ͨ)���
		if(cc_x > 0 && cc_y > 0
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x - 1] 
			&& color == input_img.at<uchar>(cc_y-1, cc_x - 1))	//���� y-1, x-1
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x - 1;
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x - 1] = true;						
		}
		if(cc_x > 0 && cc_y < height_ - 1 
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x - 1] 
			&& color == input_img.at<uchar>(cc_y+1, cc_x - 1))	//���� y+1, x-1
		{
			cc_point_buffer_[rear].y = cc_y + 1;
			cc_point_buffer_[rear].x = cc_x - 1;
			++rear;
			cc_visited_mask_[(cc_y+1) * width_ + cc_x - 1] = true;
		}
		if(cc_x < width_ - 1 && cc_y > 0
			&& false == cc_visited_mask_[(cc_y-1) * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y-1, cc_x + 1))	//����  y-1, x+1
		{
			cc_point_buffer_[rear].y = cc_y - 1;
			cc_point_buffer_[rear].x = cc_x + 1;	
			++rear;
			cc_visited_mask_[(cc_y-1) * width_ + cc_x + 1] = true;
		}
		if(cc_x < width_ - 1 && cc_y < height_ - 1
			&& false == cc_visited_mask_[(cc_y+1) * width_ + cc_x + 1]
			&& color == input_img.at<uchar>(cc_y, cc_x + 1))	//���� y+1, x+1
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
