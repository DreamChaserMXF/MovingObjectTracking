#include "ImageSubplot.h"


void ImageSubplot::subplot(cv::Mat &figure, const cv::Mat &img, int row, int col, int show_idx)
{
	--show_idx;
	assert(show_idx >= 0);
	assert(show_idx < row * col);
	const int SPACE_WIDTH = 2;
	int subplot_height = (figure.rows - (row - 1) * SPACE_WIDTH) / row;
	int subplot_width = (figure.cols - (col - 1) * SPACE_WIDTH) / col;
	int row_idx = show_idx / col;
	int col_idx = show_idx % col;
	
	cv::Rect roi((subplot_width + SPACE_WIDTH) * col_idx, (subplot_height + SPACE_WIDTH) * row_idx, subplot_width, subplot_height);
	cv::Mat subplot_figure = figure(roi);
	
	cv::resize(img, subplot_figure, cv::Size(subplot_width, subplot_height));
}

cv::Point ImageSubplot::RecoverPoint(cv::Size figure_size, cv::Size img_size, int row, int col, int show_idx, cv::Point point)
{
	--show_idx;
	assert(show_idx >= 0);
	assert(show_idx < row * col);
	const int SPACE_WIDTH = 2;
	int subplot_height = (figure_size.height - (row - 1) * SPACE_WIDTH) / row;
	int subplot_width = (figure_size.width - (col - 1) * SPACE_WIDTH) / col;
	int row_idx = show_idx / col;
	int col_idx = show_idx % col;

	// 先求出相对偏移
	cv::Point ret_point(point.x - (subplot_width + SPACE_WIDTH) * col_idx, point.y - (subplot_height + SPACE_WIDTH) * row_idx);
	// 再逆缩放回去
	ret_point.x = (int)((double)ret_point.x / (double)subplot_width * img_size.width);
	ret_point.y = (int)((double)ret_point.y / (double)subplot_height * img_size.height);
	return ret_point;
}