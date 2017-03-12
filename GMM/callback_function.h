#ifndef CALLBACK_FUNCTION
#define CALLBACK_FUNCTION

#include <opencv.hpp>
#include "global_variable.h"

class MouseClickEvent
{
public:
	MouseClickEvent();
	void reset();
	bool is_clicked_;
	cv::Point clicked_point_;
};


void on_mouse(int event, int x, int y, int flags, void* param);
void on_trackbar(int pos, void* userdata);









#endif