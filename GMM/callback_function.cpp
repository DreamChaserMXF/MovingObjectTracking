#include "callback_function.h"


MouseClickEvent::MouseClickEvent() : is_clicked_(false), clicked_point_()
{ }

void MouseClickEvent::reset()
{
	is_clicked_ = false;
}

//void on_mouse(int event, int x, int y, int flags, void* param)
//{
//	if(CV_EVENT_LBUTTONDOWN == event)
//	{
//		MouseClickEvent *p_click_event = (MouseClickEvent*)param;
//		p_click_event->is_clicked_ = true;
//		p_click_event->clicked_point_.x = x;
//		p_click_event->clicked_point_.y = y;
//	}
//}
void on_trackbar(int pos, void* userdata)
{
	//printf("\npos: %d\n", pos);
	bool * p_trackbar_changed = reinterpret_cast<bool*>(userdata);
	*p_trackbar_changed = true;
}