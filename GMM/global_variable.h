#ifndef GLOBAL_VARIABLE_H
#define GLOBAL_VARIABLE_H

#include <stdio.h>
#include <math.h>
#include <cv.h>

#define FIXED_WIDTH_HEIGHT	// 用于将从视频中获取的图像转换成固定大小的图像
//#define BACKGROUND_SHOW		// 标志是否展示背景模型
//#define EIGHT_CONNECTED		// 八连通的连通域标记
#define SPLITTING_MERGING		// 是否处理运动目标的分裂与合并
//#define SHADOW_DETECTION

// 调试用
//extern int g_GlobalFrameCount;

const CvScalar COLOR_WHITE	= CV_RGB(255, 255, 255);
const CvScalar COLOR_BLACK	= CV_RGB(0, 0, 0);
const CvScalar COLOR_GRAY	= CV_RGB(100, 100, 100);
const CvScalar COLOR_RED	= CV_RGB(255, 0, 0);
const CvScalar COLOR_GREEN	= CV_RGB(0, 255, 0);
const CvScalar COLOR_BLUE	= CV_RGB(0, 0, 255);
const CvScalar COLOR_YELLOW	= CV_RGB(255, 255, 0);
const uchar COLOR_SHADOW = 127;
#endif