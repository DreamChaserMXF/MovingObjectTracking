#ifndef GLOBAL_VARIABLE_H
#define GLOBAL_VARIABLE_H

#include <stdio.h>
#include <math.h>
#include <cv.h>

#define FIXED_WIDTH_HEIGHT	// ���ڽ�����Ƶ�л�ȡ��ͼ��ת���ɹ̶���С��ͼ��
//#define BACKGROUND_SHOW		// ��־�Ƿ�չʾ����ģ��
//#define EIGHT_CONNECTED		// ����ͨ����ͨ����
#define SPLITTING_MERGING		// �Ƿ����˶�Ŀ��ķ�����ϲ�
//#define SHADOW_DETECTION

// ������
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