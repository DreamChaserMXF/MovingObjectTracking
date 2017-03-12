# MovingObjectTracking
Track moving object through static camera
对指定视频中的运动目标进行跟踪。
视频必须来源于静止摄像机的拍摄。
对运动目标用高斯混合模型进行检测；连通域分析进行去噪；最大后验概率进行相邻帧目标对应，并处理目标的出现、消失、分裂与合并；用卡尔曼滤波进行轨迹的最优估计。
