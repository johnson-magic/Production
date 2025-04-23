#pragma once
#include <opencv2/opencv.hpp>

// define a struct to save some information
typedef struct {
	cv::RotatedRect rotated_rect;
	float score;
	int class_index;
}RotatedObj;
