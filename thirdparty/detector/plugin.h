#pragma once
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "data_structure.h"
#include "utils.h"



class PrecisionAngleDetection{

public:
    PrecisionAngleDetection(std::string image_path, std::string result_path, std::string vis_path): image_path_(image_path), result_path_(result_path), vis_path_(vis_path){
        Clear();





    };

private:
    std::string image_path_;
    std::string result_path_;
    std::string vis_path_;

    cv::Point2f center_point_;
	float diameter_;
	float angle_;
	float slider_angle_;
	std::string position_;
    cv::Point2f slider_center_point_;

public:
    void Process(const std::vector<RotatedObj> & rotated_objs);
    void SaveRes();
    void VisRes(const std::vector<RotatedObj> & rotated_objs);
    void Clear();
};
