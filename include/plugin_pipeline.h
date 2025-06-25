#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "data_structure.h"


// static std::vector<std::string> target_texts = {"02", "05", "07", "10", "12", "15", "17", "20", "22", "25", "27", "30", "32", "35", "37", 
//     "025", "25", "050", "50", "075", "75", "100", "125", "150", "175", "200", "225", "250", "275", "300", "325", "350", "375"};
void plugin_pipeline(const std::vector<RotatedObj> &detector_res, const std::vector<int> &direction_res, const std::vector<std::string> &text_res,
    const std::string &save_path, cv::Mat img, const std::string &vis_path, const int img_w, const int img_h);
