#include "detector/inferencer.h"


// Inferencer detector_model_init(const std::string &detector_model_path);
std::vector<RotatedObj> detector_infer(Inferencer *detector, const cv::Mat img);

