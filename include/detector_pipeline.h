#include "detector_inference.h"


// Inferencer detector_model_init(const std::string &detector_model_path);
std::vector<RotatedObj> detector_infer(DetectorInferencer *detector, const cv::Mat img);

