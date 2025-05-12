#include "detector_pipeline.h"

// Inferencer detector_model_init(std::string detector_model_path){
//     return Inferencer(detector_model_path);
// }

std::vector<RotatedObj> detector_infer(Inferencer *detector, cv::Mat img){
    detector->GetInputInfo();
	detector->GetOutputInfo();
    detector->PreProcess(img);
    detector->Inference();
    detector->PostProcess();
    return detector->Get_remain_rotated_objects();
}

