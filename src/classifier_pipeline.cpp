#include "classifier_pipeline.h"

// ClassifierInferencer classifier_model_init(std::string classifier_model_path){
//     return ClassifierInferencer(classifier_model_path);
// }

int classifier_pipeline(ClassifierInferencer* classifier, cv::Mat image){
    classifier->GetInputInfo();
    classifier->GetOutputInfo();
    classifier->PreProcess(image);
    classifier->Inference();
    classifier->PostProcess();
    std::pair<std::vector<int>, std::vector<float>> classifier_res = classifier->GetRes();
    std::vector<int> classifier_classes = classifier_res.first;
    std::vector<float> classifier_scores = classifier_res.second;

    int cls = classifier_classes[0];
    return cls;
}