#include "sequence_pipeline.h"

// SequenceInferencer sequence_model_init(std::string sequence_model_path, std::string charset_path){
//     return SequenceInferencer(sequence_model_path, charset_path);
// }

std::string sequence_pipeline(SequenceInferencer *sequence, cv::Mat image){
    sequence->GetInputInfo();
    sequence->GetOutputInfo();
    sequence->PreProcess(image);
    sequence->Inference();
    sequence->PostProcess();
    std::pair<std::vector<int>, std::vector<char>> sequence_res = sequence->GetRes();
    std::vector<int> sequence_ids = sequence_res.first;
    std::vector<char> sequence_chars = sequence_res.second;

    return std::string(sequence_chars.begin(), sequence_chars.end());
}
