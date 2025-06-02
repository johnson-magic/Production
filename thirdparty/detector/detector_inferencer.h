#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "data_structure.h"


class DetectorInferencer {
    public:
        DetectorInferencer(std::string& model_path, const int& class_num){    

            modelScoreThreshold_ = 0.25;
            modelNMSThreshold_ = 0.7;
            class_num_ = class_num;

            model_path_ = model_path;
            Init(model_path_);
        };

        
        void GetInputInfo();
        void GetOutputInfo();


        void PreProcess(std::string& image_path);
        void PreProcess(cv::Mat image);
        void Inference();
        void PostProcess();

        void Release();

        std::vector<RotatedObj> Get_remain_rotated_objects();

    private:
        Ort::Session* session_;
        Ort::SessionOptions options_;
        Ort::Env env_{nullptr};

        std::string model_path_;
        std::string image_path_;
        cv::Mat image_;


        size_t numInputNodes_;  // now, only support 1
        std::vector<std::string> input_node_names_;
        size_t numOutputNodes_;
        std::vector<std::string> output_node_names_;
        std::vector<Ort::Value> ort_outputs_;
        
        std::vector<int> net_w_;  // net input (width)
        std::vector<int> net_h_;  // net input (height)
        std::vector<int> output_w_;  // net output (width)
        std::vector<int> output_h_;  // net output (height)

        float x_factor_;
        float y_factor_;
        float scale_;
        int top_;  // border
        int bottom_;
        int left_;
        int right_;

        std::vector<RotatedObj> rotated_objects_;  // before nms
        std::vector<RotatedObj> remain_rotated_objects_;  // after nms

        double modelScoreThreshold_;
        double modelNMSThreshold_;
        int class_num_;
    
    private:
        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();

        cv::Mat formatToSquare(cv::Mat image);
        void PrepareForNms(const cv::Mat & det_output, const int & i, cv::Point classIdPoint, const double & score, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);

        void Nms(std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);
        void Init(std::string model_path);
        static std::basic_string<ORTCHAR_T> ConvertToWString(std::string& model_path){ 
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }
};
