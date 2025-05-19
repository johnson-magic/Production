#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class ClassifierInferencer {
    public:  // 被外部调用的函数放这里，否则放private
        ClassifierInferencer(std::string& model_path){    
            model_path_ = model_path;
            Init(model_path_);
        };

        void GetInputInfo();
        void GetOutputInfo();
        void PreProcess(const std::string& image_path);
        void PreProcess(cv::Mat image);
        void Inference();
        void PostProcess();
        std::pair<std::vector<int>, std::vector<float>> ClassifierInferencer::GetRes();
        void Release();
        
    private:  // 跨函数声明周期的放这里，否则用临时变量即可
        Ort::Session* session_;
        Ort::SessionOptions options_;
        Ort::Env env_{nullptr};

        std::string model_path_;
        std::string image_path_;
        cv::Mat image_;
        
        size_t numInputNodes_;  // 通常，输入节点和输出节点的数量均为1
        std::vector<std::string> input_node_names_;
        size_t numOutputNodes_;
        std::vector<std::string> output_node_names_;
        std::vector<Ort::Value> ort_outputs_;
        
        std::vector<int> net_w_;  // 事实上，通常仅仅只有1个输入node和1个输出node, 这里用vector而不是直接定义为int变量，仅仅是为了接口通用
        std::vector<int> net_h_;
        std::vector<int> class_num_;

        std::vector<int> predictions_;
        std::vector<float> scores_;
    
    private:  // 私有函数
        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();

        cv::Mat pad_and_resize(const cv::Mat &image);    
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);

        void Init(const std::string &model_path);

        static std::basic_string<ORTCHAR_T> ConvertToWString(const std::string& model_path){            
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }
};
