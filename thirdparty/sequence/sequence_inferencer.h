
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>



class SequenceInferencer {
    public:
        SequenceInferencer(const std::string& model_path, const std::string& charset_path){   
            model_path_ = model_path;
            charset_path_ = charset_path;
            Init(model_path_, charset_path_);
        };

        void GetInputInfo();
        void GetOutputInfo();


        void PreProcess(const std::string& image_path);
        void PreProcess(cv::Mat image);
        void Inference();
        void PostProcess();
        std::pair<std::vector<int>, std::vector<char>> GetRes();

        void Release();
        

    private:
        Ort::Session* session_;
        Ort::SessionOptions options_;
        Ort::Env env_{nullptr};

        std::string model_path_;
        std::string image_path_;
        std::string charset_path_;
	    std::vector<char> charset_;  // charset of sequence model
        int charset_len_;
        int charset_len_with_blank_;  // "--hh-e-l-ll-oo--", "-" represents blank
	    cv::Mat image_;

        // Ort::Value input_tensor_;
        std::vector<Ort::Value> ort_outputs_;
        
        size_t numInputNodes_;  // usually, it is 1
        std::vector<std::string> input_node_names_;
        size_t numOutputNodes_;
	    std::vector<std::string> output_node_names_;
        
        std::vector<int> 
        
        net_w_;  // net input (width)  # 事实上，通常仅仅只有1个输入node和1个输出node, 这里仅仅是为了接口通用
        std::vector<int> net_h_;  // net input (height)
        std::vector<int> class_num_;  // net output(class_num_)

        std::vector<int> decoded_indices_;
        std::vector<char> decoded_chars_;

    private:
        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();
 
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);

        void Init(std::string &model_path, std::string &charset_path);
        static std::basic_string<ORTCHAR_T> ConvertToWString(std::string& model_path){
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }
};
