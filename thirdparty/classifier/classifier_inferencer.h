#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "config.h"
#include "utils.h"
#include "data_structure.h"



class ClassifierInferencer {
    public:
        ClassifierInferencer(std::string& model_path){    
            labels_ = {"0", "90", "180", "270"};
            
            model_path_ = model_path;
            Init(model_path_);
        };

        void GetInputInfo();
        void GetOutputInfo();


        void PreProcess(std::string& image_path);
        void PreProcess(cv::Mat& image);
        void Inference();
        void PostProcess();
        std::pair<std::vector<int>, std::vector<float>> ClassifierInferencer::GetRes();

        void Release();
        

    private:
        Ort::SessionOptions options_;
        Ort::Session* session_;
        Ort::Env env_{nullptr};

        std::string image_path_;
        std::string model_path_;
	std::vector<std::string> labels_;
        
	cv::Mat image_;
        // Ort::Value input_tensor_;
        std::vector<Ort::Value> ort_outputs_;
        
        size_t numInputNodes_;  // usually, it is 1
        size_t numOutputNodes_;
        std::vector<std::string> input_node_names_;
	    std::vector<std::string> output_node_names_;
        std::vector<int> input_w_;  // net input (width)  # 事实上，通常仅仅只有1个输入node和1个输出node, 这里仅仅是为了接口通用
        std::vector<int> input_h_;  // net input (height)
        std::vector<int> output_class_num_;  // net output(class_num_)

        float x_factor_;
        float y_factor_;
        float scale_;
        int top_;  // border
        int bottom_;
        int left_;
        int right_;

        std::vector<int> predictions_;
        std::vector<float> scores_;

        void Init(std::string model_path){
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");  //holds the logging state 
            
            Ort::SessionOptions option;
            option.SetIntraOpNumThreads(1);
            option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            session_ = new Ort::Session(env, ConvertToWString(model_path).c_str(), option);
        }

        //Ort::Env
        static Ort::Env CreateEnv(){
           
            return Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov11-onnx");
            
        }

        //Ort::SessionOptions
        static Ort::SessionOptions CreateSessionOptions(){
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            
            return options;
        }

        //convert std::string to std::basic_string<ORTCHAR_T>
        static std::basic_string<ORTCHAR_T> ConvertToWString(std::string& model_path){
            
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }

        static std::string return_image_path(std::string image_path){
            return image_path;
             
        }

        // TO DO, input and output count fix 1
        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();

        cv::Mat pad_and_resize(cv::Mat image);    
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);

};
