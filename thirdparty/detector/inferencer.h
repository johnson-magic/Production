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



class Inferencer {
    public:
        Inferencer(std::string& model_path){    

            modelScoreThreshold_ = 0.2;
            modelNMSThreshold_ = 0.8;
            labels_ = {"big_cirlce","plates","slide"};
            
            // image_path_ = return_image_path(image_path);
            model_path_ = model_path;
            Init(model_path_);
        };
        // Inferencer(std::string& model_path, std::string& image_path){};

        
        void GetInputInfo();
        void GetOutputInfo();


        void PreProcess(std::string& image_path);
        void PreProcess(cv::Mat image);
        void Inference();
        void PostProcess();

        void Release();

        std::vector<RotatedObj> Get_remain_rotated_objects();

    private:
        Ort::SessionOptions options_;
        Ort::Session* session_;
        Ort::Env env_{nullptr};

        std::string image_path_;
        std::string model_path_;
        
        cv::Mat image_;
        // Ort::Value input_tensor_;
        std::vector<Ort::Value> ort_outputs_;
        
        // size_t numInputNodes_;  now, only support 1
        // size_t numOutputNodes_;
        std::vector<std::string> input_node_names_;
	    std::vector<std::string> output_node_names_;
        int input_w_;  // net input (width)
        int input_h_;  // net input (height)
        int output_w_;  // net output (width)
        int output_h_;  // net output (height)

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
        std::vector<std::string> labels_;

        void Init(std::string model_path){
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");
            
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
        //size_t GetSessionInputCount();
        //size_t GetSessionOutputCount();

        cv::Mat formatToSquare(cv::Mat image);
        void PrepareForNms(const cv::Mat & det_output, const int & i, cv::Point classIdPoint, const double & score, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);

        void Nms(std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);

};
