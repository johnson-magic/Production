#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <filesystem>
#include <chrono>
#include <thread>
#include <sys/stat.h>

#include "detector_pipeline.h"
#include "classifier_pipeline.h"
#include "sequence_pipeline.h"
#include "plugin_pipeline.h"
#include "utils.h"


volatile bool keepRunning = true;

BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}





std::mutex file_mutex;  // 全局锁
std::filesystem::file_time_type lastCheckedTime;

bool check_file_modified(const std::string& image_path) {
    std::lock_guard<std::mutex> lock(file_mutex);  // 自动加锁
    
    if (!std::filesystem::exists(image_path)) {
        std::cout << "File does not exist: " << image_path << std::endl;
        return false;
    }
    
    auto curWriteTime = std::filesystem::last_write_time(image_path);
    if (curWriteTime != lastCheckedTime) {
        lastCheckedTime = curWriteTime;
        return true;
    }
    return false;
}

cv::Mat loadImageWithRetry(const std::string& img_path) {
    const int max_retries = 3;
    cv::Mat img;
    
    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        try {
            img = cv::imread(img_path);
            if (!img.empty()) return img;
            
            std::cerr << "[Attempt " << attempt 
                      << "] Failed to read image: " 
                      << img_path << std::endl;
        } 
        catch (const cv::Exception& e) {
            std::cerr << "[Attempt " << attempt 
                      << "] Exception: " << e.what() << std::endl;
        }
        
        if (attempt < max_retries) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        }
    }
    
    // 报警逻辑（此处以抛出异常为例）
    throw std::runtime_error("Critical error: Failed to load image after " + 
                            std::to_string(max_retries) + " attempts");
}



bool isFileStable(const std::string& path) {
    struct stat st1, st2;
    if(stat(path.c_str(), &st1) != 0) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(stat(path.c_str(), &st2) != 0) return false;
    return (st1.st_size == st2.st_size); // 文件大小稳定
}

static int stamp = 0;

int main(int argc, char** argv){

    if(argc != 8){
        std::cout<<"[ERROR] production.exe  text_det.onnx text_direction_cla.onnx text_rec.onnx charset.txt test.jpg res.txt vis.jpg"<<std::endl;
		return 0;
    }
    std::string detector_model_path = argv[1];
    std::string classifier_model_path = argv[2];
    std::string sequence_model_path = argv[3];
    std::string charset_path = argv[4];
    std::string img_path = argv[5];
    std::string res_path = argv[6];
    std::string vis_path = argv[7];

    Inferencer *detectordfd = new Inferencer(detector_model_path);
    ClassifierInferencer *classifier = new ClassifierInferencer(classifier_model_path);
    SequenceInferencer *sequence = new SequenceInferencer(sequence_model_path, charset_path);

    std::filesystem::file_time_type lastCheckedTime = std::filesystem::file_time_type();
    
    cv::Scalar pre_pixel_sum=cv::Scalar(0, 0, 0, 0);


    while (keepRunning) {
        if (hasImageUpdated(img_path, pre_pixel_sum)) {

            std::vector<RotatedObj> detector_res;
            std::vector<int> direction_res;
            std::vector<std::string> text_res;
            
            cv::Mat img;
            std::ifstream test_open(img_path);
            if(test_open.is_open()){
                img = cv::imread(img_path);
                test_open.close(); 
            }else{
                continue;
            }

            if(img.empty()){
                std::cout<<"open img failed"<<std::endl;
            }
            int img_w = img.cols;
            int img_h = img.rows;

            detector_res = detector_infer(detectordfd, img);
            for(auto rotated_obj : detector_res){
                // cv::Point2f vertices[4];
                // rotated_obj.rotated_rect.points(vertices);
                // if(rotated_obj.class_index == 0){
                //     drawRotatedRect(img_copy, rotated_obj.rotated_rect, cv::Scalar(255, 0, 0));
                // }
                // else if(rotated_obj.class_index == 1){
                //     drawRotatedRect(img_copy, rotated_obj.rotated_rect, cv::Scalar(0, 255, 0));
                // }
                // else{
                //     drawRotatedRect(img_copy, rotated_obj.rotated_rect, cv::Scalar(0, 0, 255));
                // }
                if(rotated_obj.class_index != 2){
                    continue;
                }
                cv::Mat img_crop = cropAffineTransformedQuadrilateral(img, rotated_obj.rotated_rect);
                
                int cls = classifier_pipeline(classifier, img_crop);
                
                direction_res.push_back(cls);
                
                rotateImage(img_crop, img_crop, 360 - cls * 90);
                
                
                std::string text = sequence_pipeline(sequence, img_crop);
                text_res.push_back(text);

            }

            plugin_pipeline(detector_res, direction_res, text_res, res_path, img, vis_path, img_w, img_h);
            std::cout<<"process finished"<<std::endl;
            img.release();

            
            
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "exit" << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}