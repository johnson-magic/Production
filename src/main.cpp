#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <Windows.h>

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

int main(int argc, char** argv){

    #ifdef ENCRYPT
		TimeLimit timelimit;
		readFromBinaryFile("onnx.dll", timelimit);
		int left = decrypt(timelimit.left, 20250124);
	#endif

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

    DetectorInferencer *detector= new DetectorInferencer(detector_model_path, 3);
    ClassifierInferencer *classifier = new ClassifierInferencer(classifier_model_path);
    SequenceInferencer *sequence = new SequenceInferencer(sequence_model_path, charset_path);

    std::filesystem::file_time_type lastCheckedTime = std::filesystem::file_time_type();
    
    cv::Scalar pre_pixel_sum=cv::Scalar(0, 0, 0, 0);


    while (keepRunning) {
        if (hasImageUpdated(img_path, pre_pixel_sum)) {
            #ifdef ENCRYPT
				if(left == 0){
					std::cerr<<"Error 3, please contact the author!"<<std::endl;
					return 0;
				}
				left = left - 1;
				timelimit.left = encrypt(left, 20250124);
				saveToBinaryFile(timelimit, "onnx.dll");
			#endif

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

            detector_res = detector_infer(detector, img);
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