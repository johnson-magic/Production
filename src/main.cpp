#include <iostream>
#include <opencv2/opencv.hpp>
#include "classifier/classifier_inferencer.h"
#include "detector/inferencer.h"
#include "sequence/sequence_inferencer.h"

#define PI acos(-1)

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect) {

    cv::Point2f vertices[4];
	


    rotatedRect.points(vertices);

   
    for(int i = 0; i < 4; ++i) {
		cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
	}
      
        
}

std::vector<cv::Point2f> convert_coordinates(const std::vector<cv::Point2f>& coors) {
    // 步骤1：计算中心点
    float center_x = 0.0f, center_y = 0.0f;
    for (const auto& coor : coors) {
        center_x += coor.x;
        center_y += coor.y;
    }
    center_x /= coors.size();
    center_y /= coors.size();

    // 步骤2：分割左右点
    std::vector<cv::Point2f> left_coors, right_coors;
    for (const auto& coor : coors) {
        if (coor.x < center_x) {
            left_coors.push_back(coor);
        } else if (coor.x > center_x) {
            right_coors.push_back(coor);
        }
    }

    // 步骤3：按y坐标排序左右点
    auto sort_by_y = [](const cv::Point2f & a, const cv::Point2f & b) {
        return a.y < b.y; // 升序排列
    };

    // 处理左边点 (保证排序后是 [左上, 左下])
    if (left_coors.size() == 2) {
        std::sort(left_coors.begin(), left_coors.end(), sort_by_y);
    }

    // 处理右边点 (保证排序后是 [右上, 右下])
    if (right_coors.size() == 2) {
        std::sort(right_coors.begin(), right_coors.end(), sort_by_y);
    }

    // 步骤4：组装最终坐标序列
    std::vector<cv::Point2f> result;
    if (!left_coors.empty() && !right_coors.empty()) {
        // 按 [左上，右上，右下，左下] 顺序组装
        result.push_back(left_coors[0]);    // 左上
        result.push_back(right_coors[0]);   // 右上
        result.push_back(right_coors[1]);   // 右下
        result.push_back(left_coors[1]);    // 左下
    }

    return result;
}

cv::Mat transform_and_crop(cv::Mat& image, 
    const std::vector<cv::Point2f>& src, 
    const std::vector<cv::Point2f>& dst,
    int output_width, 
    int output_height) {
/*
:param image: 输入图像 (Mat 对象)
:param src: 源四边形顶点坐标，格式为 vector<Point2f>{ {x1,y1}, {x2,y2}, {x3,y3}, {x4,y4} }
:param dst: 目标四边形顶点坐标，格式同上
:return: 透视变换后的图像
*/

// 计算透视变换矩阵
cv::Mat M = getPerspectiveTransform(src, dst);

// 执行透视变换
cv::Mat warped;
warpPerspective(image, warped, M, cv::Size(output_width, output_height));

return warped;
}

void rotateImage(const cv::Mat& src, cv::Mat& dst, int degree) {
    switch (degree) {
        case 90:  // 顺时针90°
            cv::transpose(src, dst);     // 转置矩阵
            cv::flip(dst, dst, 1);       // 水平翻转
            break;
        case 180: // 顺时针180°
            cv::flip(src, dst, -1);      // 同时水平和垂直翻转
            break;
        case 270: // 顺时针270°
            cv::transpose(src, dst);     // 转置矩阵
            cv::flip(dst, dst, 0);       // 垂直翻转
            break;
        default:
            dst = src.clone();
            break;
    }
}

cv::Point2f GetCenterPoint(std::vector<cv::Point2f>& coors)
{
    float center_x = 0.0f, center_y = 0.0f;
    for (const auto& coor : coors) {
        center_x += coor.x;
        center_y += coor.y;
    }
    center_x /= coors.size();
    center_y /= coors.size();

    return cv::Point2f(center_x, center_y);
}

// 判断点P在AB的哪一侧，返回1（左）、-1（右）、0（线上）
int point_side(cv::Point2f point1, cv::Point2f point2, cv::Point2f point) {
    const double epsilon = 1e-10; // 浮点精度阈值
    
    // 计算向量AB的坐标差
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    
    // 计算向量AP的坐标差
    double APx = point.x - point1.x;
    double APy = point.y - point1.y;
    
    // 计算叉积
    double cross = dx * APy - dy * APx;
    
    // 判断叉积范围
    if (cross > epsilon) return 1;   // 左侧
    if (cross < -epsilon) return -1; // 右侧
    return 0;                        // 线上
}

int main(){
    std::string detector_model_path = "text_det.onnx";
    std::string classifier_model_path = "text_direction_cla.onnx";
    std::string sequence_model_path = "text_rec.onnx";
    std::string charset_path = "charset.txt";

    std::string img_path = "2022-12-08 14-54-28_000001.bmp";
    Inferencer detector(detector_model_path);
    ClassifierInferencer classifier(classifier_model_path);
    SequenceInferencer sequence(sequence_model_path);
    
    detector.GetInputInfo();
	detector.GetOutputInfo();
    detector.PreProcess(img_path);
    detector.Inference();
    detector.PostProcess();
    std::vector<RotatedObj> res = detector.Get_remain_rotated_objects();

    // 以下是检测结果可视化
    cv::Mat image = cv::imread(img_path);
	if (image.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		return -1;
	}
	for(auto rotated_obj : res){
        std::cout<<"111"<<std::endl;
		cv::RotatedRect rotated_rect = rotated_obj.rotated_rect;
		drawRotatedRect(image, rotated_rect);
	}
	cv::imwrite("vis.png", image);

    for(auto rotated_obj : res){

        std::cout<<"class id:"<<rotated_obj.class_index<<std::endl;
        if(rotated_obj.class_index != 2){
            continue;
        }
        
        cv::RotatedRect rotated_rect = rotated_obj.rotated_rect;
        std::vector<cv::Point2f> vertices;
        rotated_rect.points(vertices);
        vertices = convert_coordinates(vertices);  // 确保是按照左上点---》右上点---》右下点---》左下点，方便后续的仿射变换

        double edge1 = cv::norm(vertices[0] - vertices[1]); // 等价于 np.linalg.norm
        double edge2 = cv::norm(vertices[1] - vertices[2]);

        std::vector<cv::Point2f> dst;
        dst = {
            cv::Point2f(0, 0),          // 左上角
            cv::Point2f(edge1, 0),      // 右上角
            cv::Point2f(edge1, edge2), // 右下角
            cv::Point2f(0, edge2)      // 左下角
        };
        int output_width = cvRound(edge1);
        int output_height = cvRound(edge2);
        
        //仿射变换，并抠出小图（检测图）
        cv::Mat image_crop = transform_and_crop(image, 
            vertices,  // src 
            dst,
            output_width, 
            output_height);
        
        // 文字方向的判断，并基于方向判断进行旋转调整
        classifier.GetInputInfo();
        classifier.GetOutputInfo();
        classifier.PreProcess(image_crop);  // 不仅仅要支持image_path, 而且要支持image TO DO
        classifier.Inference();
        classifier.PostProcess();
        std::pair<std::vector<int>, std::vector<float>> classifier_res = classifier.GetRes();
        std::vector<int> classifier_classes = classifier_res.first;
        std::vector<float> classifier_scores = classifier_res.second;

        int cls = classifier_classes[0];  // 0 for 0 degree, 1 for 90 degree, 2 for 180 degree, 3 for 270 degree; 顺时针旋转360-degree即为正
        std::cout<<"Angle: "<<cls*90<<std::endl;
        // int score = scores[0];
        rotateImage(image_crop, image_crop, 360 - cls * 90);


        // 送入文字识别模型，输出字符串
        sequence.GetInputInfo();
	    sequence.GetOutputInfo();
        sequence.PreProcess(image_crop);  //不仅仅要支持image_path, 而且要支持image TO DO
        sequence.Inference();
        sequence.PostProcess();
        std::pair<std::vector<int>, std::vector<float>> sequence_res = sequence.GetRes();
        std::vector<int> sequence_classes = sequence_res.first;
        std::vector<float> sequence_scores = sequence_res.second;
        for(int j=0; j<sequence_classes.size(); j++){
            std::cout<<"class: "<<sequence_classes[j]<<"scores: "<<sequence_scores[j]<<std::endl;
        }
    }

    // 基于上述所有信息，汇总最终的业务输出
    std::vector<std::vector<cv::Point2f>> shape;
    std::vector<std::vector<cv::Point2f>> shape_half_ellipse;
    std::vector<std::vector<cv::Point2f>> text;

    
    for(auto rotated_obj : res){
        cv::RotatedRect rotated_rect = rotated_obj.rotated_rect;
        int class_id = rotated_obj.class_index;
        std::vector<cv::Point2f> vertices;
        rotated_rect.points(vertices);
        
        if(class_id == 0){
            shape.push_back(vertices);
        }
        else if(class_id == 0){
            shape_half_ellipse.push_back(vertices);
        }
        else{
            text.push_back(vertices);
        }
	}

    if(shape.size()==2 && text.size()==1){  // 最期待看到的场景
        cv::Point2f point1 = GetCenterPoint(shape[0]);
        cv::Point2f point2 = GetCenterPoint(shape[1]);
        cv::Point2f text_point = GetCenterPoint(text[0]);
        float distance = cv::norm(point1 - point2);

        double dx = point2.x - point1.x;
        double dy = point2.y - point1.y;
        double angle_rad = atan2(dy, dx);
        double angle_deg = angle_rad * 180.0 / PI;

        int side = point_side(point1, point2, text_point);
        int isright = 1;
        if(side == 1){
            isright = 1;
        }  // 左侧
        else if(side == -1){  // 右侧
            isright = -1;
        }

    std::ofstream outFile("res.txt");
	if (!outFile) {
		std::cerr << "can not open: " << "res.txt" << std::endl;
		return 0;
	}

	outFile<<"Point1 " <<point1.x <<"," << point1.y << std::endl;
	outFile<<"Point2 "<<point2.x <<"," << point2.y << std::endl;
	outFile << "Distance " <<distance << std::endl;
    outFile << "Angle " <<angle_deg << std::endl;
    outFile << "IsRight " <<isright << std::endl;
	outFile.close();
    }
    else{  // 异常情形，后续逐步完善
        std::cout<<"to deal"<<std::endl;
    }

    return 0;
}