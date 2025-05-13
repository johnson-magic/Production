#include "plugin_pipeline.h"
#include <math.h>

#define PI acos(-1)

std::vector<cv::Point2f> computeMidPoints(const std::vector<cv::Point2f>& rect) {
    std::vector<cv::Point2f> midPoints;
    if (rect.size() != 4) return midPoints;

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& p = rect[i];
        const cv::Point2f& q = rect[(i + 1) % 4];
        midPoints.emplace_back((p.x + q.x) * 0.5f, (p.y + q.y) * 0.5f);
    }
    return midPoints;
}

void plugin_input_adapter(const std::vector<RotatedObj> &detector_res,
    std::vector<std::vector<cv::Point2f>>& shape_objs, std::vector<std::vector<cv::Point2f>>& shape_half_ellipse_objs,
    std::vector<std::vector<cv::Point2f>>& text_objs)
{
       
    for(auto rotated_obj : detector_res){
        cv::RotatedRect rotated_rect = rotated_obj.rotated_rect;
        std::vector<cv::Point2f> vertices;
        rotated_rect.points(vertices);
        int class_id = rotated_obj.class_index;
        
        if(class_id == 0){
            shape_objs.push_back(vertices);
        }
        else if(class_id == 1){
            shape_half_ellipse_objs.push_back(vertices);
        }
        else{
            text_objs.push_back(vertices);
        }
	}
}

void get_center_point_case1(std::vector<cv::Point2f> &shape_obj1, std::vector<cv::Point2f> &shape_obj2,
    cv::Point2f &point1, cv::Point2f &point2){
    point1 = GetCenterPoint(shape_obj1);
    point2 = GetCenterPoint(shape_obj2);
}

void get_center_point_case2(std::vector<cv::Point2f> shape_half_ellipse_obj1, std::vector<cv::Point2f> shape_half_ellipse_obj2,
    cv::Point2f &point1, cv::Point2f &point2){
    cv::Point2f c1 = GetCenterPoint(shape_half_ellipse_obj1);
    cv::Point2f c2 = GetCenterPoint(shape_half_ellipse_obj2);

    shape_half_ellipse_obj1 = convert_coordinates(shape_half_ellipse_obj1);
    shape_half_ellipse_obj2 = convert_coordinates(shape_half_ellipse_obj2);

    std::vector<cv::Point2f> allMidpoints;
    for (int i = 0; i < 4; ++i) {
        
        const cv::Point2f& p1 = shape_half_ellipse_obj1[i];
        const cv::Point2f& p2 = shape_half_ellipse_obj1[(i+1)%4];
        
        
        allMidpoints.emplace_back(
            (p1.x + p2.x) * 0.5f,
            (p1.y + p2.y) * 0.5f
        );
    }

    for (int i = 0; i < 4; ++i) {
        
        const cv::Point2f& p1 = shape_half_ellipse_obj2[i];
        const cv::Point2f& p2 = shape_half_ellipse_obj2[(i+1)%4];
        
        
        allMidpoints.emplace_back(
            (p1.x + p2.x) * 0.5f,
            (p1.y + p2.y) * 0.5f
        );
    }

    float maxDist = 0.0f;
    for (size_t i = 0; i < allMidpoints.size(); ++i) {
        for (size_t j = i + 1; j < allMidpoints.size(); ++j) {
            float dist = cv::norm(allMidpoints[i] - allMidpoints[j]);
            if (dist > maxDist) {
                maxDist = dist;
                point1 = allMidpoints[i];
                point2 = allMidpoints[j];
            }
        }
    }
}

void save_res(cv::Point2f point1, cv::Point2f point2, float distance, float angle_deg, int isright, std::string save_path){
    std::ofstream outFile(save_path);
	if (!outFile) {
		std::cerr << "can not open: " << "res.txt" << std::endl;
		return;
	}
	outFile<<"Point1 "<<point1.x<<","<<point1.y<<std::endl;
	outFile<<"Point2 "<<point2.x <<","<<point2.y<<std::endl;
	outFile<<"Distance "<<distance<<std::endl;
    outFile<<"Angle "<<angle_deg<<std::endl;
    outFile<<"IsRight "<<isright<<std::endl;
	outFile.close();
}

cv::Point2f rotatePoint(const cv::Point2f& point, 
                       int angle, 
                       int width, 
                       int height) 
{
    
    switch(angle % 360) {
    case 270:
        return cv::Point2f(
            point.y,             
            width - point.x - 1  
        );                       

    case 180:
        return cv::Point2f(
            width - point.x - 1, 
            height - point.y - 1
        );

    case 90: 
        return cv::Point2f(
            height - point.y - 1, 
            point.x
        );

    default: 
        return point;
    }
}


int isright_fun(cv::Point2f point1, cv::Point2f point2, cv::Point2f text_point, int cls, int img_w, int img_h){
    int angle = 360 - cls * 90;  
    point1 = rotatePoint(point1, angle, img_w, img_h);
    point2 = rotatePoint(point2, angle, img_w, img_h);
    text_point = rotatePoint(text_point, angle, img_w, img_h);

    float distance1 = cv::norm(point1 - text_point);
    float distance2 = cv::norm(point2 - text_point);

    if(point1.x < point2.x){
        if(distance1 < distance2){
            return 0;  
        }
        else if(distance1 > distance2){
            return 1;  
        }
        else{
            return -1; 
        }
    }
    else if(point1.x > point2.x){
        if(distance1 < distance2){
            return 1;  
        }
        else if(distance1 > distance2){
            return 0;  
        }
        else{
            return -1; 
        }
    }
    else{
        return -1;  
    }

}

void get_angle(cv::Point2f point1, cv::Point2f point2, int cls, int img_w, int img_h, float &angle){

    int angle_rotate = (360 - cls * 90) % 360;  
    point1 = rotatePoint(point1, angle_rotate, img_w, img_h);
    point2 = rotatePoint(point2, angle_rotate, img_w, img_h);
    

    const double dx = point2.x - point1.x;  
    const double dy = point2.y - point1.y; 
    
    
    const double radians = std::atan(dy/dx);
    
    
    double degrees = radians * (180.0 / PI);

    if(degrees - angle_rotate > 0){
        angle = fmod(degrees - angle_rotate, 180);
        if(angle > 90){
            angle = angle -180;
        }
    }
    else if(degrees - angle_rotate < 0){
        angle = fmod(degrees - angle_rotate, -180);
        if(angle <= -90){
            angle = 180 + angle;
        }
    }
    else{
        angle = 0;
    }
   
}
// detector_res, direction_res, text_res, res_path, img, vis_path, img_w, img_h

void plugin_pipeline(const std::vector<RotatedObj> &detector_res, const std::vector<int> &direction_res, const std::vector<std::string> &text_res,
    const std::string &save_path, cv::Mat img, const std::string &vis_path, const int img_w, const int img_h){

   
    cv::Point2f point1(-1.0f, -1.0f), point2(-1.0f, -1.0f);
    float distance=-1.0, angle_deg=360.0;
    int isright=-1;

   
    std::vector<std::vector<cv::Point2f>> shape_objs;
    std::vector<std::vector<cv::Point2f>> shape_half_ellipse_objs;
    std::vector<std::vector<cv::Point2f>> text_objs;
    plugin_input_adapter(detector_res, shape_objs, shape_half_ellipse_objs, text_objs);
    if((shape_objs.size()==2 || shape_half_ellipse_objs.size()==2) && text_objs.size()>=1){
       
        int cls = -1;
        std::string text="";
        std::vector<cv::Point2f> text_obj;
        for(int i=0; i<text_objs.size(); i++){
            if(check_substring_exists(text_res[i])){
                cls = direction_res[i];
                text = text_res[i];
                text_obj = text_objs[i];
                break;
            }
        }

        if(text == ""){ 
            std::cout<<"save default results, case1"<<std::endl; 
            save_res(point1, point2, distance, angle_deg, isright, save_path);
            return;
        }
        
        if(shape_objs.size()==2){ 
            get_center_point_case1(shape_objs[0], shape_objs[1], point1, point2); 
        }
        else{
            get_center_point_case2(shape_half_ellipse_objs[0], shape_half_ellipse_objs[1], point1, point2); 
        }
        distance = cv::norm(point1 - point2);

        get_angle(point1, point2, cls, img_w, img_h, angle_deg);

        
        cv::Point2f text_point = GetCenterPoint(text_obj);
        isright = isright_fun(point1, point2, text_point, cls, img_w, img_h);
        save_res(point1, point2, distance, angle_deg, isright, save_path);

        cv::circle(img, point1, 3, (0, 0, 255), -1);
        cv::circle(img, point2, 3, (0, 0, 255), -1);
        cv::imwrite(vis_path, img);
        
    }
    else{ 
        std::cout<<"save default results, case2"<<std::endl;
        save_res(point1, point2, distance, angle_deg, isright, save_path);
    }

}

