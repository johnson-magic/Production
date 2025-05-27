#include "utils.h"

// #define PI acos(-1)

bool hasImageUpdated(const std::string& img_path, cv::Scalar &pre_pixel_sum){
    cv::Mat img_temp = cv::imread(img_path);
    cv::Scalar cur_pixel_sum=cv::sum(img_temp);
    if(pre_pixel_sum == cur_pixel_sum){
        return false;
    }
    else{
        pre_pixel_sum = cur_pixel_sum;
        return true;
    }
}

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect, cv::Scalar color) {

    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    for(int i = 0; i < 4; ++i) {
		cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
	}
          
}


std::vector<cv::Point2f> convert_coordinates(const std::vector<cv::Point2f>& coors) {

    float center_x = 0.0f, center_y = 0.0f;
    for (const auto& coor : coors) {
        center_x += coor.x;
        center_y += coor.y;
    }
    center_x /= coors.size();
    center_y /= coors.size();


    std::vector<cv::Point2f> left_coors, right_coors;
    for (const auto& coor : coors) {
        if (coor.x < center_x) {
            left_coors.push_back(coor);
        } else if (coor.x > center_x) {
            right_coors.push_back(coor);
        }
    }


    auto sort_by_y = [](const cv::Point2f & a, const cv::Point2f & b) {
        return a.y < b.y;
    };


    if (left_coors.size() == 2) {
        std::sort(left_coors.begin(), left_coors.end(), sort_by_y);
    }


    if (right_coors.size() == 2) {
        std::sort(right_coors.begin(), right_coors.end(), sort_by_y);
    }


    std::vector<cv::Point2f> result;
    if (!left_coors.empty() && !right_coors.empty()) {
        
        result.push_back(left_coors[0]); 
        result.push_back(right_coors[0]);
        result.push_back(right_coors[1]); 
        result.push_back(left_coors[1]); 
    }

    return result;
}


cv::Mat transform_and_crop(const cv::Mat& image, 
    const std::vector<cv::Point2f>& src, 
    const std::vector<cv::Point2f>& dst,
    int output_width, 
    int output_height) {

cv::Mat M = getPerspectiveTransform(src, dst);


cv::Mat warped;
warpPerspective(image, warped, M, cv::Size(output_width, output_height));

return warped;
}

void rotateImage(cv::Mat& src, cv::Mat& dst, int degree) {
    switch (degree) {
        case 90: 
            cv::transpose(src, dst);    
            cv::flip(dst, dst, 1);       
            break;
        case 180:
            cv::flip(src, dst, -1);     
            break;
        case 270:
            cv::transpose(src, dst);    
            cv::flip(dst, dst, 0);      
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


int point_side(cv::Point2f point1, cv::Point2f point2, cv::Point2f point) {
    const double epsilon = 1e-10;
    
  
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    
   
    double APx = point.x - point1.x;
    double APy = point.y - point1.y;
    
   
    double cross = dx * APy - dy * APx;
    
  
    if (cross > epsilon) return 1;   
    if (cross < -epsilon) return -1; 
    return 0;                       
}




cv::Mat cropAffineTransformedQuadrilateral(const cv::Mat &image, const cv::RotatedRect &rotated_rect){
    std::vector<cv::Point2f> vertices;
    rotated_rect.points(vertices);
    vertices = convert_coordinates(vertices); 

    double edge1 = cv::norm(vertices[0] - vertices[1]); 
    double edge2 = cv::norm(vertices[1] - vertices[2]);

    std::vector<cv::Point2f> dst;
    dst = {
        cv::Point2f(0, 0),          
        cv::Point2f(edge1, 0),      
        cv::Point2f(edge1, edge2), 
        cv::Point2f(0, edge2)      
    };
    int output_width = cvRound(edge1);
    int output_height = cvRound(edge2);
    
   
    cv::Mat image_crop = transform_and_crop(image, 
        vertices,  // src 
        dst,
        output_width, 
        output_height);
    return image_crop;
}


bool check_substring_exists(const std::string& s) {
    if (s.empty()) {
        return false;
    }

    static const std::vector<std::string> target_texts = {
        "02", "05", "07", "10", "12", "15", "17", "20", "22", "25", "27", "30", "32", "35", "37",
        "025", "25", "050", "50", "075", "75", "100", "125", "150", "175", "200", "225", "250", "275",
        "300", "325", "350", "375"
    };

    static const std::unordered_set<std::string> targets(target_texts.begin(), target_texts.end());
    static const int max_len = []{
        int max_len = 0;
        for (const auto& str : target_texts) {
            if (str.size() > max_len) {
                max_len = str.size();
            }
        }
        return max_len;
    }();

    for (size_t i = 0; i < s.size(); ++i) {
        const int remaining_length = s.size() - i;
        const int current_max_len = std::min(max_len, remaining_length);
        
        // 检查从最长到最短的子字符串
        for (int len = current_max_len; len >= 1; --len) {
            std::string substr = s.substr(i, len);
            if (targets.find(substr) != targets.end()) {
                return true;
            }
        }
    }

    return false;
}

bool check_string_exists(const std::string& s) {
    if (s.empty()) {
        return false;
    }

    static const std::vector<std::string> target_texts = {
        "02", "05", "07", "10", "12", "15", "17", "20", "22", "25", "27", "30", "32", "35", "37",
        "025", "25", "050", "50", "075", "75", "100", "125", "150", "175", "200", "225", "250", "275",
        "300", "325", "350", "375"
    };

    std::string s_clone = s;
    if ((s_clone[0] == '-' || s_clone[0] == '+')) {
        s_clone.erase(0, 1);
    }

    return std::find(target_texts.begin(), target_texts.end(), s_clone) != target_texts.end();
}


void readFromBinaryFile(const std::string& filename, const TimeLimit& timelimit) {
    // 读取二进制文件
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr <<"Error 1, please contact the author!"<< filename << std::endl;
        return;
    }

    infile.read((char*)&timelimit, sizeof(timelimit));
    infile.close();
}


void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename) {
    // 打开二进制文件用于写入
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error 2, please contact the author!" << filename << std::endl;
        return;
    }

    outfile.write((char*)&timelimit, sizeof(timelimit));
    outfile.close();
}
 
int encrypt(int number, int key) {
    return number ^ key;
}
 
int decrypt(int encrypted_number, int key) {
    return encrypted_number ^ key;
}