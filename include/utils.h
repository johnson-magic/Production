#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unordered_set>
#include <algorithm>


void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect, cv::Scalar color);
void rotateImage(cv::Mat& src, cv::Mat& dst, int degree);
cv::Point2f GetCenterPoint(std::vector<cv::Point2f>& coors);
std::vector<cv::Point2f> convert_coordinates(const std::vector<cv::Point2f>& coors);
int point_side(cv::Point2f point1, cv::Point2f point2, cv::Point2f point);
cv::Mat cropAffineTransformedQuadrilateral(const cv::Mat &image, const cv::RotatedRect &rotate_rect);
bool hasImageUpdated(const std::string& img_path, cv::Scalar &pre_pixel_sum);
bool check_substring_exists(const std::string& s);