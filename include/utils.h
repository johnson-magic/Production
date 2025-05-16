#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unordered_set>
#include <algorithm>
#include <string>

#include "time_limit.h"


void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect, cv::Scalar color);
void rotateImage(cv::Mat& src, cv::Mat& dst, int degree);
cv::Point2f GetCenterPoint(std::vector<cv::Point2f>& coors);
std::vector<cv::Point2f> convert_coordinates(const std::vector<cv::Point2f>& coors);
int point_side(cv::Point2f point1, cv::Point2f point2, cv::Point2f point);
cv::Mat cropAffineTransformedQuadrilateral(const cv::Mat &image, const cv::RotatedRect &rotate_rect);
bool hasImageUpdated(const std::string& img_path, cv::Scalar &pre_pixel_sum);
bool check_substring_exists(const std::string& s);
bool check_string_exists(const std::string& s);

void readFromBinaryFile(const std::string& filename, const TimeLimit& timelimit);
void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename);
int encrypt(int number, int key);
int decrypt(int encrypted_number, int key);