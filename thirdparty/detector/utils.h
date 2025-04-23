#pragma once
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "data_structure.h"

#include "time_limit.h"

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect);
void printRotatedRect(const cv::RotatedRect& rotatedRect);
bool hasImageUpdated(const std::string& image_path, std::filesystem::file_time_type& lastCheckedTime);
long long GetSecondsInterval(SYSTEMTIME start, SYSTEMTIME end);
void SaveOrtValueToTextFile(Ort::Value& ortValue, const std::string& filename);
std::pair<std::string, std::string> splitext(const std::string& filename);
void SaveRotatedObjsToTextFile(std::vector<RotatedObj>& rotated_objs, const std::string& filename);
void readFromBinaryFile(const std::string& filename, const TimeLimit& timelimit);
void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename);
int encrypt(int number, int key);
int decrypt(int encrypted_number, int key);
