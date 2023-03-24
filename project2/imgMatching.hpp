/**
 CS 5330 Computer Vision
 Project 2
 Wenlin Fang
*/
#ifndef imgMatching_hpp
#define imgMatching_hpp

#include <stdio.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

#endif /* imgMatching_hpp */

// all matching methods
int baselineMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfilenames, int matching_method);
int histogramMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int multiHistogramMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int textureAndColor(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int lawsFilterMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int customMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int HSVMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);
int gaborFilterMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method);

// Gabor filter helper functions
int calcGaborFeatures(char* file, std::vector<float> &features);
int calcGaborAllFiles(std::vector<char*> filenames);

// Laws filter helper functions
int calcLawsFeatures(char* file, std::vector<float> &features);
int calcLawsL5E5AllFiles(std::vector<char*> filenames);
int calcLawsS5W5AllFiles(std::vector<char*> filenames);
int calcLawsL5L5AllFiles(std::vector<char*> filenames);
int calcLawsE5E5AllFiles(std::vector<char*> filenames);
int calcLawsE5W5AllFiles(std::vector<char*> filenames);
int calcLawsFeaturesS5W5(char* file, std::vector<float> &features);
int calcLawsFeaturesL5E5(char* file, std::vector<float> &features);
int calcLawsFeaturesE5E5(char* file, std::vector<float> &features);
int calcLawsFeaturesE5W5(char* file, std::vector<float> &features);
int calcLawsFeaturesL5L5(char* file, std::vector<float> &features);

// HSV matching helper functions
int calcHSVAllFiles(std::vector<char*> filenames);
int calcHSV(char* filename, std::vector<float> &features);

// texture and color helper functions
int calcTextureAllFiles(std::vector<char*> filenames);
int calcTexture(char* filename, std::vector<float> &features);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// baseline matching helper functions
int calcBaselineFeatures(char* file, std::vector<float> &features);
int calcBaselineAllFiles(std::vector<char*> filenames);

// multi histogram helper functions
int calcTopOrBottomHistogram(char* filename, int pos, std::vector<float> &features);
int calcMultiHistogramsAllFiles(std::vector<char*> filenames);

// histogram matching helper functions
int calcHistograms(cv::Mat src, std::vector<float> &features);
int calcHistogramsAllFiles(std::vector<char*> filenames);
int histogramIntersection(std::vector<float> file, std::vector<float> target, float &intersection);

// center histogram matching helper functions
int calcCenterAllFiles(std::vector<char*> filenames);
int getCenterPixels(char* filename, cv::Mat &dst);
int calcCenterHistogram(char* filename, std::vector<float> &features);

// utility functions
int findTopK(std::vector<std::pair<float, char*>> distances, int ascending, int k,  std::vector<char*> &topK);
int filename_conversion(std::string orig_file, char* &converted);
int calculateDistances(std::vector<std::vector<float>> compare_features, std::vector<char *> filenames, std::vector<float> target_features, std::vector<std::pair<float, char*>> &distances, int matchingMethod);
int findKNeighbors(std::vector<char*> filenames, char* targetfile, int k_neighbors, int feature_type, int matching_method, std::vector<char*> &outputfilenames);
