/**
 CS 5330 Computer Vision
 Project 2
 Wenlin Fang
*/
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iterator>
#include <utility>
#include <opencv2/opencv.hpp>
#include "csv_util.hpp"
#include "imgMatching.hpp"

/**
 This function finds the top k matches for the given target file from the given files.
 Parameters:
 - std::vector<char*> filenames - names of all files
 - char* targetfile - name of the target file
 - int k_neighbors - number of matches we want
 - int feature_type - the feature we want to use to find the matches
 - int matching_method - what distance metric we want to use
 - std::vector<char*> &outputfilenames - to store the top k matches we find
 */
int findKNeighbors(std::vector<char*> filenames, char* targetfile, int k_neighbors, int feature_type, int matching_method, std::vector<char*> &outputfilenames) {
    if (feature_type == 0) {
        baselineMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 1) {
        histogramMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 2) {
        multiHistogramMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 3) {
        textureAndColor(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 4) {
        customMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 5) {
        lawsFilterMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 6) {
        HSVMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    if (feature_type == 7) {
        gaborFilterMatching(filenames, targetfile, k_neighbors, outputfilenames, matching_method);
    }
    return 0;
}

/**
 This function is for task 5 custom matching. It uses texture, color histograms for the entire image, and color histograms for the 300 * 300 pixels in the center of the image as features, and histogram intersections to find the top k matches.
 Paramteters:
 - std::vector<char*> filenames - names of all files
 - char* targetfile - name of the target file
 - int k_neighbors - number of matches we want
 - int matching_method - what distance metric we want to use
 - std::vector<char*> &outputfilenames - to store the top k matches we find
 */
int customMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    std::ifstream ifile;
    // check if file exists
    ifile.open("texture.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcTextureAllFiles(filenames);
    }
    ifile.close();

    ifile.open("histogram_feature.csv");
    if(!ifile) {
        calcHistogramsAllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("center_histogram_feature.csv");
    if(!ifile) {
        calcCenterAllFiles(filenames);
    }
    ifile.close();
    
    // calculate texture features for target file
    std::vector<float> target_features;
    calcTexture(targetfile, target_features);
    
    // calculate color feature for target
    std::vector<float> target_colors;
    calcHistograms(cv::imread(targetfile), target_colors);
    
    // calculate center color histogram for target
    std::vector<float> target_centers;
    calcCenterHistogram(targetfile, target_centers);
    
    // calculate center color distances between target and all files
    std::vector<std::pair<float, char*>> center_distances;
    char* inputfile;
    filename_conversion("center_histogram_feature.csv", inputfile);
    std::vector<char *> allfilesCenters;
    std::vector<std::vector<float>> dataCenters;
    read_image_data_csv(inputfile, allfilesCenters, dataCenters, 0);
    calculateDistances(dataCenters, allfilesCenters, target_centers, center_distances, matching_method);
    
    // calculate color distances between target and all files
    std::vector<std::pair<float, char*>> distances;
    filename_conversion("histogram_feature.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_colors, distances, matching_method);
    
    // calculate texture distances
    std::vector<std::pair<float, char*>> texture_distances;
    char* texture_inputfile;
    filename_conversion("texture.csv", texture_inputfile);
    std::vector<char *> allInfiles;
    std::vector<std::vector<float>> texture_data;
    read_image_data_csv(texture_inputfile, allInfiles, texture_data, 0);
    calculateDistances(texture_data, allInfiles, target_features, texture_distances, matching_method);
    
    std::vector<std::pair<float, char*>> avg_distances;
    for (int i = 0; i < texture_distances.size(); i++) {
        avg_distances.push_back(std::make_pair( 0.45 * texture_distances.at(i).first + 0.2 * distances.at(i).first + 0.35 * center_distances.at(i).first, texture_distances.at(i).second));
    }
    
    // find top k matches
    findTopK(avg_distances, 0, k_neighbors, outputfiles);

    return 0;
}

/**
 This function applies W5 and R5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesW5R5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int W5[] = {1, -2, 0, 2, -1};
    int R5[] = {1, -4, 6, -4, 1};
    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * W5[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * W5[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * W5[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * W5[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * W5[4]) / 3;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * R5[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * R5[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * R5[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * R5[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * R5[4]) / 8;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function applies L5 and E5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesL5E5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int L5[] = {1, 4, 6, 4, 1};
    int E5[] = {1, 2, 0, -2, -1};
    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * L5[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * L5[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * L5[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * L5[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * L5[4]) / 3;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * E5[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * E5[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * E5[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * E5[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * E5[4]) / 8;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function applies E5 and E5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesE5E5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int filter[] = {1, 2, 0, -2, -1};

    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * filter[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * filter[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * filter[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * filter[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * filter[4]) / 16;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * filter[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * filter[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * filter[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * filter[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * filter[4]) / 16;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function applies L5 and L5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesL5L5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int filter[] = {1, 4, 6, 4, 1};
    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * filter[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * filter[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * filter[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * filter[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * filter[4]) / 16;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * filter[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * filter[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * filter[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * filter[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * filter[4]) / 16;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function applies W5 and E5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesE5W5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int E5[] = {1, 2, 0, -2, -1};
    int W5[] = {1, -2, 0, 2, -1};
    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * E5[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * E5[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * E5[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * E5[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * E5[4]) / 3;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * W5[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * W5[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * W5[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * W5[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * W5[4]) / 3;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function applies S5 and W5 vectors of the Laws filter to a given image and stores the result.
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsFeaturesS5W5(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2HSV);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int S5[] = {-1, 0, 2, 0, -1};
    int W5[] = {1, -2, 0, 2, -1};
    for (int j = 0; j < src.rows; j++) {
        for (int i = 2; i < src.cols - 2; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(gray.at<cv::Vec3b>(j - 2, i)[k] * S5[0]
                                + gray.at<cv::Vec3b>(j - 1, i)[k] * S5[1]
                                + gray.at<cv::Vec3b>(j, i)[k] * S5[2]
                                + gray.at<cv::Vec3b>(j + 1, i)[k] * S5[3]
                                + gray.at<cv::Vec3b>(j + 2, i)[k] * S5[4]) / 3;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int j = 2; j < src.rows - 2; j++) {
        for (int i = 0; i < src.cols; i++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(j, i)[k] = (int)(temp.at<cv::Vec3b>(j, i - 2)[k] * W5[0]
                                + temp.at<cv::Vec3b>(j, i - 1)[k] * W5[1]
                                + temp.at<cv::Vec3b>(j, i)[k] * W5[2]
                                + temp.at<cv::Vec3b>(j, i + 1)[k] * W5[3]
                                + temp.at<cv::Vec3b>(j, i + 2)[k] * W5[4]) / 3;
            }
        }
    }
    calcHistograms(dst, features);
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the E5E5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsE5E5AllFiles(std::vector<char*> filenames) {
    // Apply E5E5 to all files
    std::string outfile2 ="laws_feature_E5E5.csv";
    char* outfile_cstr;
    filename_conversion(outfile2, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesE5E5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the E5W5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsE5W5AllFiles(std::vector<char*> filenames) {
    std::string outfile2 ="laws_feature_E5W5.csv";
    char* outfile_cstr;
    filename_conversion(outfile2, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesE5W5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the S5W5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsS5W5AllFiles(std::vector<char*> filenames) {
    // Apply S5W5 to all files
    std::string outfile2 ="laws_feature_S5W5.csv";
    char* outfile_cstr;
    filename_conversion(outfile2, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesS5W5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the W5R5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsW5R5AllFiles(std::vector<char*> filenames) {
    // Apply W5R5 to all files
    std::string outfile2 ="laws_feature_W5R5.csv";
    char* outfile_cstr;
    filename_conversion(outfile2, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesW5R5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the L5E5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsL5E5AllFiles(std::vector<char*> filenames) {
    // Apply L5E5 to all files
    std::string outfile ="laws_feature_L5E5.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesL5E5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function writes the feature vecotrs of all files after being applied the L5L5 laws filter to a CSV file
 Parameters:
 - char* file - name of the target file
 - std::vector<float> &features - to store the result feature vectors
 */
int calcLawsL5L5AllFiles(std::vector<char*> filenames) {
    // Apply L5L5 to all files
    std::string outfile ="laws_feature_L5L5.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcLawsFeaturesL5L5(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function applies the Laws filter to the target file and the files in the database, then uses those as feature vectors as long as the color histograms to compare the distances between two files using histogram intersections to find the top K matches.
 Parameters:
 - std::vector<char*> filenames - name of all files in database
 - char* targetfile - name of target file
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - to store the result files
 - int matching_method - distance metric we want to use
 */
int lawsFilterMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("laws_feature_L5E5.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcLawsL5E5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("laws_feature_S5W5.csv");
    if(!ifile) {
        calcLawsS5W5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("laws_feature_W5R5.csv");
    if(!ifile) {
        calcLawsW5R5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("laws_feature_L5L5.csv");
    if(!ifile) {
        calcLawsL5L5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("laws_feature_E5E5.csv");
    if(!ifile) {
        calcLawsE5E5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("laws_feature_E5W5.csv");
    if(!ifile) {
        calcLawsE5W5AllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("histogram_feature.csv");
    if(!ifile) {
        calcHistogramsAllFiles(filenames);
    }
    ifile.close();
    
    ifile.open("center_histogram_feature.csv");
    if(!ifile) {
        calcCenterAllFiles(filenames);
    }
    ifile.close();
    
    // calculate target file feature vector
    std::vector<float> target_featuresL5E5;
    calcLawsFeaturesL5E5(targetfile, target_featuresL5E5);
    
    std::vector<float> target_featuresS5W5;
    calcLawsFeaturesS5W5(targetfile, target_featuresS5W5);
    
    std::vector<float> target_featuresW5R5;
    calcLawsFeaturesW5R5(targetfile, target_featuresW5R5);
    
    std::vector<float> target_featuresL5L5;
    calcLawsFeaturesL5L5(targetfile, target_featuresL5L5);
    
    std::vector<float> target_featuresE5E5;
    calcLawsFeaturesE5E5(targetfile, target_featuresE5E5);
    
    std::vector<float> target_featuresE5W5;
    calcLawsFeaturesE5W5(targetfile, target_featuresE5W5);
    
    // calculate color feature for target
    std::vector<float> target_colors;
    calcHistograms(cv::imread(targetfile), target_colors);
    
    // calculate center histogram for target
    std::vector<float> target_centers;
    calcCenterHistogram(targetfile, target_centers);
    
    // calculate center color distances between target and all files
    std::vector<std::pair<float, char*>> center_distances;
    char* inputfile;
    filename_conversion("center_histogram_feature.csv", inputfile);
    std::vector<char *> allfilesCenters;
    std::vector<std::vector<float>> dataCenters;
    read_image_data_csv(inputfile, allfilesCenters, dataCenters, 0);
    calculateDistances(dataCenters, allfilesCenters, target_centers, center_distances, matching_method);
    
    // calculate Laws filter distances between target and all files
    std::vector<std::pair<float, char*>> distancesL5E5;
    filename_conversion("laws_feature_L5E5.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_featuresL5E5, distancesL5E5, matching_method);
    
    std::vector<std::pair<float, char*>> distancesL5L5;
    filename_conversion("laws_feature_L5L5.csv", inputfile);
    std::vector<char *> allfilesL5L5;
    std::vector<std::vector<float>> dataL5L5;
    read_image_data_csv(inputfile, allfilesL5L5, dataL5L5, 0);
    calculateDistances(dataL5L5, allfilesL5L5, target_featuresL5L5, distancesL5L5, matching_method);
    
    std::vector<std::pair<float, char*>> distancesS5W5;
    filename_conversion("laws_feature_S5W5.csv", inputfile);
    std::vector<char *> allfileS5W5;
    std::vector<std::vector<float>> dataS5W5;
    read_image_data_csv(inputfile, allfileS5W5, dataS5W5, 0);
    calculateDistances(dataS5W5, allfileS5W5, target_featuresS5W5, distancesS5W5, matching_method);
    
    std::vector<std::pair<float, char*>> distancesW5R5;
    filename_conversion("laws_feature_W5R5.csv", inputfile);
    std::vector<char *> allfileW5R5;
    std::vector<std::vector<float>> dataW5R5;
    read_image_data_csv(inputfile, allfileW5R5, dataW5R5, 0);
    calculateDistances(dataW5R5, allfileW5R5, target_featuresW5R5, distancesW5R5, matching_method);
    
    std::vector<std::pair<float, char*>> distancesE5E5;
    filename_conversion("laws_feature_E5E5.csv", inputfile);
    std::vector<char *> allfileE5E5;
    std::vector<std::vector<float>> dataE5E5;
    read_image_data_csv(inputfile, allfileE5E5, dataE5E5, 0);
    calculateDistances(dataE5E5, allfileE5E5, target_featuresE5E5, distancesE5E5, matching_method);
    
    std::vector<std::pair<float, char*>> distancesE5W5;
    filename_conversion("laws_feature_E5W5.csv", inputfile);
    std::vector<char *> allfileE5W5;
    std::vector<std::vector<float>> dataE5W5;
    read_image_data_csv(inputfile, allfileE5W5, dataE5W5, 0);
    calculateDistances(dataE5W5, allfileE5W5, target_featuresE5W5, distancesE5W5, matching_method);
    
    // calculate color distances between target and all files
    std::vector<std::pair<float, char*>> color_distances;
    filename_conversion("histogram_feature.csv", inputfile);
    std::vector<char *> allfilesColor;
    std::vector<std::vector<float>> dataColor;
    read_image_data_csv(inputfile, allfilesColor, dataColor, 0);
    calculateDistances(dataColor, allfilesColor, target_colors, color_distances, matching_method);
    
    std::vector<std::pair<float, char*>> distances;
    for (int i = 0; i < distancesL5E5.size(); i++) {
        distances.push_back(std::make_pair((0.14 * distancesE5E5.at(i).first + 0.07 * distancesL5L5.at(i).first + 0.22 * distancesL5E5.at(i).first  + 0.1 * distancesW5R5.at(i).first + 0.11 * distancesS5W5.at(i).first + 0.32 * center_distances.at(i).first + 0.04 * color_distances.at(i).first), distancesL5E5.at(i).second));
    }
    
    // find top k matches
    findTopK(distances, 0, k_neighbors, outputfiles);
    return 0;
}

/**
 This function does the baseline matching for target file
 Parameters:
 - std::vector<char*> filenames - the names of all the files we want to match
 - char* targetfile - name of the target file we want to find matches for
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - names of the k matches that we find
 - int matching_method - distance metric we want to use
 */
int baselineMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("baseline_feature.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcBaselineAllFiles(filenames);
    }
    ifile.close();
    
    // calculate target file feature vector
    std::vector<float> target_features;
    calcBaselineFeatures(targetfile, target_features);
    
    // calculate distances between target and all files
    std::vector<std::pair<float, char*>> distances;
    char* inputfile;
    filename_conversion("baseline_feature.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_features, distances, matching_method);
    
    // find top k matches
    findTopK(distances, 1, k_neighbors, outputfiles);
    return 0;
}

/**
 This function calculates the baseline feature vectors for all files and writes to a csv file
 Parameters:
 - std::vector<char*> filenames - names of all files
 */
int calcBaselineAllFiles(std::vector<char*> filenames) {
    std::string outfile ="baseline_feature.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcBaselineFeatures(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function calculates the baseline feature vector for a given file by taking the 9x9 pixels in the center of an image.
 Parameters:
 - char* file - the name of the file we want to calculate
 - std::vector<float> &features - to store the feature vector of the given file
 */
int calcBaselineFeatures(char* file, std::vector<float> &features) {
    cv::Mat src = cv::imread(file, cv::IMREAD_COLOR);
    int midHeight= (src.rows % 2 == 0) ? src.rows / 2 : src.rows / 2 + 1;
    int midWidth = (src.cols % 2 == 0) ? src.cols / 2 : src.cols / 2 + 1;
    for (int k = midHeight - 9 / 2 ; k < midHeight - 9 / 2 + 9; k++) {
        for (int j = midWidth - 9 / 2;  j < midWidth - 9 / 2 + 9; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(k, j);
            for (int l = 0; l < 3; l++) {
                features.push_back(pixel[l]);
            }
        }
    }
    return 0;
}

/**
 This function calculates the histogram feature vectors for all files and writes to a csv file
 Parameters:
 - std::vector<char*> filenames - names of all files
 */
int calcHistogramsAllFiles(std::vector<char*> filenames) {
    std::string outfile ="histogram_feature.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcHistograms(cv::imread(filename), features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function calculates the center color histograms for all files in the database.
 Parameters:
 - std::vector<char*> filenames - names of all files
 */
int calcCenterAllFiles(std::vector<char*> filenames) {
    std::string outfile ="center_histogram_feature.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        // get center pixels store in a Mat
        calcCenterHistogram(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function calculates the center color histograms for a given file.
 Parameters:
 - char* file - the name of the file we want to calculate
 - std::vector<float> &features - to store the feature vector of the given file
 */
int calcCenterHistogram(char* filename, std::vector<float> &features) {
    int sizes[] = { 300, 300 };
    cv::Mat dst(2, sizes, CV_8UC3);
    getCenterPixels(filename, dst);
    calcHistograms(dst, features);
    return 0;
}

/**
 This function gets the center pixels of an image and store the results in a cv::Mat.
 */
int getCenterPixels(char* filename, cv::Mat &dst) {
    cv::Mat src = cv::imread(filename);
    int midWidth = src.cols / 2;
    int midHeight = src.rows / 2;
    int width = 150;

    for (int i = midHeight - width; i < midHeight + width; i++) {
        for (int j = midWidth - width; j < midWidth + width; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(i - (midHeight - width), j - (midWidth - width))[k]
                = src.at<cv::Vec3b>(i, j)[k];
            }
            
        }
    }
    return 0;
}



/**
 This function takes in a file and calculate histogram for that file.
 Parameters:
 - char* file - name of the file
 - std::vector<float> &features - stores the feature vector for the file
 */
int calcHistograms(cv::Mat src, std::vector<float> &features) {
//    cv::Mat src = cv::imread(file);
    int num_bins = 8;
    int sizes[] = { num_bins, num_bins, num_bins };
    cv::Mat matrix(3, sizes, CV_32FC1);
    for (int i = 0; i <8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                matrix.at<float>(i, j, k) = 0;
            }
        }
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];
            int x = b * num_bins / 256;
            int y = g * num_bins / 256;
            int z = r * num_bins / 256;

            matrix.at<float>(x, y, z) = matrix.at<float>(x, y, z) + 1;
        }
    }
    int totalPixels = src.rows * src.cols;
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
            for (int k = 0; k < num_bins; k++) {
                float num = matrix.at<float>(i, j, k);
                features.push_back(num / totalPixels); 
            }
        }
    }
    return 0;
}

/**
 This function does the histogram matching for target file
 Parameters:
 - std::vector<char*> filenames - the names of all the files we want to match
 - char* targetfile - name of the target file we want to find matches for
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - names of the k matches that we find
 - int matching_method - distance metric we want to use
 */
int histogramMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("histogram_feature.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcHistogramsAllFiles(filenames);
    }
    ifile.close();
    
    // calculate features for target file
    std::vector<float> target_features;
    
    calcHistograms(cv::imread(target), target_features);
    
    // calculate distances between target and all files
    std::vector<std::pair<float, char*>> distances;
    char* inputfile;
    filename_conversion("histogram_feature.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_features, distances, matching_method);
    
    // find top k matches
    findTopK(distances, 0, k_neighbors, outputfiles);
    
    return 0;
}

/**
 This function calculates either the top or the bottom histograms for a given file, based on the position parameter passed in.
 Parameters:
 - char* filename - name of the file
 - int pos - which part we want to calculate. 0 - top, 1 - bottom
 - std::vector<float> &features - to store the feature vectors
 */
int calcTopOrBottomHistogram(char* filename, int pos, std::vector<float> &features) {
    cv::Mat src = cv::imread(filename);
    int mid = src.rows / 2;
    int num_bins = 8;
    int sizes[] = { num_bins, num_bins, num_bins };
    cv::Mat matrix(3, sizes, CV_32FC1);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                matrix.at<float>(i, j, k) = 0;
            }
        }
    }
    
    if (pos == 0) {
        for (int i = 0; i < mid; i++) {
            for (int j = 0; j < src.cols; j++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
                uchar b = pixel[0];
                uchar g = pixel[1];
                uchar r = pixel[2];
                int x = b * num_bins / 256;
                int y = g * num_bins / 256;
                int z = r * num_bins / 256;

                matrix.at<float>(x, y, z) = matrix.at<float>(x, y, z) + 1;
            }
        }
    } else {
        for (int i = mid; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
                uchar b = pixel[0];
                uchar g = pixel[1];
                uchar r = pixel[2];
                int x = b * num_bins / 256;
                int y = g * num_bins / 256;
                int z = r * num_bins / 256;

                matrix.at<float>(x, y, z) = matrix.at<float>(x, y, z) + 1;
            }
        }
    }
    int totalPixels = src.rows * src.cols / 2;
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
            for (int k = 0; k < num_bins; k++) {
                float num = matrix.at<float>(i, j, k);
                features.push_back(num / totalPixels);
            }
        }
    }
    return 0;
}

/**
 This function calculates the multi histogram feature vectors for all files and writes to a csv file
 Parameters:
 - std::vector<char*> filenames - names of all files
 */
int calcMultiHistogramsAllFiles(std::vector<char*> filenames) {
    // write top half and bottom half to different CSV files
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features_top;
        calcTopOrBottomHistogram(filename, 0, features_top);
        std::vector<float> features_bottom;
        calcTopOrBottomHistogram(filename, 1, features_bottom);
        char* top_outfile_cstr;
        filename_conversion("top_hist_features.csv", top_outfile_cstr);
        append_image_data_csv(top_outfile_cstr, filename, features_top, 0);
        char* bottom_outfile_cstr;
        filename_conversion("bottom_hist_features.csv", bottom_outfile_cstr);
        append_image_data_csv(bottom_outfile_cstr, filename, features_bottom, 0);
    }
    return 0;
}

/**
 This function calculates the HSV feature vectors for all files in the database.
 */
int calcHSVAllFiles(std::vector<char*> filenames) {
    char* outfile_cstr;
    filename_conversion("HSV_feature.csv", outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcHSV(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function calculates the HSV feature vector for a given file.
 */
int calcHSV(char* filename, std::vector<float> &features) {
    cv::Mat src = cv::imread(filename);
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    calcHistograms(hsv, features);
    return 0;
}

/**
 This function uses HSV values as features to find the top k matches for a given target file.
 Parameters:
 - std::vector<char*> filenames - the names of all the files we want to match
 - char* targetfile - name of the target file we want to find matches for
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - names of the k matches that we find
 - int matching_method - the distance metric we want to use.
 */
int HSVMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
        // check if file exists
        std::ifstream ifile;
        ifile.open("HSV_feature.csv");
        if(!ifile) {
            // if file doesn't exist, calculate feature vectors and write to csv
            calcHSVAllFiles(filenames);
        }
        ifile.close();
        
        // calculate features for target file
        std::vector<float> target_features;
        calcHSV(target, target_features);
        
        // calculate distances between target and all files
        std::vector<std::pair<float, char*>> distances;
        char* inputfile;
        filename_conversion("HSV_feature.csv", inputfile);
        std::vector<char *> allfiles;
        std::vector<std::vector<float>> data;
        read_image_data_csv(inputfile, allfiles, data, 0);
        calculateDistances(data, allfiles, target_features, distances, matching_method);
        
        // find top k matches
        findTopK(distances, 0, k_neighbors, outputfiles);
        
        return 0;
}

/**
 This function takes in a file, applies the gabor filter, and calculate gabor filter feature vector histogram for that file.
 Parameters:
 - char* file - name of the file
 - std::vector<float> &features - stores the feature vector for the file
 */
int calcGaborFeatures(char* file, std::vector<float> &features) {
    cv::Mat src = imread(file, cv::IMREAD_GRAYSCALE);
    cv::Size ksize(21, 21);
    double sigma = 10;
    double theta = 0;
    double lambda = 10;
    double gamma = 0.5;
    cv::Mat gaborKernel = getGaborKernel(ksize, sigma, theta, lambda, gamma);
    cv::Mat dst;
    filter2D(src, dst, -1, gaborKernel);
    calcHistograms(dst, features);
    return 0;
}

/**
 This function calculates the gabor filter feature vectors for all files and writes to a csv file
 Parameters:
 - std::vector<char*> filenames - names of all files
 */
int calcGaborAllFiles(std::vector<char*> filenames) {
    std::string outfile ="gabor_feature.csv";
    char* outfile_cstr;
    filename_conversion(outfile, outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcGaborFeatures(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}

/**
 This function uses gabor filter and color histograms, and center color histograms as features to find the top k matches for a given target file.
 Parameters:
 - std::vector<char*> filenames - the names of all the files we want to match
 - char* targetfile - name of the target file we want to find matches for
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - names of the k matches that we find
 - int matching_method - the distance metric we want to use.
 */
int gaborFilterMatching(std::vector<char*> filenames, char* target, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("gabor_feature.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcGaborAllFiles(filenames);
    }
    ifile.close();
    
    // calculate features for target file
    std::vector<float> target_features;
    calcGaborFeatures(target, target_features);
    
    // calculate color feature for target
    std::vector<float> target_colors;
    calcHistograms(cv::imread(target), target_colors);
    
    // calculate center histogram for target
    std::vector<float> target_centers;
    calcCenterHistogram(target, target_centers);
    
    // calculate center color distances between target and all files
    std::vector<std::pair<float, char*>> center_distances;
    char* inputfile;
    filename_conversion("center_histogram_feature.csv", inputfile);
    std::vector<char *> allfilesCenters;
    std::vector<std::vector<float>> dataCenters;
    read_image_data_csv(inputfile, allfilesCenters, dataCenters, 0);
    calculateDistances(dataCenters, allfilesCenters, target_centers, center_distances, matching_method);
    
    // calculate distances between target and all files
    std::vector<std::pair<float, char*>> distances;
    filename_conversion("gabor_feature.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_features, distances, matching_method);
    
    // calculate color distances between target and all files
    std::vector<std::pair<float, char*>> color_distances;
    filename_conversion("histogram_feature.csv", inputfile);
    std::vector<char *> allfilesColor;
    std::vector<std::vector<float>> dataColor;
    read_image_data_csv(inputfile, allfilesColor, dataColor, 0);
    calculateDistances(dataColor, allfilesColor, target_colors, color_distances, matching_method);
    
    std::vector<std::pair<float, char*>> avg_distances;
    for (int i = 0; i < color_distances.size(); i++) {
        avg_distances.push_back(std::make_pair((0.4 * center_distances.at(i).first + 0.3 * color_distances.at(i).first + 0.3 * distances.at(i).first) , color_distances.at(i).second));
    }
    
    // find top k matches
    findTopK(avg_distances, 0, k_neighbors, outputfiles);
    
    return 0;
}

/**
 This function uses multi histogram matching to find the top k matches for the target file.
 Parameters:
 - std::vector<char*> filenames - the file names of all files
 - char* targetfile - the name of the target file
 - int k_neighbors - the number of matches we want
 - std::vector<char*> &outputfiles - to store the file names of the matches we find
 */
int multiHistogramMatching(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("top_hist_features.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcMultiHistogramsAllFiles(filenames);
    }
    ifile.close();
    
    // calculate top and bottom half histograms for target file
    std::vector<float> target_top;
    std::vector<float> target_bottom;
    calcTopOrBottomHistogram(targetfile, 0, target_top);
    calcTopOrBottomHistogram(targetfile, 1, target_bottom);
    
    // calculate distances between target and all files
    // calculate the intersections between the top histograms
    std::vector<std::pair<float, char*>> top_distances;
    char* inputfile;
    filename_conversion("top_hist_features.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_top, top_distances, matching_method);
    
    // calculate the intersections between the bottom histograms
    std::vector<std::pair<float, char*>> bottom_distances;
    char* bottom_inputfile;
    filename_conversion("bottom_hist_features.csv", bottom_inputfile);
    std::vector<char *> allInfiles;
    std::vector<std::vector<float>> bottom_data;
    read_image_data_csv(bottom_inputfile, allInfiles, bottom_data, 0);
    calculateDistances(bottom_data, allInfiles, target_bottom, bottom_distances, matching_method);
    
    // calculate the average distances between top and bottom histograms
    std::vector<std::pair<float, char*>> avg_distances;
    for (int i = 0; i < top_distances.size(); i++) {
        avg_distances.push_back(std::make_pair((top_distances.at(i).first + bottom_distances.at(i).first) / 2, allInfiles.at(i)));
    }
    
    // find top k matches
    findTopK(avg_distances, 0, k_neighbors, outputfiles);

    return 0;
}

/**
 This function calculates the histogram intersection of two files.
 Parameters:
 - std::vector<float> file - feature vector of the file
 - std::vector<float> target - feature vector of the target file
 - float &intersection - the intersection between two histograms
 */
int histogramIntersection(std::vector<float> file, std::vector<float> target, float &intersection) {
    intersection = 0.0f;
    for (int i = 0; i < file.size(); i++) {
        intersection += fmin(file.at(i), target.at(i));
    }
    return 0;
}

/**
 This function applies the sobel X filter for a given image.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    
    int horizontal_filter[] = {-1, 0, 1};
    int vertical_filter[] = {1, 2, 1};
    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3s>(i, j)[k] = (
                                              src.at<cv::Vec3b>(i, j - 1)[k] * horizontal_filter[0]
                                              + src.at<cv::Vec3b>(i, j)[k] * horizontal_filter[1]
                                              + src.at<cv::Vec3b>(i, j + 1)[k] * horizontal_filter[2]) / 4;
            }
        }
    }
    
    cv::Mat temp;
    dst.copyTo(temp);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3s>(i, j)[k] = (
                                              temp.at<cv::Vec3s>(i - 1, j)[k] * vertical_filter[0]
                                              + temp.at<cv::Vec3s>(i, j)[k] * vertical_filter[1]
                                              + temp.at<cv::Vec3s>(i + 1, j)[k] * vertical_filter[2]);
            }
        }
    }
    return 0;
}

/**
 This function applies the sobel Y filter for a given image.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    int horizontal_filter[] = {1, 2, 1};
    int vertical_filter[] = {-1, 0, 1};
    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3s>(i, j)[k] = (int)(
                                              src.at<cv::Vec3b>(i, j - 1)[k] * horizontal_filter[0]
                                              + src.at<cv::Vec3b>(i, j)[k] * horizontal_filter[1]
                                              + src.at<cv::Vec3b>(i, j + 1)[k] * horizontal_filter[2]) / 4;
            }
        }
    }
    cv::Mat temp;
    dst.copyTo(temp);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3s>(i, j)[k] = (int)(
                                              temp.at<cv::Vec3s>(i - 1, j)[k] * vertical_filter[0]
                                              + temp.at<cv::Vec3s>(i, j)[k] * vertical_filter[1]
                                              + temp.at<cv::Vec3s>(i + 1, j)[k] * vertical_filter[2]) ;
            }
        }
    }
    return 0;
}

/**
 This function computes the gradient magnitudes for a given image.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    dst = cv::Mat::zeros(sx.size(), CV_8UC3);
    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            cv::Vec3s pixelX = sx.at<cv::Vec3s>(i, j);
            cv::Vec3s pixelY = sy.at<cv::Vec3s>(i, j);
            dst.at<cv::Vec3b>(i, j)[0] = sqrt(pixelX[0] * pixelX[0] + pixelY[0] * pixelY[0]);
            dst.at<cv::Vec3b>(i, j)[1] = sqrt(pixelX[1] * pixelX[1] + pixelY[1] * pixelY[1]);
            dst.at<cv::Vec3b>(i, j)[2] = sqrt(pixelX[2] * pixelX[2] + pixelY[2] * pixelY[2]);
        }
    }
    return 0;
}

/**
 This function calculates the texture vector for a given file.
 Parameters:
 - char* file - name of the file
 - std::vector<float> &features - stores the feature vector for the file
 */
int calcTexture(char* filename, std::vector<float> &features) {
    // calculate magnitude for file
    cv::Mat image = cv::imread(filename);
    cv::Mat sobelX;
    sobelX3x3(image, sobelX);
    cv::Mat sobelY;
    sobelY3x3(image, sobelY);
    cv::Mat magnitude_dst;
    magnitude(sobelX, sobelY, magnitude_dst);
    // build histogram of magnitudes
    int num_bins = 8;
    int sizes[] = { num_bins, num_bins, num_bins };
    cv::Mat matrix(3, sizes, CV_32FC1);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                matrix.at<float>(i, j, k) = 0;
            }
        }
    }
    for (int i = 0; i < magnitude_dst.rows; i++) {
        for (int j = 0; j < magnitude_dst.cols; j++) {
            cv::Vec3b pixel = magnitude_dst.at<cv::Vec3b>(i, j);
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];
            int x = b * num_bins / 256;
            int y = g * num_bins / 256;
            int z = r * num_bins / 256;

            matrix.at<float>(x, y, z) = matrix.at<float>(x, y, z) + 1;
        }
    }
    int totalPixels = image.rows * image.cols;
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
            for (int k = 0; k < num_bins; k++) {
                float num = matrix.at<float>(i, j, k);
                features.push_back(num / totalPixels);
            }
        }
    }
    return 0;
}

/**
 This function calculates the texture feature vectors for all files in the database.
 */
int calcTextureAllFiles(std::vector<char*> filenames) {
    char* outfile_cstr;
    filename_conversion("texture.csv", outfile_cstr);
    for (int i = 0; i < filenames.size(); i++) {
        char* filename = filenames.at(i);
        std::vector<float> features;
        calcTexture(filename, features);
        append_image_data_csv(outfile_cstr, filename, features, 0);
    }
    return 0;
}
/**
 This function uses the texture and color features to find the top k matches for a given target file.
 Parameters:
 - std::vector<char*> filenames - the names of all the files we want to match
 - char* targetfile - name of the target file we want to find matches for
 - int k_neighbors - number of matches we want
 - std::vector<char*> &outputfiles - names of the k matches that we find
 - int matching_method - the distance metric we want to use.
 */
int textureAndColor(std::vector<char*> filenames, char* targetfile, int k_neighbors, std::vector<char*> &outputfiles, int matching_method) {
    // check if file exists
    std::ifstream ifile;
    ifile.open("texture.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcTextureAllFiles(filenames);
    }
    ifile.close();

    ifile.open("histogram_feature.csv");
    if(!ifile) {
        // if file doesn't exist, calculate feature vectors and write to csv
        calcHistogramsAllFiles(filenames);
    }
    ifile.close();
    
    // calculate texture features for target file
    std::vector<float> target_features;
    calcTexture(targetfile, target_features);
    
    // calculate color feature for target
    std::vector<float> target_colors;
    calcHistograms(cv::imread(targetfile), target_colors);
    
    // calculate color distances between target and all files
    std::vector<std::pair<float, char*>> distances;
    char* inputfile;
    filename_conversion("histogram_feature.csv", inputfile);
    std::vector<char *> allfiles;
    std::vector<std::vector<float>> data;
    read_image_data_csv(inputfile, allfiles, data, 0);
    calculateDistances(data, allfiles, target_colors, distances, matching_method);
    
    // calculate texture distances
    std::vector<std::pair<float, char*>> texture_distances;
    char* texture_inputfile;
    filename_conversion("texture.csv", texture_inputfile);
    std::vector<char *> allInfiles;
    std::vector<std::vector<float>> texture_data;
    read_image_data_csv(texture_inputfile, allInfiles, texture_data, 0);
    calculateDistances(texture_data, allInfiles, target_features, texture_distances, matching_method);
    
    std::vector<std::pair<float, char*>> avg_distances;
    for (int i = 0; i < texture_distances.size(); i++) {
        avg_distances.push_back(std::make_pair((texture_distances.at(i).first + distances.at(i).first) / 2, texture_distances.at(i).second));
    }
    
    // find top k matches
    findTopK(avg_distances, 0, k_neighbors, outputfiles);

    return 0;
}


/**
 This function finds the top k matches of the input image.
 Parameters:
 - std::pair<float, char*>> distances - stores the result of the distances between each image and target image
 - int ascending - 0: sort in descending order, 1: sort in ascending order
 - int k - number of matches we want
 - std::vector<char*> &topK - store the names of the result files
 */
int findTopK(std::vector<std::pair<float, char*>> distances, int ascending, int k,  std::vector<char*> &topK) {
    if (ascending == 0) {
        std::sort(distances.begin(), distances.end(), std::greater <>());
    } else {
        std::sort(distances.begin(), distances.end());
    }
    for (int i = 1; i < k + 1; i++) {
        topK.push_back(distances.at(i).second);
    }
    return 0;
}

/**
 This function calculates the distances between all feature vectors and target file feature vector based on the matching method.
 Parameters:
 - std::vector<std::vector<float>> data - all the feature vectors
 - std::vector<char *> filenames - names of all the files
 - std::vector<float> target_features - feature vector for target file
 - std::vector<std::pair<float, char*>> &distances - to store the distances that we calculate for each file
 - int matchingMethod - the matching method we use, will affect how we calculate distances
 */
int calculateDistances(std::vector<std::vector<float>> data, std::vector<char *> filenames, std::vector<float> target_features, std::vector<std::pair<float, char*>> &distances, int matchingMethod) {
    for (int i = 0; i < data.size(); i++) {
        std::vector<float> feature_vector = data.at(i);
        if (matchingMethod == 0) {
            // baseline matching, calculate sum of squared differences
            float diff = 0.0f;
            for (int j = 0; j < feature_vector.size(); j++) {
                diff += (feature_vector.at(j) - target_features.at(j)) * (feature_vector.at(j) - target_features.at(j));
            }
            std::pair<float, char*> pair = std::make_pair(diff, filenames.at(i));
            distances.push_back(pair);
        } else if (matchingMethod == 1) {
            // histogram matching, calculate histogram intersections
            float intersection = 0.0f;
            histogramIntersection(feature_vector, target_features, intersection);
            distances.push_back(std::make_pair(intersection, filenames.at(i)));
        } else if (matchingMethod == 2) {
            // L1 distance
            float diff = 0.0f;
            for (int j = 0; j < feature_vector.size(); j++) {
                diff += abs(feature_vector.at(j) - target_features.at(j));
            }
            std::pair<float, char*> pair = std::make_pair(diff, filenames.at(i));
            distances.push_back(pair);
        }
    }
    return 0;
}

/**
 This function converts a string to a char*.
 */
int filename_conversion(std::string orig_file, char* &converted) {
    converted = new char[orig_file.length()];
    strcpy(converted, orig_file.c_str());
    return 0;
}

