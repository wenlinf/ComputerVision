/*
 CS 5330 Computer Vision
 Project 2
 Wenlin Fang
 
 Adpated from Prof Bruce A. Maxwell sample code
*/
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include "imgMatching.hpp"

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
    char dirname[256];
    char buffer[256];
    char* targetfile = new char[256];
    int k_neighbors;
    int matching_method;
    int feature_type;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // check for sufficient arguments
    if( argc < 6) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // get the directory path
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname );

    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    strcpy(targetfile, argv[2]);
    k_neighbors = std::stoi(argv[3]);

    std::vector<char*> filenames;
    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
    strstr(dp->d_name, ".png") ||
    strstr(dp->d_name, ".ppm") ||
    strstr(dp->d_name, ".tif") ) {

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
  
      char* filename = new char[256];
      std::string filename_str = std::string(buffer);
      strcpy(filename, filename_str.c_str());
      filenames.push_back(filename);
    }
   }
    std::vector<char*> outputfiles;
    feature_type = std::stoi(argv[4]);
    matching_method = std::stoi(argv[5]);
    findKNeighbors(filenames, targetfile, k_neighbors, feature_type, matching_method, outputfiles);
    
    std::vector<cv::Mat> topKMatrices;
    topKMatrices.push_back(cv::imread(targetfile));
    for (int i = 0; i < outputfiles.size(); i++) {
        std::cout << outputfiles.at(i) << std::endl;
        cv::Mat img = cv::imread(outputfiles.at(i));
        topKMatrices.push_back(img);
    }
    
    // display top matches in a window
    cv::Mat out;
    cv::hconcat(topKMatrices, out);
    cv::imshow("Top matches", out);
    // quit the program if user presses 'q'
    while (true) {
            char key = cv::waitKey(10);
            if (key == 'q') {
                break;
            }
        }

  printf("Terminating\n");

  return(0);
}


