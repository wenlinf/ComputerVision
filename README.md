#  CS 5330 Project 2
## Wenlin Fang

* I used MacOS and XCode to run and compile my code.
* To run the program, users need to run the program with commands that contain 5 arguments, including: 
    * directory that contains all image files. e.g. /Users/wenlin/Downloads/olympus 
    * path to the target file. e.g. /Users/wenlin/Downloads/olympus/pic.0535.jpg
    * number of matches you want. e.g. 3
    * feature type that you want to use.
        * 0 - baseline matching
        * 1 - histogram matching
        * 2 - multi-histogram matching
        * 3 - texture and color matching
        * 4 - custom matching
        * 5 - laws filter matching
        * 6 - HSV matching
        * 7 - gabor filter matching
    * distance metric that you want to use:
        * 0 - sum of squared errors
        * 1 - histogram intersection
        * 2 - L1 distance
    * if the program did not receive enough number of command line arguments, it will not run.
    * user can press the 'q' key to quit the program.
* All files in the project: 
    * The main.cpp file is the one that contains the main function. 
    * The imgMatching.cpp and imgMatching.hpp contain the matching functions of this program.
    * The csv_util.cpp and csv_util.hpp files are provided by professor Maxwell to read from and write to csv files.
* I did not use time travel days for this project.
