#include "hog.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <math.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <cfloat>
namespace fs = std::filesystem;


int main() 
{

    size_t blocksize = 64;
    size_t cellsize = 32;
    size_t stride = 64;
    size_t binning = 9;
    int count = -1;
    int count_test = -1;
    cv::Mat training(2400, 3168, CV_32FC1);
    cv::Mat label_train(2400, 1, CV_32FC1);
    cv::Mat testing(600, 3168, CV_32FC1);
    cv::Mat label_test(600, 1, CV_32FC1);
    std::string path = "/home/meowmoewrainbow/hog1/training";
    std::string name = "/home/meowmoewrainbow/hog1/training/";
    std::string path_test = "/home/meowmoewrainbow/hog1/testing";
    std::string name_test = "/home/meowmoewrainbow/hog1/testing/";
    for (const auto & entry : fs::directory_iterator(path))
    {
        std::cout << entry.path().filename() << std::endl;
        std::string file_name = entry.path().filename();
        std::string name_save = name + file_name;
        for (const auto & f : fs::directory_iterator(name_save))
        {
            count++;
            std::cout << f.path() << std::endl;
            cv::Mat image = cv::imread(f.path(), CV_8U);
            cv::resize(image, image, cv::Size(720, 570));
            HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED);
            hog.process(image);
            auto hist = hog.retrieve(cv::Rect(0,0,image.cols, image.rows));
            std::cout << "Histogram size: " << hist.size() << "\n";
            for (int i = 0; i < hist.size(); i++)
            {
                training.at<float>(count, i) = hist.at(i);
            }
            if (file_name == "polyps")
            {
                label_train.at<float>(count, 0) = 0;
            }
            else if (file_name == "normal-cecum")
            {
                label_train.at<float>(count, 0) = 1;
            }
            else if (file_name == "ulcerative-colitis")
            {
                label_train.at<float>(count, 0) = 2;
            }
            else if (file_name == "esophagitis")
            {   
                label_train.at<float>(count, 0) = 3;
            }
            else if (file_name == "normal-pylorus")
            {
                label_train.at<float>(count, 0) = 4;
            }
            else if (file_name == "normal-z-line")
            {
                label_train.at<float>(count, 0) = 5;
            }
            std::cout << "\n";
        }
    }
    std::cout << count << "\n";
    //std::cout << "label = " << std::endl << " "  << label_train << std::endl << std::endl;
    //std::cout << "training = " << std::endl << " "  << training << std::endl << std::endl;
    for (const auto & entry_test : fs::directory_iterator(path_test))
    {
        std::cout << entry_test.path().filename() << std::endl;
        std::string file_name_test = entry_test.path().filename();
        std::string name_save_test = name_test + file_name_test;
        for (const auto & f_test : fs::directory_iterator(name_save_test))
        {
            count_test++;
            std::cout << f_test.path() << std::endl;
            cv::Mat image_test = cv::imread(f_test.path(), CV_8U);
            cv::resize(image_test, image_test, cv::Size(720, 570));
            HOG hog_test(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED);
            hog_test.process(image_test);
            auto hist_test = hog_test.retrieve(cv::Rect(0,0,image_test.cols, image_test.rows));
            std::cout << "Histogram size: " << hist_test.size() << "\n";
            for (int j = 0; j < hist_test.size(); j++)
            {
                testing.at<float>(count_test, j) = hist_test.at(j);
            }
            if (file_name_test == "polyps")
            {
                label_test.at<float>(count_test, 0) = 0;
            }
            else if (file_name_test == "normal-cecum")
            {
                label_test.at<float>(count_test, 0) = 1;
            }
            else if (file_name_test == "ulcerative-colitis")
            {
                label_test.at<float>(count_test, 0) = 2;
            }
            else if (file_name_test == "esophagitis")
            {   
                label_test.at<float>(count_test, 0) = 3;
            }
            else if (file_name_test == "normal-pylorus")
            {
                label_test.at<float>(count_test, 0) = 4;
            }
            else if (file_name_test == "normal-z-line")
            {
                label_test.at<float>(count_test, 0) = 5;
            }
            std::cout << "\n";
        }
    }
    //std::cout << "test = " << std::endl << " "  << testing << std::endl << std::endl;
    std::cout << count_test << "\n";
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    std::cout << "Training..." << "\n";
    knn->train(training, cv::ml::ROW_SAMPLE, label_train);
    cv::Mat test_sample;
    int correct_class = 0;
    cv::Mat result;
    cv::Mat nei;
    std::cout << "Testing..." << "\n";
    for (int tsample = 0; tsample < testing.rows; tsample++)
    {
        test_sample = testing.row(tsample);
        knn->findNearest(test_sample, 7, result, nei);
        //std::cout << "result = " << std::endl << " "  << result << std::endl << std::endl;
        if (std::fabs(result.at<float>(0) - label_test.at<float>(tsample, 0)) < FLT_EPSILON)
        {
            correct_class++;
        }
    }
    double accuracy = correct_class * 100 / testing.rows;
    //std::cout << testing.rows << "\n";
    std::cout << "Accuracy = " << accuracy << "%" << "\n";
    return 0;

}