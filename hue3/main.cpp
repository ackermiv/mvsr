#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
//#include <Eigen/Dense>

using namespace cv;

void saveSiftFeatures(Mat image)
{
    // Create SIFT feature detector object
    Ptr<Feature2D> detector = SIFT::create();

    // Detect SIFT keypoints
    std::vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);

    // Save keypoints to file
    std::ofstream outfile;
    outfile.open("data/keypoints.csv");

    for (int i = 0; i < keypoints.size(); i++)
    {
        outfile << keypoints[i].pt.x << "," << keypoints[i].pt.y << "," << keypoints[i].size << "," << keypoints[i].angle << std::endl;
    }
    outfile.close();
}

void saveActiveSet(Mat image)
{
    // Create SIFT feature detector object
    Ptr<Feature2D> detector = SIFT::create();

    // Detect SIFT keypoints
    std::vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);

    // Create active set
    std::ofstream outfile;
    outfile.open("data/activeSet.csv");

    for (int i = 0; i < keypoints.size(); i++)
    {
        outfile << i << std::endl;
    }
    outfile.close();
}

void saveActiveSetXYZ(Mat image)
{
    // Create SIFT feature detector object
    Ptr<Feature2D> detector = SIFT::create();

    // Detect SIFT keypoints
    std::vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);

    // Create active set
    std::ofstream outfile;
    outfile.open("data/activeSet_XYZ.csv");

    // Measure x, y and z at each point
    for (int i = 0; i < keypoints.size(); i++)
    {
        // Measure x, y and z
        double x = keypoints[i].pt.x;
        double y = keypoints[i].pt.y;
        double z = 0;

        outfile << x << "," << y << "," << z << std::endl;
    }
    outfile.close();
}

Mat findPoseEstimation(Mat image)
{
    // Create SIFT feature detector object
    Ptr<Feature2D> detector = SIFT::create();

    // Read in active set
    std::ifstream infile;
    infile.open("data/activeSet.csv");
    std::vector<int> activeSet;
    int id;
    while (infile >> id) {
        activeSet.push_back(id);
    //std::cout << id << std::endl;
    }
    infile.close();

    // Read in active set XYZ
    std::ifstream infile2;
    infile2.open("data/activeSet_XYZ.csv");
    std::vector<Point3d> activeSet_XYZ;
 
    std::string line;
    
    while (getline(infile2, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        Point3d row;
        getline(lineStream, cell, ','); 
        row.x = stod(cell);
        getline(lineStream, cell, ',');
        row.y = stod(cell);
        getline(lineStream, cell, ',');
        row.z = stod(cell);
        std::cout << row << std::endl;
        activeSet_XYZ.push_back(row);
    }

    infile2.close();

    // Detect SIFT keypoints
    std::vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);

    // Create active set keypoints
    std::vector<KeyPoint> activeSet_Keypoints;
    for (int i = 0; i < activeSet.size(); i++)
    {
        activeSet_Keypoints.push_back(keypoints[activeSet[i]]);
    }
    //std::cout << activeSet_Keypoints.size() << std::endl;

    // Compute descriptors
    Mat descriptors;
    detector->compute(image, activeSet_Keypoints, descriptors);

    // Read in train image
    Mat trainImage = imread("data/trainImage.png");
    std::vector<KeyPoint> trainImage_Keypoints;
    detector->detect(trainImage, trainImage_Keypoints);
    Mat trainImage_Descriptors;
    detector->compute(trainImage, trainImage_Keypoints, trainImage_Descriptors);
    //std::cout << keypoints.size()<< std::endl;

    // Match descriptors
    BFMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors, trainImage_Descriptors, matches);
    //std::cout << matches.size()<< std::endl;
    
    // Create object points
    std::vector<Point3d> objectPoints;
    for (int i = 0; i < matches.size(); i++)
    {
        objectPoints.push_back(activeSet_XYZ[matches[i].queryIdx]);
    }

    // Create image points
    std::vector<Point2f> imagePoints;
    for (int i = 0; i < matches.size(); i++)
    {
        imagePoints.push_back(trainImage_Keypoints[matches[i].trainIdx].pt);
    }

    // Camera matrix
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    // Distortion coefficients
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

    // Undistort image
    Mat undistortedImage;
    undistort(image, undistortedImage, cameraMatrix, distCoeffs);

    // Compute poses
    Mat rvec, tvec;
    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    // Print translation
    std::cout << "Translation: " << tvec << std::endl;

    return tvec;/**/
    //return Mat::ones(3,4,5);
}

int main()
{
    // Load image
    Mat image = imread("data/image.png");

    // Save SIFT features
    //saveSiftFeatures(image);

    // Save active set
    //saveActiveSet(image);

    // Save active set XYZ
    //saveActiveSetXYZ(image);

    // Compute pose estimation
    Mat tvec = findPoseEstimation(image);

    return 0;
}