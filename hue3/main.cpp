#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
//#include <Eigen/Dense>

using namespace cv;


void drawImageKeypoints(Mat &image, std::vector<KeyPoint> &keypoints)
{
    // Display SIFT features on image
    // Draw the keypoints
	Mat output;
	drawKeypoints(image, keypoints, output);

	// Display the image
	imshow("SIFT Features", output);
	waitKey(0);
}


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
    drawImageKeypoints(image, keypoints);
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

Mat findPoseEstimation(Mat &image)
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
        //std::cout << row << std::endl;
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


    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("data/webcam.webm"); 
        
    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
    }
    
    while(1){
        
        Mat trainImage; //trainImage and its derivates refer to each frame of the computed video. 
        // Capture frame-by-frame
        cap >> trainImage;
        // Read in train image
        //Mat trainImage = imread("data/trainImage3.png");
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
        //Mat cameraMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

        // Distortion coefficients
        Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

        // Undistort image
        Mat undistortedImage;
        undistort(image, undistortedImage, cameraMatrix, distCoeffs);

        // Compute poses
        Mat rvec, tvec;
        solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

        // Print translation/rotation
        std::cout << "Translation: " << tvec << std::endl;
        std::cout << "Rotation: " << rvec << std::endl;
        // Display the resulting frame
        //drawImageKeypoints(trainImage, trainImage_Keypoints);
        Mat drawImg;
        std::vector< char > matches_mask;
        drawMatches(image,keypoints,trainImage,trainImage_Keypoints,matches,drawImg, 0, 0,matches_mask,0);
        imshow("frame",drawImg);

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
        break;
    }
    //return tvec;/**/
    return Mat::ones(3,4,5);
}


//code from https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
void videoread(Mat &image)
    {
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("data/webcam.webm"); 
        
    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
    }
    
    while(1){
    
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
    
        // If the frame is empty, break immediately
        if (frame.empty())
        break;

        //does the magic
        //Mat tvec = findPoseEstimation(image);

        // Display the resulting frame
        imshow( "Frame", frame );
    
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
        break;
    }
    
    // When everything done, release the video capture object
    cap.release();
    
    // Closes all the frames
    destroyAllWindows();
}


int main()
{
    // Load image
    Mat image = imread("data/image5.jpg");

    // Save SIFT features
    saveSiftFeatures(image);

    // Save active set
    saveActiveSet(image);

    // Save active set XYZ
    saveActiveSetXYZ(image);

    // Compute pose estimation
    findPoseEstimation(image);
    //videoread(image);
    return 0;
}