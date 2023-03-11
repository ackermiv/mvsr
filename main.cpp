#include <iostream>           //Basic I/O Operations
#include "opencv2/opencv.hpp" //Include all OpenCV header

  
using namespace std;





//Class template
class discreteConvolution
{   
private:
    //private members
    cv::Mat kernel_;

public:
    //public members
    
    //default constructor
    discreteConvolution();
    discreteConvolution(cv::Mat);
    //deconstructor
    ~discreteConvolution();
    
    //public member functions
    void print();
    cv::Mat conv(cv::Mat);
};

//default constructor
discreteConvolution::discreteConvolution()
{
    //initialize private members
    kernel_ = cv::Mat::zeros(3,3, CV_64F);
}

//angabe constructor
discreteConvolution::discreteConvolution(cv::Mat kernel)
{
    //initialize private members
    kernel_ = kernel;
}

//deconstructor
discreteConvolution::~discreteConvolution()
{
    //clean up any resources
}

//public member functions definitions
void discreteConvolution::print()           //for debugging
{
    //function implementation
    std::cout << kernel_ << endl;
}

cv::Mat discreteConvolution::conv(cv::Mat image)
{
    image.convertTo(image, CV_64FC1);
    double min, max; 
    cv::Mat convimage;
    convimage = cv::Mat::zeros(image.rows-kernel_.rows+1,image.cols-kernel_.cols+1, CV_64FC1);
    double sum = 0;
        for(int i = 0; i < image.rows-kernel_.rows+1; i++)
        {
            for(int j = 0; j < image.cols-kernel_.cols+1; j++)
            {
                sum = 0;                                                                        //build sum over dot product of kernel and the relevant section of image
                for (int k = 0; k < kernel_.rows; k++)
                {
                    for (int l = 0; l < kernel_.cols; l++)
                    {
                        sum += image.at<double>(i+k,j+l)*kernel_.at<double>(k,l); 
                    }
                }
                convimage.at<double>(i,j)=sum;
                //min=std::min(min,sum);
                //max=std::max(max,sum);
            }
        }

    /*    cv::minMaxIdx (convimage, &min, &max,NULL);
    cout  << "min " << min << "  max " << max <<endl;
    convimage = convimage-min;
    convimage = (convimage/(max-min))*255;
    convimage.convertTo(convimage, CV_8UC1);
    return convimage; */


    //convimage = convimage-min;
    //convimage = (convimage/(max-min))*255;
    //convimage.convertTo(convimage, CV_8UC1);
    return convimage;
}

class sobelDetector
{
private:
        double m[3][3] = {{1., 2., 1.}, {0., 0., 0.}, {-1., -2., -1.}};     //explicit definition of double array
        cv::Mat kernel_ = cv::Mat(3, 3, CV_64F, m);                         //feed Mat with the double array 
        discreteConvolution sobelx_ = discreteConvolution(kernel_);
        discreteConvolution sobely_ = discreteConvolution(kernel_.t());

public:
    sobelDetector(){}
    
    
    void getEdges(cv::Mat &imgsobel, cv::Mat img) 
    {
        if (img.type()!=CV_64F) img.convertTo(img, CV_64F);                  //check if img is double and convert if it isnt
        if (imgsobel.type()!=CV_64F) imgsobel.convertTo(imgsobel, CV_64F);   //check if img is double and convert if it isnt
        
        //sobelx_.print();
        cv::Mat imgsobelx = sobelx_.conv(img);                          //sobel in one dimension
        //sobely_.print();
        cv::Mat imgsobely = sobely_.conv(img);                          //sobel in the other dimension
        imgsobel = (imgsobelx.mul(imgsobelx)+imgsobely.mul(imgsobely)); //pythagoras in two lines
        cv::sqrt(imgsobel,imgsobel);
    }


};

//color segmentation of BGR based on the channel chosen in prefered_channel
void segment(std::vector<cv::Mat> BGR, cv::Mat &color, int prefered_channel) 
{
    if( prefered_channel < 0 || prefered_channel > 2 ) {
        std::cout << "choose channel 0,1 or 2\n"; // Print error msg
        color=cv::Mat::zeros(3,3,CV_64F);                                  // Failure doesn't occur to lazy find a compilable solution
    }
    
    for(int i=0; i< BGR.size() ;++i){                       //check if BGR is double and convert if it isnt
    if (BGR[i].type()!=CV_64F) BGR[i].convertTo(BGR[i], CV_64F);  
    }

    if(prefered_channel==0) color =   2*BGR[0] - BGR[1] - BGR[2];       //color segmentation based on chosen channel
    if(prefered_channel==1) color = - BGR[0] + 2*BGR[1] - BGR[2];
    if(prefered_channel==2) color = - BGR[0] - BGR[1] + 2*BGR[2];
    
}


//returns an 8bit mat that is scaled to the [0,255] interval
cv::Mat normalize(cv::Mat mat)
{
    if (mat.type()!=CV_64F) mat.convertTo(mat, CV_64F);                 //check if img is double and convert if it isnt
    
    double min=255, max=0;                                              //slow but works
        for(int i = 0; i < mat.rows; i++)
        {
            for(int j = 0; j < mat.cols; j++)
            {
                min=std::min(min,mat.at<double>(i,j));
                max=std::max(max,mat.at<double>(i,j));
            }
        }
    mat = mat-min;
    mat = (mat/(max-min))*255;
    //cout  << "min " << min << "  max " << max <<endl;
    mat.convertTo(mat, CV_8UC1);
    return mat;
}

int main( int argc, char** argv ) {
    if( argc != 2 ) {           //example code by woeber
        std::cout << "Usage: ./OpenCV <Pfad-zum-Bild>\n"; // Pritn usage message
        return -1;                                        // Failure occurs
    }


    double m[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};     //explicit definition of double array
    cv::Mat kernel = cv::Mat(3, 3, CV_64F, m);              //feed Mat with the double array 
    discreteConvolution *denoise = new discreteConvolution(kernel);
    //denoise->print();




    std::string path_to_image = std::string( argv[ 1 ] );    //example code by woeber
    std::cout << "Got path to image:" << path_to_image << std::endl;
    //------------------------------//
    //--- Load iamge and test it ---//
    //------------------------------//
    cv::Mat img = cv::imread( path_to_image ); // Load image
    if( img.rows <= 0 ) {
        std::cout << "Faulty iamge file\n"; // Pritn error msg
        return -1;                          // Failure occurs
    }//End check image

    std::vector< cv::Mat > layers;
    cv::Mat convimage, charconvimage, segimg; 
    cv::split(img, layers); // Split channles 

    segment(layers, segimg ,2);
    //segimg=normalize(segimg);

    //bimg = {onemat,zeromat,eyemat}; //fill with different pointers?
    //segimg=cv::Mat::zeros(img.rows, img.cols,CV_64F);

    /*for (int i=0;i<3;++i)
    {
        convimage = denoise->conv(layers[i]);
        convimage.convertTo(bimg.at(i), CV_8UC1);

        //convimage.convertTo(charconvimage), CV_8UC1);                     //why does it overwrite itself if i save it in between? does it save a pointer to charconvimage into bimg?
        //bimg.at(i)=charconvimage;
    } */
    convimage = denoise->conv(segimg);
    charconvimage=normalize(convimage);
    
    cv::Mat edges; 
    
    sobelDetector sbldt;        //initialzise sobel class
    sbldt.getEdges(edges, convimage);
    
    cv::Mat charedges=normalize(edges);     //make it bite sized
    cv::Mat thresholdedimg;                 
    cv::threshold(charedges, thresholdedimg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  //otsu threshold returns binary image


    //Detect circles in the image
    vector<cv::Vec3f> circles;
    cv::HoughCircles(thresholdedimg, circles, cv::HOUGH_GRADIENT, 2,
                     thresholdedimg.rows / 8,  // min distance between circles
                     100, 30, 20, 50 // last two parameters (min_radius & max_radius) are circle size in pixel
                     // 
    );

    cv::Mat imgcopy=charedges.clone();

    //Draw circles detected
    for(size_t i = 0; i < circles.size(); i++) {
        cv::Vec3i c = circles[i];
        //std::cout << circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        // circle center
        //cv::circle(imgcopy, center, 1, 255, 3, cv::LINE_AA);
        // circle outline
        cv::circle(imgcopy, center, radius, 255, 3, cv::LINE_AA);
    }

    //Show the result
    cv::imshow("Hough Circles", img);
    
    //cv::Mat merged;
    //cv::merge(bimg, merged);

    //cv::namedWindow("denoised Image", 0);
    //cv::imshow("denoised Image", merged);
    cv::namedWindow("original Image", 0); 
    cv::imshow("original Image", img);

    cv::namedWindow("colorsegmented Image", 0); 
    cv::imshow("colorsegmented Image", charconvimage);

    cv::namedWindow("sobel Image", 0);
    cv::imshow("sobel Image", charedges);

    cv::namedWindow("thresholded Image", 0);  
    cv::imshow("thresholded Image", thresholdedimg);

    cv::namedWindow("original Image with circles", 0); 
    cv::imshow("original Image with circles", imgcopy);

    cv::waitKey(0);                     //Print iamge and wait for user input 


} // end main function