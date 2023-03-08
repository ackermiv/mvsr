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
    kernel_ = cv::Mat::ones(3,3, CV_32F);
    kernel_ = kernel_/9;
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
    convimage=cv::Mat::zeros(image.rows-kernel_.rows+1,image.cols-kernel_.cols+1, CV_64FC1);
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
                min=std::min(min,sum);
                max=std::max(max,sum);
            }
        }

    /*    cv::minMaxIdx (convimage, &min, &max,NULL);
    cout  << "min " << min << "  max " << max <<endl;
    convimage = convimage-min;
    convimage = (convimage/(max-min))*255;
    convimage.convertTo(convimage, CV_8UC1);
    return convimage; */


    convimage = convimage-min;
    convimage = (convimage/(max-min))*255;
    //convimage.convertTo(convimage, CV_8UC1);
    return convimage;
}

cv::Mat segment(std::vector<cv::Mat> BGR, int prefered_channel)
{
    if( prefered_channel < 0 || prefered_channel > 2 ) {
        std::cout << "choose channel 0,1 or 2\n"; // Print error msg
        return BGR[0];                          // Failure doesn't occur
    }
    cv::Mat color = cv::Mat::zeros(BGR[0].size(),CV_64F);

    for(int i=0; i< BGR.size() ;++i){
    BGR[i].convertTo(BGR[i], CV_64F);  
    }

    if(prefered_channel==0) color =   2*BGR[0] - BGR[1] - BGR[2];
    if(prefered_channel==1) color = - BGR[0] + 2*BGR[1] - BGR[2];
    if(prefered_channel==2) color = - BGR[0] - BGR[1] + 2*BGR[2];
    cout <<color.type()<<endl;
    //color=normalize(color);
    return color;
}

cv::Mat normalize(cv::Mat mat)
{
    mat.convertTo(mat, CV_64F);
    double min=255, max=0;  
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
    cout  << "min " << min << "  max " << max <<endl;
    mat.convertTo(mat, CV_8UC1);
    return mat;
}

int main( int argc, char** argv ) {
    if( argc != 2 ) {
        std::cout << "Usage: ./OpenCV <Pfad-zum-Bild>\n"; // Pritn usage message
        return -1;                                        // Failure occurs
    }


    double m[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};     //explicit definition of double array
    cv::Mat kernel = cv::Mat(3, 3, CV_64F, m);              //feed Mat with the double array 
    //kernel = kernel/16;                                   //no longer neccessary due to normalization function
    //cv::Mat kernel = cv::Mat::ones(3,3, CV_64F);           //shorter but less flexible
    discreteConvolution *denoise = new discreteConvolution(kernel);
    //denoise->print();



    //example code from woeber
    std::string path_to_image = std::string( argv[ 1 ] );
    std::cout << "Got path to image:" << path_to_image << std::endl;
    //------------------------------//
    //--- Load iamge and test it ---//
    //------------------------------//
    cv::Mat img = cv::imread( path_to_image ); // Load image
    if( img.rows <= 0 ) {
        std::cout << "Faulty iamge file\n"; // Pritn error msg
        return -1;                          // Failure occurs
    }//End check image

    std::vector< cv::Mat > layers;//, bimg; 
    //cv::Mat matbimg[3];
    cv::Mat convimage, charconvimage, segimg; //mat for different steps
    /*cv::Mat zeromat=cv::Mat::zeros(img.rows-kernel.rows+1,img.cols-kernel.cols+1, CV_8UC1); 
    cv::Mat onemat=cv::Mat::ones(img.rows-kernel.rows+1,img.cols-kernel.cols+1, CV_8UC1); 
    cv::Mat eyemat=cv::Mat::eye(img.rows-kernel.rows+1,img.cols-kernel.cols+1, CV_8UC1);  */
    cv::split( img, layers ); // Split channles 

    segimg=segment(layers,2);
    //segimg=normalize(segimg);

    //bimg = {onemat,zeromat,eyemat}; //fill with different pointers?
    //segimg=cv::Mat::zeros(img.rows, img.cols,CV_64F);

    /*for (int i=0;i<3;++i)
    {
        convimage = denoise->conv(layers[i]);
        convimage.convertTo(bimg.at(i), CV_8UC1);

        //convimage.convertTo(charconvimage), CV_8UC1);                     why does it overwrite itself if i save it in between? does it save a pointer to charconvimage into bimg?
        //bimg.at(i)=charconvimage;
    } */
    convimage = denoise->conv(segimg);
    charconvimage=normalize(convimage);

    //cv::Mat merged;
    //cv::merge(bimg, merged);

    //cv::namedWindow("denoised Image", 0);           //Create window with adjustable settigns
    //cv::imshow("denoised Image", merged);           //Move image to window
    cv::namedWindow("original Image", 0);           //Create window with adjustable settigns
    cv::imshow("original Image", img);           //Move image to window    
    cv::namedWindow("colorsegmented Image", 0);           //Create window with adjustable settigns
    cv::imshow("colorsegmented Image", charconvimage);           //Move image to window

    cv::waitKey(0);                     //Pritn iamge and wait for user input 


} // end main function
