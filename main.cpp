#include <iostream>           //Basic I/O Operations
#include "opencv2/opencv.hpp" //Include all OpenCV header

  
using namespace std;





//Class template
class discreteConvolution
{   
private: //private members
    cv::Mat kernel_;

public: //public members
    //constructors
    discreteConvolution();
    discreteConvolution(cv::Mat);
    
    //public member functions
    void print();
    void conv(cv::Mat image, cv::Mat &convimage);
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


//public member functions definitions
void discreteConvolution::print()           //for debugging
{
    std::cout << kernel_ << endl;
}

void discreteConvolution::conv(cv::Mat image, cv::Mat &convimage)
{
    image.convertTo(image, CV_64FC1);
    double min, max; 
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

            }
        }

}



class sobelDetector
{
    private:
            double m[3][3] = {{1., 2., 1.}, {0., 0., 0.}, {-1., -2., -1.}};     //explicit definition of sobel gradient detector kernel
            cv::Mat kernel_ = cv::Mat(3, 3, CV_64F, m);                         //feed Mat with the double array 
            discreteConvolution sobelx_ = discreteConvolution(kernel_);
            discreteConvolution sobely_ = discreteConvolution(kernel_.t());     //.t transposes cv::Mat


    public:
        sobelDetector(){}
        
        //returns the edges of an image to imgsobel
        void getEdges(cv::Mat img,cv::Mat &imgsobel) 
        {
            if (img.type()!=CV_64F) img.convertTo(img, CV_64F);                  //check if img is double and convert if it isnt
            if (imgsobel.type()!=CV_64F) imgsobel.convertTo(imgsobel, CV_64F);   //check if img is double and convert if it isnt
            
            //sobelx_.print();
            cv::Mat imgsobelx, imgsobely;
            sobelx_.conv(img, imgsobelx);                          //sobel in one dimension
            //sobely_.print();
            sobely_.conv(img, imgsobely);                          //sobel in the other dimension
            imgsobel = (imgsobelx.mul(imgsobelx)+imgsobely.mul(imgsobely)); //pythagoras in two lines
            cv::sqrt(imgsobel,imgsobel);
        }


};



//color segmentation of BGR based on the channel chosen in prefered_channel
void segment(std::vector<cv::Mat> BGR, cv::Mat &color, int prefered_channel) 
{
    if( prefered_channel < 0 || prefered_channel > 2 ) {
        std::cout << "choose channel 0,1 or 2\n";                           // Print error msg
        color=cv::Mat::zeros(3,3,CV_64F);                                  // Failure doesn't occur to lazy find a compilable solution
    }
    
    for(int i=0; i< BGR.size() ;++i){                                       //check if BGR is double and convert if it isnt
    if (BGR[i].type()!=CV_64F) BGR[i].convertTo(BGR[i], CV_64F);  
    }
    if (color.type()!=CV_64F) color.convertTo(color, CV_64F);                 //check if img is double and convert if it isnt

    if(prefered_channel==0) color =   2*BGR[0] - BGR[1] - BGR[2];       //color segmentation based on chosen channel
    if(prefered_channel==1) color = - BGR[0] + 2*BGR[1] - BGR[2];
    if(prefered_channel==2) color = - BGR[0] - BGR[1] + 2*BGR[2];
}



//returns an 8bit mat that is scaled to the [0,255] interval
void normalize(cv::Mat mat, cv::Mat &normmat)
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
    mat.convertTo(normmat, CV_8UC1);
    
}



int main( int argc, char** argv ) {             //example code by woeber
    if( argc != 2 ) {           
        std::cout << "Usage: ./OpenCV <Pfad-zum-Bild>\n"; // Pritn usage message
        return -1;                                        // Failure occurs
    }

    std::string path_to_image = std::string( argv[ 1 ] );    //example code by woeber
    std::cout << "Got path to image:" << path_to_image << std::endl;
    //------------------------------//
    //--- Load iamge and test it ---//
    //------------------------------//
    cv::Mat img = cv::imread( path_to_image ); // Load image
    if( img.rows <= 0 ) {
        std::cout << "Faulty iamge file\n"; // Pritn error msg
        return -1;                          // Failure occurs
    }//End check image                          //end of excample code


    double m[3][3] = {{1., 2., 1.}, {2., 4., 2.}, {1., 2., 1.}};     //explicit definition of double array gaussian
    cv::Mat kernel = cv::Mat(3, 3, CV_64F, m);              //feed Mat with the double array 
    discreteConvolution *denoise = new discreteConvolution(kernel);
    //denoise->print();

    std::vector< cv::Mat > layers;
    cv::Mat convimage, charconvimage, segimg, edges, charedges, thresholdedimg; 
    cv::split(img, layers);         // Split channles into 3 cv::Mats instead of 1 cv::Mat that has vec3 entries
    segment(layers, segimg ,2);     // colorsegmentation

    denoise->conv(segimg, convimage);   //linear convolution with gaussian kernel
    normalize(convimage, charconvimage); //just for controlimage 
    
    
    sobelDetector sbldt;                    //initialzise sobel class
    sbldt.getEdges(convimage, edges);       //uses sobel detector
    normalize(edges, charedges);            //make it byte sized

    cv::threshold(charedges, thresholdedimg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  //otsu threshold returns binary image


    //Detect circles in the image
    vector<cv::Vec3f> circles;
    cv::HoughCircles(thresholdedimg, circles, cv::HOUGH_GRADIENT, 2,
                     thresholdedimg.rows / 8,       // min distance between circles
                     100, 30, 20, 50                // last two parameters (min_radius & max_radius) are circle size in pixels
    );

    cv::Mat imgcopy=charedges.clone();

    //Draw circles detected
    for(size_t i = 0; i < circles.size(); i++) {
        cv::Vec3i c = circles[i];
        //std::cout << circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        cv::circle(imgcopy, center, radius, 255, 3, cv::LINE_AA);   // circle outline
    }

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