/*Code by Ivo ackermann unless stated otherwise
loads the mnist dataset and computes a classifier for 4 and 7. */

#include <iostream>           //Basic I/O Operations
#include "opencv2/opencv.hpp" //Include all OpenCV header

  
using namespace std;


int main()
{       //mnist dataset is in a subfolder
    cv::Ptr< cv::ml::TrainData > tdata = cv::ml::TrainData::loadFromCSV( "MNIST_CSV/mnist_test.csv", 0, 0, 1 ); // First col is the target as a float (32 bit, CV_32F, 5)
    cv::Mat samples = tdata->getTrainSamples( );                                                        // Get design matrix
    cv::Mat target = tdata->getTrainResponses( );                                                          // Get target values

    cout << samples.type()<<"size:"<<samples.size() <<endl;

    vector< int > indices_4, indices_7;        // extract target indices of 4 and 7
    for( int i = 0; i < target.rows; i++ )
    {
        if( target.at< float >( i ) == 4.0 )indices_4.push_back( i ); 
        if( target.at< float >( i ) == 7.0 )indices_7.push_back( i );
    }

    cout << indices_4.size() << endl;

    cv::Mat samples_4,samples_7;              //extract samples of 4 and 7
    for( int i = 0; i < indices_4.size( ); i++ )        samples_4.push_back( samples.row( indices_4[ i ] ) );
    for( int i = 0; i < indices_7.size( ); i++ )        samples_7.push_back( samples.row( indices_7[ i ] ) );

    cout << "type: " << samples_4.type() << " size: " << samples_4.size() <<endl;

    cv::Mat samples_47;                     //create design matrix for 4 and 7
    samples_47.push_back( samples_4 );
    samples_47.push_back( samples_7 );


    cv::Mat labels_47 = cv::Mat::ones( samples_47.rows, 1, CV_32SC1 );    // create binary labels -1 for 4 and +1 for 7
    for( int i=0; i <indices_4.size( ); i++ ) labels_47.row(i)*= -1;
    // labels_47.rowRange( 0, samples_4.rows ) *= -1;




    return 0;
}