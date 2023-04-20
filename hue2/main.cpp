/*Code by Ivo ackermann unless stated otherwise
loads the mnist dataset and computes a logistic regression classifier for 4 and 7. */

#include <iostream>           //Basic I/O Operations
#include "opencv2/opencv.hpp" //Include all OpenCV header


using namespace std;

void extract47(cv::Mat &samples_47t, cv::Mat &labels_47, std::string source)
{
    cv::Ptr< cv::ml::TrainData > tdata = cv::ml::TrainData::loadFromCSV( source, 0, 0, 1 ); // First col is the target as a float (32 bit, CV_32F, 5)
    cv::Mat samples = tdata->getTrainSamples( );                                                        // Get design matrix
    cv::Mat target = tdata->getTrainResponses( );                                                          // Get target values

    //cout << "type:" << samples.type()<<"size:"<<samples.size() <<endl;

    vector< int > indices_4, indices_7;        // extract target indices of 4 and 7
    for( int i = 0; i < target.rows; i++ )
    {
        if( target.at< float >( i ) == 4.0 )indices_4.push_back( i ); 
        if( target.at< float >( i ) == 7.0 )indices_7.push_back( i );
    }

    //cout << indices_4.size() << endl;

    cv::Mat samples_4,samples_7;              //extract samples of 4 and 7
    for( int i = 0; i < indices_4.size( ); i++ )        samples_4.push_back( samples.row( indices_4[ i ] ) );
    for( int i = 0; i < indices_7.size( ); i++ )        samples_7.push_back( samples.row( indices_7[ i ] ) );

    //cout << "type: " << samples_4.type() << " size: " << samples_4.size() <<endl;

    cv::Mat samples_47;                     //create design matrix for 4 and 7
    samples_47.push_back( samples_4 );
    samples_47.push_back( samples_7 );
    //cv::Mat samples_47t=samples_47.t();     //transpose to fit the script 
    samples_47t=samples_47;
    cout << "type: " << samples_47t.type() << " size: " << samples_47t.size() <<endl;

    labels_47 = cv::Mat::ones( samples_47.rows, 1, CV_32F );    // create binary labels 0 for 4 and +1 for 7
    for( int i=0; i <indices_4.size( ); i++ ) labels_47.row(i) = 0;
}
/*void sigmoid (cv::Mat &x, cv::Mat &y)
{
    y=1/(1+())
}*/

int main()
{       //mnist dataset is in a subfolder
    cv::Mat samples_47t;
    cv::Mat labels_47;
    std::string traindata ="MNIST_CSV/mnist_train.csv";
    extract47(samples_47t,labels_47,traindata);
    // labels_47.rowRange( 0, samples_4.rows ) *= -1;
 
    //cout << labels_47.size() <<endl;

    //Preprocessing 
    //Standardise the training dataset
    cv::Mat samples_47t_std= cv::Mat(samples_47t.size(), CV_32F);
    cv::Mat mu = cv::Mat::zeros( samples_47t.cols, 1, CV_32F );  // Memory for the means
    cv::Mat var = cv::Mat::ones( samples_47t.cols, 1, CV_32F ); // Memory for the scale
//    cout << "type: " << samples_47t_std.type() << " size: " << samples_47t_std.size() <<endl;

    cv::Scalar mean,stdev;
    //cv::meanStdDev(samples_47t,mean,stdev);
    //cout << mean[2] <<endl;
    for (int i=0;i<samples_47t.cols;i++)
    {
        cv::meanStdDev(samples_47t.col(i),mean,stdev);
        //cout <<mean<<endl<<stdev<<endl;
        mu.at< float >( i ) = static_cast<float>(mean[0]);         // Store mean
        var.at< float >( i ) = static_cast<float>(stdev[0]);         // Store stddeviation
        
        //cout << "size links: " << samples_47t_std.size() << " size: " << samples_47t.col(i).size() <<endl;
        if(stdev[0]==0)
        {samples_47t_std.col(i)=1;}
        else
        {samples_47t_std.col(i)=((samples_47t.col(i)-mean[0])/stdev[0]);}
    }
    //cout<<samples_47t_std<<endl;
        //cout << " size: " << samples_47t_std.col(3) <<endl;
        //cout << " size: " << mu <<endl;


    //samples_47t_std=samples_47t_std.t();  //transpose back because its more intuitive to me 

    //Reduce the standardised training dataset with PCA
    int latentdim = 200;
    cv::PCA pca(samples_47t_std,cv::Mat(),cv::PCA::DATA_AS_ROW,latentdim);
    cv::Mat samples_47t_std_pca=cv::Mat(samples_47t_std.rows,latentdim,CV_32F);
    pca.project(samples_47t_std,samples_47t_std_pca);
    cout << "Latente Dimensionen: " << pca.eigenvectors.rows << std::endl;

    //Apply preprocessing to test dataset
    std::string testdata = "MNIST_CSV/mnist_test.csv";
    cv::Mat test_samples_47t;
    cv::Mat test_target_47t;
    extract47(test_samples_47t,test_target_47t,testdata);
    
    cv::Mat test_samples_47t_std =cv::Mat(test_samples_47t.size(),CV_32F);
    for (int i=0;i<test_samples_47t.cols;i++)
    {
        if(var.at< float >( i )==0)
        {test_samples_47t_std.col(i)=1;}
        else
        {test_samples_47t_std.col(i)=((test_samples_47t.col(i)-mu.at< float >( i ))/var.at< float >( i ));}
        //test_samples_47t_std.col(i)=((test_samples_47t.col(i)-mean[i])/stdev[i]);
    }


    cv::Mat test_samples_47t_std_pca=cv::Mat(test_samples_47t_std.rows,latentdim,CV_32F);
    pca.project(test_samples_47t_std,test_samples_47t_std_pca); 

    //Train logistic regression
    cv::Mat weights=cv::Mat::zeros(samples_47t_std_pca.cols,1,CV_32F);  //initialize weights
    cv::Mat output, errors, gradient, test_output;
    float accuracy;
    for (int i=0;i<15;i++)  //15 iterations
    {
        
        //calculate outputs and errors
        cv::exp(-(samples_47t_std_pca*weights).mul(labels_47),output);
        output=1.0/(output+1.0);
        errors=labels_47-output;
        //cout << "errors size: " << errors.size() << "type: " << errors.type() << endl;
        
        //calculate gradient
        gradient=samples_47t_std_pca.t()*errors;
        //cout << "gradient size: " << gradient.size() << "type: " << gradient.type() << endl;

        //update weights
        weights=weights+0.01*gradient;
        //cout << "weights size: " << weights.size() << "type: " << weights.type() << endl;

        //calculate accuracy
        cv::exp(-(test_samples_47t_std_pca*weights).mul(test_target_47t),test_output);
        test_output=1.0/(test_output+1.0);
        accuracy=100*cv::mean(cv::abs(test_target_47t-test_output))[0];
        std::cout << "Iteration " << i+1 << ": " << accuracy << std::endl;
    }/**/
return 0;
}