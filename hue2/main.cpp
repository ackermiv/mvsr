/*Code by Ivo ackermann unless stated otherwise
loads the mnist dataset and computes a logistic regression classifier for 4 and 7. */
//mnist dataset is in a subfolder

#include <iostream>           //Basic I/O Operations
#include "opencv2/opencv.hpp" //Include all OpenCV header


using namespace std;

void extract47(cv::Mat &samples_47, cv::Mat &labels_47, std::string source)
{
    cv::Ptr< cv::ml::TrainData > tdata = cv::ml::TrainData::loadFromCSV( source, 0, 0, 1 ); // First col is the target as a float (32 bit, CV_32F, 5)
    cv::Mat samples = tdata->getTrainSamples( );                                                        // Get design matrix
    cv::Mat target = tdata->getTrainResponses( );                                                       // Get target values

    //cout << "type:" << samples.type()<<"size:"<<samples.size() <<endl;

    vector< int > indices_4, indices_7;        // extract target indices of 4 and 7
    for( int i = 0; i < target.rows; i++ )
    {
        if( target.at< float >( i ) == 4.0 )indices_4.push_back( i ); 
        if( target.at< float >( i ) == 7.0 )indices_7.push_back( i );
    }

    //extract samples of 4 and 7
    for( int i = 0; i < indices_4.size( ); i++ )        samples_47.push_back( samples.row( indices_4[ i ] ) );
    for( int i = 0; i < indices_7.size( ); i++ )        samples_47.push_back( samples.row( indices_7[ i ] ) );

    //cout << "type: " << samples_47.type() << " size: " << samples_47.size() <<endl;

    labels_47 = cv::Mat::ones( samples_47.rows, 1, CV_32F );    // create binary labels 0 for 4 and +1 for 7
    for( int i=0; i <indices_4.size( ); i++ ) labels_47.row(i) = 0;
}


void sigmoid (cv::Mat &x, cv::Mat &y)
{
    cv::exp(-x,x);
    y=1/(1+x);
}


float faccuracy(cv::Mat &prediction,cv::Mat &truth)
{
    //prediction.forEach<float>([](float& element) { element = round(element); });
    cv::Mat tmp = cv::Mat(prediction.size(),CV_32F);
    cv::Scalar one = cv::Scalar (1);
    cv::Scalar ret;
    for ( int i=0; i < prediction.rows; i++)
    {prediction.at<float>(i)=round(prediction.at<float>(i));}
    //cout <<"sizes:"<< prediction.size() << "," <<truth.size() << endl;
    tmp= prediction-truth;
    ret= cv::sum(cv::abs(tmp))/prediction.rows;
    ret= one-ret;//cv::Mat::ones(ret.size(),CV_32F)
    return ret[0]*100;
}


int main()
{       
    cv::Mat samples_47;
    cv::Mat labels_47;
    std::string traindata ="MNIST_CSV/mnist_train.csv";
    extract47(samples_47,labels_47,traindata); //loads the bigger train dataset and extracts 4 and 7
    //cout << labels_47.size() <<endl;

    //Preprocessing 
    //Standardise the training dataset
    cv::Mat samples_47_std= cv::Mat(samples_47.size(), CV_32F);  //could overwrite samples_47 to save space
    cv::Mat mu = cv::Mat::zeros( samples_47.cols, 1, CV_32F );  // Memory for the means
    cv::Mat var = cv::Mat::ones( samples_47.cols, 1, CV_32F ); // Memory for the scale
//    cout << "type: " << samples_47_std.type() << " size: " << samples_47_std.size() <<endl;

    cv::Scalar mean,stdev;
    for (int i=0;i<samples_47.cols;i++)
    {
        cv::meanStdDev(samples_47.col(i),mean,stdev);
        //cout <<mean<<endl<<stdev<<endl;
        mu.at< float >( i ) = static_cast<float>(mean[0]);         // Store mean
        var.at< float >( i ) = static_cast<float>(stdev[0]);         // Store stddeviation
        
        //cout << "size links: " << samples_47_std.size() << " size: " << samples_47.col(i).size() <<endl;
        if(stdev[0]==0)
        {samples_47_std.col(i)=1;}
        else
        {samples_47_std.col(i)=((samples_47.col(i)-mean[0])/stdev[0]);}
    }

    //Reduce the standardised training dataset with PCA
    int latentdim = 350; //bisection to a reasonable point where nearly no losses in accuracy occur
    cv::PCA pca(samples_47_std,cv::Mat(),cv::PCA::DATA_AS_ROW,latentdim);
    cv::Mat samples_47_std_pca=cv::Mat(samples_47_std.rows,latentdim,CV_32F);
    pca.project(samples_47_std,samples_47_std_pca);
    //cout << "Latente Dimensionen: " << pca.eigenvectors.rows << std::endl;

    //Apply preprocessing to test dataset
    std::string testdata = "MNIST_CSV/mnist_test.csv";
    cv::Mat test_samples_47;
    cv::Mat test_target_47;
    extract47(test_samples_47,test_target_47,testdata); //loads the smaller test dataset and extracts 4 and 7  
    
    //standardize with same mean/stdev as with training data
    cv::Mat test_samples_47_std =cv::Mat(test_samples_47.size(),CV_32F);
    for (int i=0;i<test_samples_47.cols;i++)
    {
        if(var.at< float >( i )==0)
        {test_samples_47_std.col(i)=1;}
        else
        {test_samples_47_std.col(i)=((test_samples_47.col(i)-mu.at< float >( i ))/var.at< float >( i ));}
    }

    //project into the same space as training data
    cv::Mat test_samples_47_std_pca=cv::Mat(test_samples_47_std.rows,latentdim,CV_32F);
    pca.project(test_samples_47_std,test_samples_47_std_pca); 

    //Train logistic regression
    cv::Mat weights=cv::Mat::ones(samples_47_std_pca.cols,1,CV_32F);  //initialize weights
    cv::Mat prediction, errors, gradient, test_prediction;
    float accuracy;
    for (int i=0;i<20;i++)  //roughly stabil after 15 iterations
    {
        
        //calculate errors in the training set
        cv::Mat Z=(samples_47_std_pca*weights);
        sigmoid(Z,prediction);
        errors=labels_47-prediction;
        //cout << errors << endl;
        
        //calculate gradient
        gradient=samples_47_std_pca.t()*errors;

        //update weights
        weights=weights+0.01*gradient;

        //calculate accuracy of the test set
        cv::Mat test_Z=(test_samples_47_std_pca*weights);
        sigmoid(test_Z,test_prediction);
        accuracy = faccuracy(test_prediction,test_target_47);
        cout << "Iteration " << i+1 << ": " << accuracy <<"%" << endl;
    }/**/
return 0;
}