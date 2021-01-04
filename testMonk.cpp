#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <functional>
#include <string>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <thread> 
#include <mutex>
#include <sstream>
#include <algorithm>
#include <random>
#include "MLP.h"
#include "Utilities.h"

// #define LOG

using namespace std;

#define MONKS_ROOT "/Users/phamgiang/Study/Master/ThirdSemeter/ML/MLProject/Monks/"

/**
 * Function to read monk files
 * filename: name of the file
 * cls: vector to store the list of all targets  
 * atts: vector to store the list of all data
 * names: vector to store the list of all data ids 
 * */
void read_file_monks(const char* filename,vector<vector<float> >& cls, vector<vector<float> >& atts, vector<string>& names)
{
    //read data 
    ifstream file (filename);
    if(file.is_open())
    {
        string line;
        while (getline(file, line))
        {
            // cout << line << endl;
            int ou, in1, in2, in3, in4, in5, in6;
            char cname[16];
            sscanf(line.c_str(),"%d %d %d %d %d %d %d %s", &ou, &in1, &in2, &in3, &in4, &in5, &in6, cname);
            vector<float> temou {(float) ou};
            cls.push_back(temou);
            
            vector<float> tem;
            //one-hot-encoding for monk data 

            // a1: 1, 2, 3  
            switch (in1)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f); tem.push_back(0.0f);
                break;
            case 2: 
                tem.push_back(0.0f); tem.push_back(1.0f); tem.push_back(0.0f);
                break;
            default:
                tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(1.0f);
                break;
            }
            // a2: 1, 2, 3
            switch (in2)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f); tem.push_back(0.0f);
                break;
            case 2: 
                tem.push_back(0.0f); tem.push_back(1.0f); tem.push_back(0.0f);
                break;
            default:
                tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(1.0f);
                break;
            }
            // a3: 1, 2
            switch (in3)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f); 
                break;
            default:
                tem.push_back(0.0f); tem.push_back(1.0f); 
                break;
            }
            // a4: 1, 2, 3
            switch (in4)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f); tem.push_back(0.0f);
                break;
            case 2: 
                tem.push_back(0.0f); tem.push_back(1.0f); tem.push_back(0.0f);
                break;
            default:
                tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(1.0f);
                break;
            }
            // a5: 1, 2, 3, 4
            switch (in5)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(0.0f);
                break;
            case 2: 
                tem.push_back(0.0f); tem.push_back(1.0f); tem.push_back(0.0f); tem.push_back(0.0f);
                break;
            case 3:
                tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(1.0f); tem.push_back(0.0f);
                break;
            default:
                tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(0.0f); tem.push_back(1.0f);
                break;
            }
            // a6: 1, 2
            switch (in6)
            {
            case 1:
                tem.push_back(1.0f); tem.push_back(0.0f);
                break;
            default:
                tem.push_back(0.0f); tem.push_back(1.0f); 
                break;
            }
            atts.push_back(tem);
            names.push_back(string(cname));
        }
        file.close();
        #ifdef DEBUG 
        cout << "Finish read file " << filename << endl;
        #endif
    }
    else 
    {
        cerr << "Cannot open file " << filename << endl; 
    }
    #ifdef TEST_UNIT
    for(int i = 0; i < names.size(); i++)
    {
        cout << cls[i][0] << " ";
        for(int j = 0; j < atts[0].size(); j++)
        {
            cout << atts[i][j] << " ";
        }
        cout << names[i] << endl;
    }
    #endif 
}

/**
 * Function train mlp with monks dataset
 * Read the training file and then train mlp with the obtained data 
 * + monk_id: id of monks dataset (1, 2, 3)
 * + mlp: network 
 * + ep: stopping criteation of total sumsquare error
 * + max_iter: maximum iteration
 * + eta: learning rate
 * + alfa: momentum parameter 
 * + lambda: weight decay (regularization) parameter
 * */
void train_monk(int monk_id, MLP* mlp, float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
{
    vector<vector<float> > trcls;
    vector<vector<float> > tratts;
    vector<string> trnames;
    string monks_name = string("monks-")  + to_string(monk_id) + string(".train");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), trcls, tratts, trnames);
    mlp->train_batch(tratts, trcls, ep, max_iter, eta, alfa, lambda);
    // train_stochastic(mlp, tratts, trcls, 0.1f, 50000, 0.03f, 0.0f, 0.0001f);
}

/**
 * Function to test monks dataset with trained mlp 
 * read the test file and them test the trained mlp with the test data
 * + monk_id: id of monk dataset 1,2 or 3
 * + mlp: trained network
 * */
float test_monk(int monk_id, MLP* mlp)
{
    vector<vector<float> > tcls;
    vector<vector<float> > tatts;
    vector<string> tnames;
    string monks_name = string("monks-")  + to_string(monk_id) + string(".test");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), tcls, tatts, tnames);

    return binary_classify_evaluate(mlp, tatts, tcls);
}


/**
 * Function to get the learning curves from monk datasets.
 * Read monk files (train and test) the perform training and testing together in each epoch
 * + monk_id: monk dataset 1, 2 or 3
 * */
void get_curve_monk(int monk_id)
{
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;
    vector<string> trnames;

    vector<vector<float> > tsdata;
    vector<vector<float> > tstarget;
    vector<string> tsnames;

    string train_name = string(MONKS_ROOT) + string("monks-")  + to_string(monk_id) + string(".train");
    string test_name = string(MONKS_ROOT) + string("monks-")  + to_string(monk_id) + string(".test");

    read_file_monks(train_name.c_str(), trtarget, trdata, trnames);
    read_file_monks(test_name.c_str(), tstarget, tsdata, tsnames);

    //create the model 
    int insize = 18; // 17 + 1 bias term
    int outsize = 1; // binary classification 

    vector<int> hid {8, 4};
    MLP* mlp = new MLP(insize, outsize, hid.size(), hid, "gauss", "sigmoid");

    float trmse, tsmse, tracc, tsacc;
    int iter = 0;

    do
    {
        iter ++;
        mlp->train_batch(trdata, trtarget, 0.001, 1, 0.07f, 0.001f, 0.0007f);
        trmse = mlp->compute_sum_square_multithreads(trdata, trtarget);
        tsmse = mlp->compute_sum_square_multithreads(tsdata, tstarget);
        tracc = binary_classify_evaluate(mlp, trdata, trtarget);
        tsacc = binary_classify_evaluate(mlp, tsdata, tstarget);
        // cout << iter << "\t" << trmse << "\t" << tsmse << "\t" << tracc << "\t" << tsacc << endl;
        if (tsacc >= 1.0f) break;
    }while (iter < 1000);

    cout << iter << "\t" << trmse << "\t" << tsmse << "\t" << tracc << "\t" << tsacc << endl;

}

/**
 * Function evaluate monk by train the mlp and then test the trained mlp
 * + monk_id: id of monk dataset 1, 2 or 3
 * */
void eval_monk(int monk_id)
{
    int in_size = 18; // 17 one hot attributes for data plus one bias term
    int out_size = 1; // binary classification

    //train 
    vector<int> hid_size {4, 2};
    MLP* mlp = new MLP(in_size, out_size, hid_size.size(), hid_size, "leakyRelu", "tanh");
    train_monk(monk_id, mlp, 0.001f, 1, 0.7f, 0.0007f, 0.001f);

    //evaluate 
    float res = test_monk(monk_id, mlp);
    // cout << "Res = " << res << endl;
    // if (res >= 0.5)
    // {
    //     stringstream filename;
    //     filename << MONKS_ROOT << "/monks-" << monk_id << "-" << res << ".model";
    //     // cout << "Save model to file: " << filename.str() << endl;
    //     mlp->save_model(filename.str());
    // }

    delete mlp;
}

/**
 * Function to evaluate monk dataset, create a mlp then train and validate by k-fold validation
 * Retrain with all the training data then test monk
 * + monk_id: monk dataset 1, 2 or 3
 * */
void validation_monks(int monk_id)
{
    //create network 
    int in_size = 18; // 17 attributes for data plus one bias term
    int out_size = 1; // binary classification

    vector<int> hid_size {20, 10, 4};
    MLP* mlp = new MLP(in_size, out_size, hid_size.size(), hid_size, "leakyRelu", "tanh");

    vector<vector<float> > data;
    vector<vector<float> > target;
    vector<string> names;

    string monks_name = string("monks-")  + to_string(monk_id) + string(".train");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), target, data, names);

    float res = kfold_validation(mlp, 2, 1, data, target, 5, 1000, 0.01f, 0.7f, 0.001f, 0.001f);

    cout << "K-fold result: " << res << endl;

    mlp->reset_mlp();
    train_monk(monk_id, mlp, 0.1f, 2000, 0.7f, 0.001, 0.001);

    float testres = test_monk(monk_id, mlp);
    cout << "Test result: " << testres << endl;

    if (testres == 1.0f)
    {
        stringstream filename;
        filename << MONKS_ROOT << "monks-" << monk_id << "-" << testres << ".model";
        cout << "Save model to file: " << filename.str() << endl;
        mlp->save_model(filename.str());
    }

}
/**
 * Grid search to find the best model for monk_id dataset 
 * */
void grid_search_monks(int monk_id)
{
    //create network 
    int in_size = 18; // 17 attributes for data plus one bias term
    int out_size = 1; // binary classification

    vector<int> hid_size {8, 4};
    MLP* mlp = new MLP(in_size, out_size, hid_size.size(), hid_size, "gauss", "sigmoid");

    //get the data from file 
    vector<vector<float> > data;
    vector<vector<float> > target;
    vector<string> names;

    string monks_name = string("monks-")  + to_string(monk_id) + string(".train");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), target, data, names);

    //create vector lists of parameters 
    vector<float> eta_values{0.03f, 0.1f, 0.3f};
    vector<float> alfa_values{0.001f, 0.0003f, 0.00009f};
    vector<float> lambda_values{0.001f, 0.0003f, 0.00009f};
    vector<float> best_com = grid_search(mlp, 2, 1, 5, data, target, eta_values, alfa_values, lambda_values);
    for(int i = 0; i < best_com.size(); i++)
    {
        cout << best_com[i] << " \t";
    }
    cout << endl;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    grid_search_monks(3);
    return 0;
}   