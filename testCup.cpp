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

using namespace std; 

#define CUP_ROOT    "/Users/phamgiang/Study/Master/ThirdSemeter/ML/MLProject/CUP/"

#define TRFILE      "train.csv"
#define INTSFILE    "ints.csv"
#define ALLTRFILE   "ML-CUP20-TR.csv"
#define REALTEST    "ML-CUP20-TS.csv"

// #define SAVE_SPLIT_FILE 

vector<float> scaler {25.978398f, 78.83648f, -41.864859, -4.064733}; //vector contains min_out1, max_out1, min_out2, max_out2 for normalizaion reason 
// #define TEST_UNIT

/**
 * Function to read both the training and testing data 
 * + alltridx: list of training data indices
 * + alltrdata: list of traning data 
 * + alltrtarget: list of traning targets 
 * */
void read_trdata(string trname, vector<int> &alltridx, vector<vector<float> > &alltrdata, vector<vector<float> > &alltrtarget)
{
    //clear all the vectors first 
    alltridx.clear();
    alltrdata.clear();
    alltrtarget.clear();

    //read training file
    ifstream trfile(trname);
    if(trfile.is_open())
    {
        string line;
        //get rid of 7 first lines or the original file
        if (trname == string(CUP_ROOT) + string(ALLTRFILE) || 
            trname == string(CUP_ROOT) + string(REALTEST))
        {
            for(int i = 0; i < 7; i++)
            {
                getline(trfile, line); 
            }
        }
        while(getline(trfile, line))
        {
            stringstream ss;
            ss << line;
            int temidx;
            ss >> temidx;
            char c;
            ss >> c;
            vector<float> temin;
            for(int i = 0; i < 10; i++) //get 10 input
            {
                float tematt;
                ss >> tematt;
                ss >> c;
                // cout << tematt;
                temin.push_back(tematt);
            }
            vector<float> temout;
            for(int i = 0; i < 2; i++)
            {
                float tematt;
                ss >> tematt;
                // if (tematt < 0) {tematt = tematt*(-1);}
                ss >> c;
                temout.push_back(tematt);
            }
            alltridx.push_back(temidx);
            alltrdata.push_back(temin);
            alltrtarget.push_back(temout);
        }
    }
    else
    {
        cerr << "Cannot open file " << trname << endl;
    }

    #ifdef TEST_UNIT
    vector<int> tidx{34, 22, 53, 35, 89, (int)alltrdata.size()-1};
    for (int j = 0; j < tidx.size(); j++)
    {
        cout << alltridx[tidx[j]] << " ";
        for(int i = 0; i < 10; i++)
        {
            cout << alltrdata[tidx[j]][i] << " ";
        }
        for(int i = 0; i < 2; i++)
        {
            cout << alltrtarget[tidx[j]][i] << " ";
        }
        cout << endl;
    }
    #endif

}

/**
 * Function to read the test file
 * - tsidx: vector will contain the test index,
 * - tsdata: vector will contain the test data
 * */
void read_tsdata(vector<int> &tsidx, vector<vector<float> > &tsdata)
{
    //clear all the vector first 
    tsidx.clear();
    tsdata.clear();
    string tsname= string(CUP_ROOT) + string("ML-CUP20-TS.csv");

    //read test file
    ifstream tsfile(tsname);
    if(tsfile.is_open())
    {
        string line;

        //get rid of 7 first lines
        for(int i = 0; i < 7; i++)
        {
            getline(tsfile, line); 
        }
        while(getline(tsfile, line))
        {
            stringstream ss;
            ss << line;
            int temidx;
            char c;
            ss >> temidx;
            ss >> c;
            vector<float> temin;
            for(int i = 0; i < 10; i++) //get 10 input
            {
                float tematt;
                ss >> tematt;
                ss >> c;
                temin.push_back(tematt);
            }
            tsidx.push_back(temidx);
            tsdata.push_back(temin);
        }
    }
    else
    {
        cerr << "Cannot open file " << tsname << endl;
    }

    #ifdef TEST_UNIT
    vector<int> tidx{34, 22, 53, 35, 89, (int)tsdata.size()-1};
    for (int j = 0; j < tidx.size(); j++)
    {
        cout << tsidx[tidx[j]] << " ";
        for(int i = 0; i < 10; i++)
        {
            cout << tsdata[tidx[j]][i] << " ";
        }
        cout << endl;
    }
    #endif


}

/**
 * Normalize the target of training data to range [0, 1]
 * using (x - min)/(max - min) 
 * -alltrtarget: vector of training target
 * */
void preprocess(vector<vector<float> > &alltrtarget)
{
    for(int i = 0; i < alltrtarget.size(); i++)
    {
        alltrtarget[i][0] = (alltrtarget[i][0] - scaler[0])/(scaler[1] - scaler[0]);
        alltrtarget[i][1] = (alltrtarget[i][1] - scaler[2])/(scaler[3] - scaler[2]);
    }
}




/**
 * Function to split training set into internal test set + training set (will be used in kfold), 
 * run only once to save sets to file, next time load the saved files
 * - ratio: ratio of the internal test set, in range [0,1].
 * - alltridx: input vector contains all the training idx 
 * - alltrdata: input vector contains all the training data
 * - alltrtarget: input vector contains all the training target
 * 
 * - tridx: output vector contains the training idx
 * - trdata: output vector contains the training data
 * - trtarget: ouput vector contains the training target 
 * 
 * - intsidx: output vector contains the internal test idx
 * - intsdata: output vector contains the internal test data 
 * - intstarget: output vector contains the internal test target 
 * */
void split_sets(float ratio, vector<int> alltridx, vector<vector<float> > alltrdata, vector<vector<float> > alltrtarget, 
                vector<int>& tridx, vector<vector<float> >& trdata, vector<vector<float> >& trtarget, 
                vector<int>& intsidx, vector<vector<float> >& intsdata, vector<vector<float> >& intstarget)
{
    //clear all the output vector first 
    trdata.clear(); trtarget.clear(); 
    intsdata.clear(); intstarget.clear();

    //shuffle all the training data 
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(alltridx.begin(), alltridx.end(), g);

    //get the number of internal test set
    int intssize = (int)alltridx.size()*ratio;

    #ifdef SAVE_SPLIT_FILE
    ofstream ftrain (string(CUP_ROOT) + string(TRFILE));
    ofstream fintest (string(CUP_ROOT) + string(INTSFILE));
    #endif


    for(int i = 0; i < alltridx.size(); i++)
    {
        if (i < intssize) //put this sample into the internal test set
        {
            stringstream testss;
            testss.clear();
            intsidx.push_back(alltridx[i]);
            testss << alltridx[i] << ",";
            intsdata.push_back(alltrdata[alltridx[i] - 1]); // because index of the file start from 1
            for(int j = 0; j < 10; j++)
            {
                testss << alltrdata[alltridx[i] - 1][j] << ",";
            }
            intstarget.push_back(alltrtarget[alltridx[i] - 1]); // becaue index of the file start from 1
            for (int j = 0; j < 2; j++)
            {
                testss << alltrtarget[alltridx[i]-1][j];
                if(j == 0) 
                {
                    testss << ",";
                }
            }
            testss << endl;
            #ifdef SAVE_SPLIT_FILE
            fintest << testss.str();
            #endif
        }
        else // put the remaining samples into the training set 
        {
            stringstream trainss;
            tridx.push_back(alltridx[i]);
            trainss << alltridx[i] << ",";
            trdata.push_back(alltrdata[alltridx[i] - 1]); //because index of the file start from 1
            for(int j = 0; j < 10; j++)
            {
                trainss << alltrdata[alltridx[i] -1][j] << ",";
            }
            trtarget.push_back(alltrtarget[alltridx[i] - 1]); //because index of the file start from 1
            for(int j = 0; j < 2; j++)
            {
                trainss << alltrtarget[alltridx[i]-1][j];
                if (j == 0)
                {
                    trainss << ",";
                }
            }
            trainss << endl;

            #ifdef SAVE_SPLIT_FILE
            ftrain << trainss.str();
            #endif
        }
    }

    #ifdef SAVE_SPLIT_FILE
    ftrain.close();
    fintest.close();
    #endif


    #ifdef TEST_UNIT 
    for(int i = 0; i < tridx.size(); i++)
    {
        cout << tridx[i] << " ";
        for (int j = 0; j < 10; j++)
        {
            cout << trdata[i][j] << " ";
        }
        for(int j = 0; j < 2; j++)
        {
            cout << trtarget[i][j] << " ";
        }
        cout << endl;
    }

    for(int i = 0; i < intsidx.size(); i++)
    {
        cout << intsidx[i] << " ";
        for (int j = 0; j < 10; j++)
        {
            cout << intsdata[i][j] << " ";
        }
        for(int j = 0; j < 2; j++)
        {
            cout << intstarget[i][j] << " ";
        }
        cout << endl;
    }

    #endif
}

/**
 * Function to find the normalizing factor for the training data (min and max of each attribute)
 * - normalizer: the result found
 * - data: data
 * */
void find_normalizer(vector<vector<float> >& normalizer, vector<vector<float> > data)
{
    // normalizer.clear();
    //find min and max of each attribute 
    for(int i = 0; i < data[0].size(); i++)
    {
        float min = 100.0f;
        float max = -100.0f;
        // cout << 1 << endl;
        vector<float> oneatt {min, max};
        normalizer.push_back(oneatt);
    }
    

    for(int i = 0; i < data.size(); i++)
    {
        for(int j = 0; j < data[i].size(); j++)
        {
            if (data[i][j] < normalizer[j][0])
            {
                normalizer[j][0] = data[i][j];
            }
            if(data[i][j] > normalizer[j][1])
            {
                normalizer[j][1] = data[i][j];
            }
        }
    }
}

/**
 * Function to normalize the data
 * */
vector<vector<float> > normalize_data(vector<vector<float> > data, vector<vector<float> > normalizer)
{
    vector<vector<float> > ndata;
    for(int i = 0; i < data.size(); i++)
    {
        vector<float> onendata;
        for(int j = 0; j < data[i].size(); j++)
        {
            float natt = (data[i][j] - normalizer[j][0])/(normalizer[j][1] - normalizer[j][0]);
            onendata.push_back(natt);
        }
        ndata.push_back(onendata);
    }
    return ndata;
}


void k_fold_cup()
{
    vector<int> tridx;
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;

    read_trdata(string(CUP_ROOT) + string(TRFILE), tridx, trdata, trtarget);

    //preprocessing 
    preprocess(trtarget);

    //create a network 
    int n_in = 11; // 10 attributes + 1 bias term
    int n_out = 2; // 2 outputs 
    vector<int> hid{20, 10, 4};
    MLP* mlp = new MLP(n_in, n_out, hid.size(), hid, "gauss", "sigmoid");

    float res = kfold_validation(mlp, 2, 2, trdata, trtarget, 5, 2000, 0.01f, 0.3f, 0.0001f, 0.0001f, scaler);
    cout << "K fold res = " << res << endl;


    //reset and retrain with all the training data
    mlp->reset_mlp();
    mlp->train_batch(trdata, trtarget, 0.01f, 2000, 0.3f, 0.0001f, 0.0001f);

    //read test data and run the trained model with the test 

    vector<int> intsidx;
    vector<vector<float> > intsdata;
    vector<vector<float> > intstarget;

    read_trdata(string(CUP_ROOT) + string(INTSFILE), intsidx, intsdata, intstarget);

    float intsres = regression_evaluate(mlp, intsdata, intstarget, scaler);

    cout << "Internal test res = " << intsres << endl;
    delete mlp;

}

/**
 * Just a function to get some strange result from strange models
 * */
void screen_demo(float eta, float alfa, float lambda)
{    
    vector<int> tridx; 
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;

    vector<int> intsidx;
    vector<vector<float> > intsdata;
    vector<vector<float> > intstarget;

    read_trdata(string(CUP_ROOT)+string(TRFILE), tridx, trdata, trtarget);
    read_trdata(string(CUP_ROOT)+string(INTSFILE), intsidx, intsdata, intstarget);

    // normalize all the target to the range [0, 1]
    preprocess(trtarget);
    preprocess(intstarget);


    //create a network 
    int n_in = 11; // 10 attributes + 1 bias term
    int n_out = 2; // 2 outputs 
    vector<int> hid{16, 8, 4, 2};
    MLP* mlp = new MLP(n_in, n_out, hid.size(), hid, "gauss", "linear");

    float trmse, tsmse, trmee, tsmee;

    int iter = 0; 
    do
    {
        iter++;
        mlp->train_batch(trdata, trtarget, 0.001f, 1, eta, alfa, lambda);
        // mlp->train_stochastic(trdata, trtarget, 0.001f, 1, eta, alfa, lambda);
        trmse = mlp->compute_sum_square_multithreads(trdata, trtarget);
        tsmse = mlp->compute_sum_square_multithreads(intsdata, intstarget);
        trmee = regression_evaluate(mlp, trdata, trtarget, scaler);
        tsmee = regression_evaluate(mlp, intsdata, intstarget, scaler);
        if (iter%10 == 0)
        {
            cout << iter << "\t" << trmse << "\t" << tsmse << "\t" << trmee << "\t" << tsmee << endl;
        }
    } while (iter < 5000);
    cout << regression_evaluate(mlp, intsdata, intstarget, scaler);
    regression_to_file(mlp, "res.csv", intsdata, scaler);
    delete mlp;
    
}

/**
 * Function to do the grid search
 * the searching range for each hyper parameters is declare inside the code 
 * also the model is inside the code
 * */
vector<float> grid_search_cup ()
{
    //save result to file
    ofstream file ("grid_search.csv");

    //load the training set
    vector<int> tridx;
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;

    read_trdata(string(CUP_ROOT) + string(TRFILE), tridx, trdata, trtarget);

    //preprocessing 
    preprocess(trtarget);

    //create a network 
    int n_in = 11; // 10 attributes + 1 bias term
    int n_out = 2; // 2 outputs 

    vector<int> hid{16, 12, 12, 8, 4};
    MLP* mlp = new MLP(n_in, n_out, hid.size(), hid, "gauss", "linear");

    vector<float> etas {0.37f, 0.41f, 0.45f};
    vector<float> alfas {0.008, 0.011, 0.015};
    // vector<float> alfas{0.0f};
    vector<float> lambdas {0.000095, 0.00011, 0.00015};
    
    float besres = 10000.0f;
    vector<float> besparas;

    for(int e = 0; e < etas.size(); e++)
    {
        for(int a = 0; a < alfas.size(); a++)
        {
            for(int l = 0; l < lambdas.size(); l++)
            {
                float res = kfold_validation(mlp, 2, 2, trdata, trtarget, 5, 5000, 0.0001f, etas[e], alfas[a], lambdas[l], scaler);
                stringstream ss;
                ss << "e=" <<etas[e] << "-a=" << alfas[a] << "-l=" << lambdas[l] << "\t" << res << endl;
                file << ss.str();
                cout << ss.str();
                if(res < besres)
                {
                    besres = res;
                    besparas.clear();
                    besparas.push_back(etas[e]);
                    besparas.push_back(alfas[a]);
                    besparas.push_back(lambdas[l]);
                    besparas.push_back(besres);
                }

            }
        }
    }
    delete mlp;
    return besparas;
}

/**
 * Just to separated sets  
 * */
void run_just_once()
{
    vector<int> alltridx;
    vector<vector<float> > alltrdata;
    vector<vector<float> > alltrtarget;

    read_trdata(string(CUP_ROOT)+ string(ALLTRFILE), alltridx, alltrdata, alltrtarget);

    vector<int> tridx;
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;

    vector<int> intsidx;
    vector<vector<float> > intsdata;
    vector<vector<float> > intstarget;

    split_sets(0.2f, alltridx, alltrdata, alltrtarget, 
                tridx, trdata, trtarget, intsidx, intsdata, intstarget);

}

/**
 * Function to train the final model
 * */
void final_train()
{
    //load the training set
    vector<int> tridx;
    vector<vector<float> > trdata;
    vector<vector<float> > trtarget;

    read_trdata(string(CUP_ROOT) + string(TRFILE), tridx, trdata, trtarget);
    //preprocess the target
    preprocess(trtarget);

    //create a network 
    int n_in = 11; // 10 attributes + 1 bias term
    int n_out = 2; // 2 outputs 

    vector<int> hid{16, 12, 8, 8, 4};
    MLP* mlp = new MLP(n_in, n_out, hid.size(), hid, "gauss", "linear");

    train_with_early_stopping(mlp, 2, 0.15, 0.00001f, trdata, trtarget, 0.21, 0.0011, 9.5e-5, "third.model", scaler, "third.csv");

    //test the internal test set to get the final result
    vector<int> intsidx;
    vector<vector<float> > intsdata;
    vector<vector<float> > intstarget;

    read_trdata(string(CUP_ROOT)+string(INTSFILE), intsidx, intsdata, intstarget);
    //preprocess
    preprocess(intstarget);

    float res = regression_evaluate(mlp, intsdata, intstarget, scaler);
    cout << "Internal test result: " << res << endl;

    delete mlp;
}
 
/**
 * Function to load trained model and test with the test set 
 * - modelname: name of the model
 * - testname: name of the test 
 * - outname: name of the output file 
 * */
void load_and_test(string modelname, string testname, string outname)
{
    //load the model 
    MLP * mlp = new MLP("second.model");

    vector<int> intsidx;
    vector<vector<float> > intsdata;
    vector<vector<float> > intstarget;

    read_trdata(testname, intsidx, intsdata, intstarget);
    //preprocess
    preprocess(intstarget);

    regression_to_file(mlp, outname, intsdata, scaler);

}

int main(int argc, char** argv)
{
    // test_unit();
    srand(time(NULL));
    final_train();

    // string testname = string(CUP_ROOT) + string(INTSFILE);

    // load_and_test("second.model", testname);


    // vector<float> gsres = grid_search_cup();
    // cout <<"Best:" << "e=" <<gsres[0] << ",a=" << gsres[1] << ",l=" << gsres[2] << "\t" << gsres[3] << endl;

    return 0;
}