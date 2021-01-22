#ifndef UTILITIES_H
#define UTILITIES_H
/**
 * File with all the utilities functions
 * */
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
#include "Unit.h"

using namespace std;


/**
 * Function to validate binary classify problem
 * data: raw input data without bias term 
 * target: expected output 
 * return the ratio of the number of correct answers over the total number of data 
 * */
float binary_classify_evaluate(MLP* mlp, vector<vector<float> > data, vector<vector<float> >target)
{
    //check size compatible
    if(data.size() != target.size())
    {
        cerr << "Data and target size are incompatible";
        return -1.0f;
    }
    int n_corr = 0;
    for (int i = 0; i < data.size(); i++)
    {
        vector<float> bias_sample = data[i];
        int cls = int(target[i][0]);

        bias_sample.push_back(1.0f); // add bias term 
        vector<float> out = mlp->get_output(bias_sample);
        int res;
    
        if (mlp->v_outputs[0]->act == "sigmoid")
        {
            res = (out[0] >= 0.5) ? 1:0;
        }
        if (mlp->v_outputs[0]->act == "tanh")
        {
            res = (out[0] >= 0.0f) ? 1:0;
        }
        // cout << "True class: " << cls << " - " << out[0] << endl;
        if (res == cls)
        {
            n_corr += 1;
        }
    }
    return (float)n_corr/(float)data.size();
}

/**
 * Function to evaluate the regression result
 * - mlp: model 
 * - data: data
 * - target: target (normalized)
 * - scaler: scaling factor of the output format: minout1 maxout1 minout2 maxout2...
 * */
float regression_evaluate(MLP* mlp, vector<vector<float> > data, vector<vector<float> > target, vector<float> scaler = vector<float> ())
{
    if(data.size() != target.size())
    {
        cerr << "Data and target size are not compatible";
        return -1.0f;
    }
    int numdata = data.size();
    float toterr = 0.0f;
    for(int i = 0; i < data.size(); i++)
    {
        vector<float> bias_sample = data[i];
        bias_sample.push_back(1.0f); // add bias term 
        vector<float> out = mlp->get_output(bias_sample);

        if(!scaler.empty()) // there is a scaling factors vector, do the "decode" real = normalized * (max - min) + min
        {
            // cout << "decoding"<< endl;
            vector<float> real_out;
            for(int j = 0; j < out.size(); j++)
            {
                float realval;
                realval = out[j]*(scaler[j*2 + 1] - scaler[j*2]) + scaler[j*2];
                real_out.push_back(realval);
                #ifdef SHOW
                cout << realval << "\t";
                #endif
            }
            vector<float> real_tar;
            for(int j = 0; j < target[i].size(); j++)
            {
                float realval;
                realval = target[i][j]*(scaler[j*2 + 1] - scaler[j*2]) + scaler[j*2];
                real_tar.push_back(realval);
                #ifdef SHOW
                cout << realval << "\t";
                #endif
            }
            #ifdef SHOW
            cout << endl;
            #endif 

            float oneerr = 0.0f;
            for(int j = 0; j < out.size(); j++)
            {
                oneerr += (real_out[j] - real_tar[j])*(real_out[j] - real_tar[j]);
            }
            toterr += sqrt(oneerr);

        }
        else // no "decode"
        {
            // cout << "no decode" << endl;
            float oneerr = 0.0f;
            for(int j = 0; j < out.size(); j++)
            {
                oneerr += (out[j] - target[i][j])*(out[j] - target[i][j]);;
            }
            toterr += sqrt(oneerr);
        }
        
    }
    return toterr/(float) numdata;
}


/**
 * Run the regression model and flush result to file.
 * -mlp: model
 * -filename: expected name of the result file 
 * -data: input data
 * -scaler: normalizing factor 
 * */
void regression_to_file(MLP* mlp, string filename, vector<vector<float> > data, 
                        vector<float> scaler = vector<float> ())
{
    ofstream ofile (filename);
    stringstream header;
    header << "# " << "Giang Pham \t Sammat Fareed" << endl;
    header << "# " << "The noobs" << endl;
    header << "# ML-CUP20" << endl;
    header << "# Submission Date: 05/01/2021" << endl;
    ofile << header.str();
    for(int i = 0; i < data.size(); i++)
    {
        vector<float> bias_sample = data[i];
        bias_sample.push_back(1.0f); // add bias term 
        vector<float> out = mlp->get_output(bias_sample);
        if(!scaler.empty()) // there is a scaling factors vector, do the "decode" real = normalized * (max - min) + min
        {
            // cout << "decoding"<< endl;
            stringstream real_out;
            real_out << i+1 << ",";
            for(int j = 0; j < out.size(); j++)
            {
                float realval;
                realval = out[j]*(scaler[j*2 + 1] - scaler[j*2]) + scaler[j*2];
                real_out << realval;
                if (j != out.size() - 1)
                {
                    real_out << ",";
                }
            }
            real_out << endl;
            ofile << real_out.str();
        }
        else
        {
            stringstream real_out;
            for(int j = 0; j < out.size(); j++)
            {
                real_out << out[j];
                if (j != out.size() - 1)
                {
                    real_out << ",";
                }
                
            }
            real_out << endl;
            ofile << real_out.str();
        }

    }
    ofile.close();
}

/**
 * Function kfold validation, get an array of data, saparate into k sets, 
 * respectively use one set as validation, k-1 remained sets as training data
 * + mode: learning mode, 1 for stochastic, 2 for batch, 3 for mini batch, comming soon
 * + task: 1 for binary classify, 2 for regression
 * + data: full data set 
 * + targets: correponding targets of the data 
 * + k: number of folds
 * + num_iter: number of epoch
 * + ep: epsilon 
 * + eta: leaning rate 
 * + alfa: momentum parameter 
 * + lambda: weight decay parameter 
 * + scaler: vector contains scaling factor of the output data (in case of regression) - optional
 * + return the average validation result of the model 
 * */
float kfold_validation(MLP* mlp, int mode, int task, 
                       vector<vector<float> >data, vector<vector<float> >target,
                       int k, int num_iter, float ep, 
                       float eta, float alfa, float lambda, 
                       vector<float> scaler = vector<float>())
{
    //check size of data and target 
    #ifdef DEBUG 
    cout << "Size of data set " << data.size() << endl;
    #endif 
    
    if (data.size() != target.size())
    {
        cerr << "Function kfold-validation: Size of data and target are incompatible";
        return -1.0f;
    }
    int foldsize = (int)data.size()/k;

    vector<int> index_shuffle;

    //shuffle indexes of the data 
    for(int i = 0; i < data.size(); i++)
    {
        index_shuffle.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(index_shuffle.begin(), index_shuffle.end(), g);

    float totres = 0.0f;

    float tottrres = 0.0f;
    float trmee = 0.0f;

    for (int i = 0; i < k; i++) //each fold works as a validation set once 
    {
        int beg_val_fold; 
        int end_val_fold;
        beg_val_fold = i*foldsize;
        if (i == k-1)
        {
            end_val_fold = data.size();
        }
        else
        {
            end_val_fold = (i+1)*foldsize;
        }
        vector<vector<float> > trdata;
        vector<vector<float> > vdata;

        vector<vector<float> > trtarget;
        vector<vector<float> > vtarget;
        for(int j = 0; j < index_shuffle.size(); j++)
        {
            if(j >= beg_val_fold && j < end_val_fold) // samples go to validation set
            {
                vdata.push_back(data[index_shuffle[j]]);
                vtarget.push_back(target[index_shuffle[j]]);
            }
            else // samples go to training set 
            {
                trdata.push_back(data[index_shuffle[j]]);
                trtarget.push_back(target[index_shuffle[j]]);
            }
        }
        #ifdef DEBUG
        cout << "Traing data size: " << trdata.size() << endl;
        cout << "Validation data size: " << vdata.size() << endl;
        #endif 
        float trres;
        switch (mode)
        {
        case 2: // batch learning
            trres = mlp->train_batch(trdata, trtarget, ep, num_iter, eta, alfa, lambda);
            break;
        default: // stochastic learning
            trres = mlp->train_batch(trdata, trtarget, ep, num_iter, eta, alfa, lambda);
            break;
        }
        tottrres += trres;

        //test with validation set 
        float kres = 0.0f;
        float ktrmee = 0.0f;
        switch (task)
        {
        case 1: // for binary classify 
            totres += binary_classify_evaluate(mlp, vdata, vtarget);
            // cout << "Result for " << i << "th fold: " << binary_classify_evaluate(mlp, vdata, vtarget) << endl;
            break;
        case 2: // for multi regression 
            kres = regression_evaluate(mlp, vdata, vtarget, scaler);
            ktrmee = regression_evaluate(mlp, trdata, trtarget, scaler);

            totres += kres;
            trmee += ktrmee;
            cout << "Fold " << i << "-th: Train mse" << trres << "\t val mee: " << kres << "\t train mee: " << ktrmee << endl;
            break;
        default:
            totres += regression_evaluate(mlp, vdata, vtarget, scaler);
            break;
        }
        
        //reset the model to retrain
        mlp->reset_mlp();
    }
    cout << "Val MEE: " << totres/k << "\t Train MEE: " << trmee/k << "\t Train MSE: " << tottrres/k << endl;
    return totres/k;
}

/**
 * Function grid search to find the best hyperparameters with fixed model: eta, lambda, alfa, number of epoch 
 * + mlp: model
 * + mode: learning mode, 1 for stochastic, 2 for batch, 3 for mini batch (comming soon)
 * + task: 1 for binary classify, 2 for regression 
 * + k: number of folds 
 * + data: raw input data without bias term 
 * + target: corresponding target of the input data 
 * + eta_values: list of eta values to search 
 * + alfa_values: list of alfa values to search 
 * + lambda_values: list of lambda values to search 
 * + scaler: scaling factor (optional) for normalization
 * - return the vector of best combination and corresponding validation result
 * */
vector<float> grid_search (MLP* mlp, int mode, int task, int k, vector<vector<float> > data, vector<vector<float> > target, 
                  vector<float> eta_values, vector<float> alfa_values, vector<float> lambda_values, 
                  vector<float> scaler = vector<float> ()) 
{
    float best_res = 0.0f;
    vector <float> best_combination;
    for(int e = 0; e < eta_values.size(); e++)
    {
        for(int l = 0; l < lambda_values.size(); l++)
        {
            for(int a = 0; a < alfa_values.size(); a++)
            {
                float res = kfold_validation(mlp, mode, task, data, target, k, 2000, 0.01f,  
                                         eta_values[e], alfa_values[a], lambda_values[l], scaler);
                cout << eta_values[e] << "\t" << alfa_values[a] << "\t" << lambda_values[l] << 
                "\t" << res << endl;
                if (res > best_res)
                {
                    best_res = res;
                    best_combination.clear();
                    best_combination.push_back(eta_values[e]);
                    best_combination.push_back(alfa_values[a]);
                    best_combination.push_back(lambda_values[l]);
                    best_combination.push_back(res);
                }
            }
        }
    }
    return best_combination;
}

/**
 * Train the model with early stopping technique
 * - mlp: model to train 
 * - mode: 1 for stochastic, 2 for batch, use 2
 * - ratio: split ratio for training and validation set 
 * - ep: epsilon to stop
 * - data: input data 
 * - target: expected output
 * - eta, alfa, lambda: training parameters 
 * - modelname: name of the file to save the trained model
 * - scaler: normalizing factor
 * - filename: name of the file to save the training information 
 * */
void train_with_early_stopping (MLP* mlp, int mode, float ratio, float ep,
                                vector<vector<float> >data, vector<vector<float> >target,
                                float eta, float alfa, float lambda, 
                                string modelname,
                                vector<float> scaler = vector<float> (), 
                                string filename = "training-curve.csv")
{
    auto t1 = std::chrono::high_resolution_clock::now();
    //check size of data and target 
    #ifdef DEBUG 
    cout << "Size of data set " << data.size() << endl;
    #endif 
    

    if (data.size() != target.size())
    {
        cerr << "Size of data and target are incompatible";
        return;
    }

    vector<int> index_shuffle;

    //shuffle indexes of the data 
    for(int i = 0; i < data.size(); i++)
    {
        index_shuffle.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(index_shuffle.begin(), index_shuffle.end(), g);

    int stopsize = (int)data.size()*ratio;


    vector<vector<float> > trdata; 
    vector<vector<float> > trtarget;

    vector<vector<float> > stopdata;
    vector<vector<float> > stoptarget;
    for(int i = 0; i < index_shuffle.size(); i++)
    {
        if (i < stopsize)
        {
            stopdata.push_back(data[index_shuffle[i]]);
            stoptarget.push_back(target[index_shuffle[i]]);
        }
        else{
            trdata.push_back(data[index_shuffle[i]]);
            trtarget.push_back(target[index_shuffle[i]]);
        }
    }
    cout << "Number of validation data: " << stopdata.size() << endl;
    cout << "Number of training data: " << trdata.size() << endl;

    //open file to save the information 
    int n_iter = 0;
    int max_iter = 8000;
    ofstream inf(filename);
    if (mode == 1) //stochastic 
    {
        int incount = 0;
        
        float oldres = regression_evaluate(mlp, stopdata, stoptarget, scaler);
        
        for(int i = 0; i < max_iter; i++)
        {
            float trres = mlp->train_stochastic(trdata, trtarget, ep, 1, eta, alfa, lambda);
            float res = regression_evaluate(mlp, stopdata, stoptarget, scaler);
            if (res > oldres)
            {
                incount ++;
            }
            else
            {
                incount = 0;
            }
            //stop if stop score increase in 10 continuous iters or if the result increase dramatically
            // if (incount > 10 || res - oldres > 0.2f) break; 
            oldres = res;

            stringstream ss;
            ss.clear(); ss << i << "\t" << trres << "\t" << res << endl;
            cout << i << "\t" << trres << "\t" << res << endl;
            inf << ss.str();
            n_iter ++;
        } 
    }
    else // batch
    {
        int incount = 0;
        float oldres = regression_evaluate(mlp, stopdata, stoptarget, scaler);
        
        for(int i = 0; i < max_iter; i++)
        {
            float trres = mlp->train_batch(trdata, trtarget, ep, 1, eta, alfa, lambda);
            float res = regression_evaluate(mlp, stopdata, stoptarget, scaler);
            if (res > oldres)
            {
                incount ++;
            }
            else
            {
                incount = 0;
            }
            //stop if stop score increase in 10 continuous iters or if the result increase dramatically
            // if (incount > 10 || res - oldres > 0.2f) break; 
            oldres = res;

            stringstream ss;
            ss.clear(); ss << i << "\t" << trres << "\t" << res << endl;
            cout << i << "\t" << trres << "\t" << res << endl;
            inf << ss.str();
            n_iter ++;
        } 
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Train with: " << n_iter << " epochs in " << duration << endl;
    cout << "Average time: " << (float) duration /(float) n_iter << endl;
    mlp->save_model(modelname);
}



#endif 