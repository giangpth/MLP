#ifndef MLP_H
#define MLP_H

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
#include "Unit.h"

using namespace std;

// #define DEBUG 
// #define INTEGER 
// #define LOG
// #define TEST_UNIT

mutex sum_err_lock;
mutex acc_weight_lock;



class MLP
{
private:   
    int i_n_input_units; // number of input units, remember to consider bias term yet, ex: input has 3 dims so i_n_input_units = 4 because add 1 bias term
    int i_n_output_units; // number of output units, output dimension 
    int i_n_hidden_layers; // number of hidden layers 
    vector<int> v_n_hidden_units; // array of number of hidden units of each hidden layer 

    vector<Unit*> v_inputs;
    vector< vector<Unit*> > vv_hidden_units;

public:
    vector<Unit*> v_outputs; //need output units to be public to get access sometime  ß
    /**
     * Constructor
     * _in: number of input units
     * _out: number of ouput units 
     * _hid: number of hidden layers 
     * _hid_units: list of number of units at each hidden layer
     * _act_hid: name of activation function for the hidden layer
     * _act_out: name of activation function for the ouput layer
     * */
    MLP(int _in, int _out, int _hid, vector<int> _hid_units, string _act_hid, string _act_out) //for fully connected 
    {
        i_n_input_units = _in;
        i_n_output_units = _out; 
        i_n_hidden_layers = _hid; 
        v_n_hidden_units = _hid_units;

        #ifdef DEBUG 
        cout << "Create mlp with " << i_n_input_units << " input units, ";
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            cout << v_n_hidden_units[i] << " at hidden layer " << i << " ";
        }
        cout << " and  " << i_n_output_units << " ouput units" << endl << endl;
        #endif 

        //create input units
        #ifdef DEBUG 
        cout << "Create " << i_n_input_units << " input units" << endl;
        #endif

        for (int i = 0; i < i_n_input_units; i++)
        {
            v_inputs.push_back(new Unit(0, v_n_hidden_units[0], "linear")); // input units have zero input, activation function needs to be linear
            v_inputs[i]->random_weight();
        }

        // create hidden layers
        #ifdef DEBUG 
        cout << "Create " << i_n_hidden_layers << " hidden layer" << endl;
        #endif

        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            vector<Unit*> layer;

            int n_input;
            if (i == 0) {n_input = i_n_input_units;}
            else {n_input = v_n_hidden_units[i-1];}

            int n_output;
            if (i >= i_n_hidden_layers - 1) {n_output = i_n_output_units;}
            else {n_output = v_n_hidden_units[i+1];}

            #ifdef DEBUG 
            cout << "Create " << v_n_hidden_units[i] << " hidden units for hidden layer " << i << endl;
            #endif

            for (int j = 0; j < v_n_hidden_units[i]; j++)
            {
                layer.push_back(new Unit(n_input, n_output, _act_hid));
                layer[j]->random_weight();
            }
            vv_hidden_units.push_back(layer);
        }

        //create output layers
        #ifdef DEBUG 
        cout << "Create " << i_n_output_units << " output units" << endl;
        #endif

        for(int i = 0; i < i_n_output_units; i++)
        {
            v_outputs.push_back(new Unit(v_n_hidden_units[i_n_hidden_layers - 1], 0, _act_out));
            v_outputs[i]->random_weight();
        }

        //create fully connection 
        create_fully_connection();

        #ifdef TEST_UNIT // print to check the connection 
        cout << "Initial weight: ";
        print_connection();
        #endif
    }

    /**
     * Constructor, reconstruct saved model from file 
     * filename: name of the file
     * */
    MLP(string filename)
    {
        ifstream file (filename);
        if(file.is_open())
        {
            //read first line
            file >> i_n_input_units;
            file >> i_n_output_units;
            file >> i_n_hidden_layers;
            // cout << i_n_input_units << " " << i_n_output_units << " " << i_n_hidden_layers << endl;

            //read second line 
            for(int i = 0; i < i_n_hidden_layers; i++)
            {
                int tem;
                file >> tem;
                v_n_hidden_units.push_back(tem);
                // cout << tem << " ";
            }
            // cout << endl;

            //create input units and load saved weights 
            for (int i = 0; i < i_n_input_units; i++)
            {
                v_inputs.push_back(new Unit(0, v_n_hidden_units[0], "linear")); 
                //load saved
                for(int j = 0; j < v_n_hidden_units[0]; j++)
                {
                    float tem;
                    file >> tem;
                    v_inputs[i]->vf_out_weights.push_back(tem);
                    // cout << tem << " ";
                }
                // cout << endl;
            }

            //create hidden layers
            for (int i = 0; i < i_n_hidden_layers; i++)
            {
                vector<Unit*> layer;

                int n_input;
                if (i == 0) {n_input = i_n_input_units;}
                else {n_input = v_n_hidden_units[i-1];}

                int n_output;
                if (i >= i_n_hidden_layers - 1) {n_output = i_n_output_units;}
                else {n_output = v_n_hidden_units[i+1];}

                #ifdef DEBUG 
                cout << "Create " << v_n_hidden_units[i] << " hidden units for hidden layer " << i << endl;
                #endif

                for (int j = 0; j < v_n_hidden_units[i]; j++)
                {
                    //get the name of the activation function for this unit
                    string act_name;
                    file >> act_name;
                    
                    //create unit
                    layer.push_back(new Unit(n_input, n_output, act_name));

                    //load saved weight 
                    for(int k = 0; k < n_output; k++)
                    {
                        float tem;
                        file >> tem;
                        layer[j]->vf_out_weights.push_back(tem);
                        // cout << tem << " ";
                    }
                    // cout << endl;
                }
                // cout << endl;
                vv_hidden_units.push_back(layer);
            }

            //create output layers
            for(int i = 0; i < i_n_output_units; i++)
            {
                //get the name of the activation function for this unit
                string act_name;
                file >> act_name;

                //create new unit
                v_outputs.push_back(new Unit(v_n_hidden_units[i_n_hidden_layers - 1], 0, act_name));
            }
            
            //create fully connection 
            create_fully_connection();

        }
        else 
        {
            cerr << "Cannot open file " << filename << endl;
        }
    }

    /**
     * Function create fully connect of the net work
     * */
    void create_fully_connection()
    {
        if (i_n_hidden_layers > 0)
        {
            //connect input with the first hidden layer
            for (int i = 0; i < v_inputs.size(); i++)
            {
                v_inputs[i]->set_feed_to_list(vv_hidden_units[0]);
            }
            //connect hidden layers together 
            for (int i = 0; i < i_n_hidden_layers; i++)
            {
                for(int j = 0; j < vv_hidden_units[i].size(); j++)
                {
                    //feed forward 
                    if (i >= i_n_hidden_layers - 1) // connect last hidden layer to output 
                    {
                        vv_hidden_units[i][j]->set_feed_to_list(v_outputs);
                    }
                    else // connect this hidden layer to next hidden layer
                    {
                        vv_hidden_units[i][j]->set_feed_to_list(vv_hidden_units[i+1]);
                    }
                    //back prop
                    if(i == 0)
                    {
                        vv_hidden_units[i][j]->set_back_prop_list(v_inputs);   
                    }
                    else
                    {
                        vv_hidden_units[i][j]->set_back_prop_list(vv_hidden_units[i-1]);
                    }
                } 
            }
            for(int i = 0; i < v_outputs.size(); i++)
            {
                v_outputs[i]->set_back_prop_list(vv_hidden_units[i_n_hidden_layers-1]);
            }
        }
        else // connect input to output 
        {
            for(int i = 0; i < v_inputs.size(); i++)
            {
                v_inputs[i]->set_feed_to_list(v_outputs);
            }
            for(int i = 0; i < v_outputs.size(); i++)
            {
                v_outputs[i]->set_back_prop_list(v_inputs);
            }
        }

    }

    /**
     * Function to print all the weights of the net work
     * */
    void print_connection()
    {
        cout << "Print connection to test" << endl;
        //feed forward 
        for(int i = 0; i < v_inputs.size(); i++)
        {
            for (int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                
                cout << v_inputs[i]->vf_out_weights[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        for (int i = 0 ; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                for (int k  = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    cout << vv_hidden_units[i][j]->vf_out_weights[k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;

        //back prop weight 
        /*
        for (int i = 0; i < v_outputs.size(); i++)
        {
            for (int j = 0; j < v_outputs[i]->v_feed_from.size(); j++)
            {
                cout << v_outputs[i]->v_feed_from[j]->vf_out_weights[i] << " ";
            }
            cout << endl;
        }
        cout << endl;

        for (int i = i_n_hidden_layers - 1; i >= 0; i--) // i-th hidden layer 
        {  
            for (int j = 0; j < vv_hidden_units[i].size(); j++) // j-th unit of i-th hidden layer 
            {
                for (int k = 0; k < vv_hidden_units[i][j]->v_feed_from.size(); k ++) //j-th unit of i-th hidden layer  back to - k-th unit of previous layer
                {
                    cout << vv_hidden_units[i][j]->v_feed_from[k]->vf_out_weights[j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
        */ 
    }

    /**
     * Function to feed one sample through the net 
     * compute and store in the network's units all the values 
     * corresponded to this sample (f_value, f_dernet)
     * vf_one_input: one sample to feed through the network
     * */
    void feed_forward_stochastic(vector<float> vf_one_input)
    {   
        //check size compatiable
        if (vf_one_input.size() != i_n_input_units)
        {
            cerr << "Size of input is incompatible" << endl;
            return;
        }
        //feed to the input layer 
        for(int i = 0; i < i_n_input_units; i++)
        {
            v_inputs[i]->f_value = v_inputs[i]->act_func->compute(vf_one_input[i]);
            v_inputs[i]->f_dernet = v_inputs[i]->act_func->computeDer(vf_one_input[i]);
        }
        //feed to the hidden layers 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                float tem = 0.0f; // net value, sum of all the incomming inputs with corresponding weight 
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_from.size(); k++)
                {
                    tem += vv_hidden_units[i][j]->v_feed_from[k]->vf_out_weights[j]*vv_hidden_units[i][j]->v_feed_from[k]->f_value;
                }
                vv_hidden_units[i][j]->f_value = vv_hidden_units[i][j]->act_func->compute(tem);
                vv_hidden_units[i][j]->f_dernet = vv_hidden_units[i][j]->act_func->computeDer(tem);
            }
        }
        //feed to the output layer 
        for (int i = 0; i < i_n_output_units; i++)
        {
            float tem = 0; // net value
            for(int j = 0; j < v_outputs[i]->v_feed_from.size(); j++)
            {
                tem += v_outputs[i]->v_feed_from[j]->vf_out_weights[i]*v_outputs[i]->v_feed_from[j]->f_value;
                //cout << v_outputs[i]->v_feed_from[j]->vf_out_weights[i] << " * " <<  v_outputs[i]->v_feed_from[j]->f_value << endl;
            }
            v_outputs[i]->f_value = v_outputs[i]->act_func->compute(tem);
            v_outputs[i]->f_dernet = v_outputs[i]->act_func->computeDer(tem);
            // cout << v_outputs[i]->f_value << " " << v_outputs[i]->f_dernet << endl;
        }

        #ifdef DEBUG
        cout << "After feed forward, value of units: " << endl;
        print_unit_value(1);
        #endif

    }

    /**
     * Function to send back the error signal of one sample to all units of the networks
     * This function calculate the f_delta value for all the units
     * vf_one_target: the correct (expected) output of the corresponding sample
     * */
    void back_prop_stochastic(vector<float> vf_one_target)
    {
        //check size compatible 
        if (vf_one_target.size() != i_n_output_units)
        {
            cerr << "Size of ouput is incompatible " << vf_one_target.size() << " and " << i_n_output_units << endl;
            return;
        }
        //from the ouput layer
        for(int i  = 0; i < i_n_output_units; i++)
        {
            v_outputs[i]->f_delta = (vf_one_target[i] - v_outputs[i]->f_value)*v_outputs[i]->f_dernet;
        }

        //back prop to  hidden layer
        for(int i = i_n_hidden_layers - 1; i >= 0; --i)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                float tem = 0.0f;
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    tem += vv_hidden_units[i][j]->vf_out_weights[k]*vv_hidden_units[i][j]->v_feed_to[k]->f_delta;
                    // #ifdef TEST_UNIT
                    // cout << vv_hidden_units[i][j]->vf_out_weights[k] << " * " << 
                    //         vv_hidden_units[i][j]->v_feed_to[k]->f_delta << " + ";
                    // #endif
                }
                vv_hidden_units[i][j]->f_delta = tem*vv_hidden_units[i][j]->f_dernet;
                
                // #ifdef TEST_UNIT
                // cout << " = " << tem;
                // cout << tem << " * " << vv_hidden_units[i][j]->f_dernet << " = " << vv_hidden_units[i][j]->f_delta << endl;
                // #endif
            }
        }
        
        //no need back prop to input layer
        #ifdef DEBUG
        cout << "After back prop, delta of units: " << endl; 
        print_unit_value(3);
        #endif
    }

    /**
     * Function to update weight after feed forward and back prop a sample.
     * Only call this function after feed_forward_stochastic and back_prop_stochastic 
     * eta: learing rate
     * lambda: weight decay (regularization) parameter
     * alfa: momentum parameter
     * */
    void update_weight_stochastic(float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //input layer 
        for(int i = 0; i < i_n_input_units; i++)
        {
            for (int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                float big_delta = eta*v_inputs[i]->v_feed_to[j]->f_delta*v_inputs[i]->f_value +
                alfa*v_inputs[i]->vf_momentum[j] - lambda*v_inputs[i]->vf_out_weights[j];
                
                //store old delta for momentum mode
                v_inputs[i]->vf_momentum[j] = big_delta;

                //update weight 
                v_inputs[i]->vf_out_weights[j] = v_inputs[i]->vf_out_weights[j] + big_delta;
                // #ifdef TEST_UNIT
                // cout << eta << " * " << v_inputs[i]->v_feed_to[j]->f_delta  << "  *  " << v_inputs[i]->f_value << " + "
                // << alfa << " * " << v_inputs[i]->vf_momentum[j] << " - " << lambda << " * " << v_inputs[i]->vf_out_weights[j] 
                // << " = " << big_delta << endl;
                // #endif
            }
        }

        //hidden layers 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    float big_delta = eta*vv_hidden_units[i][j]->f_value*vv_hidden_units[i][j]->v_feed_to[k]->f_delta + 
                    alfa*vv_hidden_units[i][j]->vf_momentum[k] - lambda*vv_hidden_units[i][j]->vf_out_weights[k];

                    //store the old delta for the momentum mode
                    vv_hidden_units[i][j]->vf_momentum[k] = big_delta;

                    //update weight
                    vv_hidden_units[i][j]->vf_out_weights[k] = vv_hidden_units[i][j]->vf_out_weights[k] + big_delta;

                    // #ifdef TEST_UNIT
                    // cout << eta << " * " << vv_hidden_units[i][j]->f_value <<" * " << vv_hidden_units[i][j]->v_feed_to[k]->f_delta << " + "
                    // << alfa << " * " << vv_hidden_units[i][j]->vf_momentum[k]<< " - " << lambda << " * " << vv_hidden_units[i][j]->vf_out_weights[k]
                    // << " = " <<  big_delta << endl;
                    // #endif 
                }
            }
        }

        #ifdef TEST_UNIT
        cout << "Function update_weight_stochastic: ";
        print_connection();
        #endif
    }

    /**
     * Function to learn from one sample 
     * data: input data of the sample 
     * target: target of the sample 
     * eta: learning rate 
     * alfa: momentum parameter 
     * lambda: weight decay (regularization) parameter
     * */
    void stochastic_learning(vector<float> data, vector<float> target, float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        feed_forward_stochastic(data);
        back_prop_stochastic(target);
        update_weight_stochastic(eta, alfa, lambda);
    }

    /**
     * Function get output of a sample, use in test process 
     * vf_one_input: sample to test, input with bias term 
     * return: output of the test sample
     * */
    vector<float> get_output(vector<float> vf_one_input)
    {
        //check size compatible 
        if(vf_one_input.size() != i_n_input_units)
        {
            cerr << "Function get_output: input size is incompatible";
            return vector<float> ();
        }
        vector<vector<float> > thisunitsout;

        thisunitsout.push_back(vf_one_input); 

        for(int i = 0; i < i_n_hidden_layers; i++)
        {
            vector <float> v_temout; 
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                float tem = 0.0f;
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_from.size(); k++)
                {
                    tem += vv_hidden_units[i][j]->v_feed_from[k]->vf_out_weights[j]*thisunitsout[i][k];
                }
                v_temout.push_back(vv_hidden_units[i][j]->act_func->compute(tem));
            }
            thisunitsout.push_back(v_temout);
        }

        vector<float> outstem;
        for (int i = 0; i < i_n_output_units; i++)
        {
            float tem = 0;
            for(int j = 0; j < v_outputs[i]->v_feed_from.size(); j++)
            {
                tem += v_outputs[i]->v_feed_from[j]->vf_out_weights[i]*thisunitsout[i_n_hidden_layers][j];
            }
            float oneout = v_outputs[i]->act_func->compute(tem);

            outstem.push_back(oneout);
        }
        return outstem;
    }

    /**
     * Function return error of a sample and also gives back the units' output of the net for this sample
     * vf_one_input: current sample 
     * vf_one_target: current target of current sample 
     * vvf_unitsout: the units' output of this sample to return 
     * vvf_unitsder: derivative of units, to return 
     * */
    float compute_one_error(vector<float> vf_one_input, vector<float> vf_one_target, 
                            vector<vector<float> >& vvf_unitsout, vector<vector<float> >&vvf_unitsder)
    {
        
        //check size compatible 
        if (vf_one_input.size() != i_n_input_units)
        {
            cerr << "Function computer_one_error: input sizes are incompatible";
            return -1.0f;
        }
        if (vf_one_target.size() != i_n_output_units)
        {
            cerr << "Function computer_one_error: output sizes are incompatible";
            return -1.0f;
        }

        //feed into
        vector<vector<float> > thisunitsout; // this is a vector stores the output at each layer of the current input, for multithread 
        vector<vector<float> > thisunitsder; // this is a vector stores the derivative of each unit
        
        thisunitsout.push_back(vf_one_input); 
        thisunitsder.push_back(vector<float> (vf_one_input.size(), 1.0f)); // derivatives of first layer's units are all 1.0f
        for(int i = 0; i < i_n_hidden_layers; i++)
        {
            vector <float> v_temout; 
            vector <float> v_temder;
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                float tem = 0.0f;
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_from.size(); k++)
                {
                    tem += vv_hidden_units[i][j]->v_feed_from[k]->vf_out_weights[j]*thisunitsout[i][k];
                    // cout << vv_hidden_units[i][j]->v_feed_from[k]->vf_out_weights[j] << " " << thisunitsout[i][k] << endl;
                }
                v_temout.push_back(vv_hidden_units[i][j]->act_func->compute(tem));
                v_temder.push_back(vv_hidden_units[i][j]->act_func->computeDer(tem));
            }
            thisunitsout.push_back(v_temout);
            thisunitsder.push_back(v_temder);

        }

        float err = 0.0f;
        //output layer 
        vector<float> outstem;
        vector<float> derstem;
        for (int i = 0; i < i_n_output_units; i++)
        {
            float tem = 0;
            for(int j = 0; j < v_outputs[i]->v_feed_from.size(); j++)
            {
                tem += v_outputs[i]->v_feed_from[j]->vf_out_weights[i]*thisunitsout[i_n_hidden_layers][j];
            }
            float oneout = v_outputs[i]->act_func->compute(tem);

            outstem.push_back(oneout);
            derstem.push_back(v_outputs[i]->act_func->computeDer(tem));

            err += (vf_one_target[i] -oneout)*(vf_one_target[i] - oneout);
        }
        thisunitsout.push_back(outstem);
        thisunitsder.push_back(derstem);

        vvf_unitsout = thisunitsout;
        vvf_unitsder = thisunitsder;

        #ifdef TEST_UNIT
        cout << "Print the vvf_unitsout to check" << endl;
        for (int i = 0; i < vvf_unitsout.size(); i++)
        {
            for(int j = 0; j < vvf_unitsout[i].size(); j++)
            {
                cout << vvf_unitsout[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        cout << "Print the vvf_unitsder to check" << endl;
        for (int i = 0; i < vvf_unitsder.size(); i++)
        {
            for(int j = 0; j < vvf_unitsder[i].size(); j++)
            {
                cout << vvf_unitsder[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        #endif 
        return err;
    }

    /**
     * Function to compute sum square error of all the samples
     * data: list of samples
     * target: list of corresponding targets of all the sample in data
     * */
    float compute_sum_square_multithreads(vector<vector<float> > data, vector<vector<float> >target)
    {
        int numdata =  data.size();
        float toterr = 0.0f;
        unsigned int numThreads = std::thread::hardware_concurrency();
        #ifdef DEBUG 
        cout << "Using " << numThreads << " threads to calculate the sum square error" << endl;
        #endif 
        int chunk_size = (int)data.size()/numThreads;
        auto sum_err_chunk = [&] (int cidx)
        {
            float localsum = 0.0f;

            int beg = cidx * chunk_size;
            int end;
            if (cidx >= numThreads - 1) {end = data.size();}
            else {end = (cidx+1)*chunk_size;}
            //compute local sum of error for this chunk 
            for (int i = beg; i < end; i++)
            {
                vector<float> data_with_bias = data[i];
                data_with_bias.push_back(1.0f); // add bias term 1.0f
                vector<vector<float> > vvf_unitsout;
                vector<vector<float> > vvf_unitsder;
                localsum += compute_one_error(data_with_bias, target[i], vvf_unitsout, vvf_unitsder);
            }
            sum_err_lock.lock();
            toterr += localsum;
            sum_err_lock.unlock();
        }; 

        vector<thread> tids;
        for(int i = 0; i < numThreads; i++)
        {
            tids.push_back(thread(sum_err_chunk, i));
        }
        for(int i = 0; i < numThreads; i++)
        {
            tids[i].join();
        }
        return (float)toterr/(float) numdata;
    }

    /**
     * Instead of immediately update weights, 
     * this function accumulate the big_delta of each weight for batch learning 
     * eta: leanring rate 
     * alfa: momentum parameter 
     * lambda: weight decay (regularization) parameter 
     * */
    void accummulate_weight(float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //input layer 
        for(int i = 0; i < i_n_input_units; i++)
        {
            for (int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                float big_delta = eta*v_inputs[i]->v_feed_to[j]->f_delta*v_inputs[i]->f_value +
                alfa*v_inputs[i]->vf_momentum[j] - lambda*v_inputs[i]->vf_out_weights[j];

                //accummulate weight 
                v_inputs[i]->vf_acc_big_delta[j] += big_delta;

                // cout << v_inputs[i]->vf_acc_big_delta[j] << " ";
            }
            // cout << endl;
        }

        //hidden layers 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    float big_delta = eta*vv_hidden_units[i][j]->f_value*vv_hidden_units[i][j]->v_feed_to[k]->f_delta + 
                    alfa*vv_hidden_units[i][j]->vf_momentum[k] - lambda*vv_hidden_units[i][j]->vf_out_weights[k];

                    //update weight
                    vv_hidden_units[i][j]->vf_acc_big_delta[k] += big_delta;

                    // cout << vv_hidden_units[i][j]->vf_acc_big_delta[k] << " ";
                }
                // cout << endl;
            }
            // cout << endl;
        }
    }

    /**
     * Function update the weights with its accumulated value, use in batch learning
     * batch_size: size of batch
     * */
    void update_accummulate_weight(int batch_size)
    {
        //input layer 
        for(int i = 0; i < i_n_input_units; i++)
        {
            for (int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                //store the old big_delta for momentum 
                v_inputs[i]->vf_momentum[j] = v_inputs[i]->vf_acc_big_delta[j]/batch_size;

                //update the weight with the accummulated value
                v_inputs[i]->vf_out_weights[j] += v_inputs[i]->vf_acc_big_delta[j]/batch_size;

                //reset the accummulated value to 0.0f
                v_inputs[i]->vf_acc_big_delta[j] = 0.0f;

            }
        }

        //hidden layers 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    //store the old big_delta for momentum 
                    vv_hidden_units[i][j]->vf_momentum[k] = vv_hidden_units[i][j]->vf_acc_big_delta[k]/batch_size;

                    //update the weight with the accummulated value 
                    vv_hidden_units[i][j]->vf_out_weights[k] += vv_hidden_units[i][j]->vf_acc_big_delta[k]/batch_size;

                    //reset the accummulated value to 0.0f
                    vv_hidden_units[i][j]->vf_acc_big_delta[k] = 0.0f;

                }
            }
        }
        #ifdef TEST_UNIT
        cout << "Function update_accummulate_weight: ";
        print_connection();
        #endif
    }

    /**
     * function update weight with batch learning  
     * data: full training input data, raw without bias 
     * target: full training target
     * beg: begining index of the batch
     * end: end index of the batch 
     * eta: learning rate
     * alfa: momentum parameter 
     * lambda: weight decay (regularization) parameter 
     * */
    void batch_learning(vector<vector<float> > data, vector<vector<float> > target, int beg, int end,
                                  float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //check size compatible 
        if (data.size()  != target.size())
        {
            cerr <<  "Function accummulate_delta_weight: Number of sample and target are incompatible";
            return;
        }
        if(data[0].size()  != i_n_input_units - 1) // raw data without bias term
        {
            cerr << "Function accummulate_delta_weight: Size of input data is incompatible";
            return;
        }
        if (target[0].size() != i_n_output_units)
        {
            cerr << "Function accummulate_delta_weight: Size of output data is incompatible";
            return;
        }

        for(int i = beg; i < end; i++)
        {
            vector<float> bias_data = data[i];
            bias_data.push_back(1.0f); //add bias term 
            feed_forward_stochastic(bias_data);
            back_prop_stochastic(target[i]);
            accummulate_weight(eta, alfa, lambda);
        }
        update_accummulate_weight(end-beg);
    }

    /**
     * Function calculate the units' delta, for multithreads
     * but don't store the computed delta in f_delta for multithreads
     * vvf_unitsout: external output of all the units 
     * vvf_unitsder: external derivative of all the units
     * target: target of the considered sample
     * */ 
    void get_extern_delta (vector<vector<float> >vvf_unitsout, vector<vector<float> > vvf_unitsder,
                           vector<float> vf_target, vector<vector<float> >& vvf_unitsdelta)
    {
        int n_layers = vvf_unitsout.size();
        if(n_layers <= 0) 
        {
            cerr << "Function get_extern_delta: size of unitsout vector <= 0";
            return;
        }
        
        //init value for vvf_unitsdelta 
        for(int i = 0; i < vvf_unitsout.size(); i++)
        {
            vector<float> tem;
            for(int j = 0; j < vvf_unitsout[i].size(); j++)
            {
                tem.push_back(0.0f);
            }
            vvf_unitsdelta.push_back(tem);
        }

        //compute for the output units at layer [n_layers -1]
        for(int i = 0; i < vvf_unitsout[n_layers - 1].size(); i++)
        {
            float temdel = (vf_target[i] - vvf_unitsout[n_layers-1][i])*vvf_unitsder[n_layers-1][i];
            vvf_unitsdelta[n_layers-1][i] = temdel;
        }

        for(int i = n_layers - 2; i >= 1; i--)
        {
            for(int j = 0; j < vvf_unitsout[i].size(); j++)
            {
                float tem = 0.0f;
                for(int k = 0; k < vv_hidden_units[i-1][j]->v_feed_to.size(); k++)
                {
                    tem += vv_hidden_units[i-1][j]->vf_out_weights[k]*vvf_unitsdelta[i+1][k];

                    // #ifdef TEST_UNIT
                    // cout << vv_hidden_units[i-1][j]->vf_out_weights[k] << " * " << vvf_unitsdelta[i+1][k] << " + ";
                    // #endif 
                }
                
                vvf_unitsdelta[i][j] = tem*vvf_unitsder[i][j];

                // #ifdef TEST_UNIT
                // cout << " = " << tem << endl;
                // cout << tem << " * " << vvf_unitsdelta[i][j] << " = " << vvf_unitsdelta[i][j] << endl;
                // #endif
            }
        }
        #ifdef TEST_UNIT
        cout << "Print the external delta to check" << endl;
        for(int i = 0; i < vvf_unitsdelta.size(); i++)
        {
            for(int j = 0; j < vvf_unitsdelta[i].size(); j++)
            {
                cout << vvf_unitsdelta[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        #endif 
    }

    /**
     * Calculate accumulated weight with external data (for multithread)
     * unitsout: external unit's output 
     * unitsdelta: external unit's delta 
     * eta: learning rate
     * alfa: momentum parameter
     * lambda: weight decay parameter 
     * */
    void acc_weight_ex_data(vector<vector<float> >unitsout, vector<vector<float> >unitsdelta, 
                            float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //input layer
        for(int i = 0; i < i_n_input_units; i++)
        {
            for(int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                float big_delta = eta*unitsout[0][i]*unitsdelta[1][j] +
                alfa*v_inputs[i]->vf_momentum[j] - lambda*v_inputs[i]->vf_out_weights[j];
                
                v_inputs[i]->vf_acc_big_delta[j] += big_delta;
            }
        }

        //hidden layers 
        for(int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                for(int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    float big_delta = eta*unitsout[i+1][j]*unitsdelta[i+2][k] +
                    alfa*vv_hidden_units[i][j]->vf_momentum[k] - 
                    lambda*vv_hidden_units[i][j]->vf_out_weights[k];

                    vv_hidden_units[i][j]->vf_acc_big_delta[k] += big_delta;
                }
            }
        }
    }

    /**
     * Function accumulate big_delta for all the weight for batch learning 
     * data: full training input data, raw without bias 
     * target: full training target
     * beg: begining index of the batch
     * end: end index of the batch 
     * eta: learning rate
     * alfa: momentum parameter 
     * lambda: weight decay (regularization) parameter 
     * */
    void batch_learning_multithreads(vector<vector<float> > data, vector<vector<float> > target, int beg, int end,
                                  float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //check size compatible 
        if (data.size()  != target.size())
        {
            cerr <<  "Function accummulate_delta_weight: Number of sample and target are incompatible";
            return;
        }
        if(data[0].size()  != i_n_input_units - 1) // raw data without bias term
        {
            cerr << "Function accummulate_delta_weight: Size of input data is incompatible";
            return;
        }
        if (target[0].size() != i_n_output_units)
        {
            cerr << "Function accummulate_delta_weight: Size of output data is incompatible";
            return;
        }

        //multithreads 
        unsigned int numThreads = std::thread::hardware_concurrency();
        #ifdef DEBUG 
        cout << "Using " << numThreads << " threads to calculate a batch" << endl;
        #endif  
        int chunk_size = (int) (end-beg)/numThreads;
        auto accum_chunk = [&](int cidx)
        {   
            int beg_chunk = cidx*chunk_size;
            int end_chunk;
            if (cidx >= numThreads - 1) {end_chunk = end;}
            else { end_chunk = (cidx+1)*chunk_size;}
            
            //init all the local accumulate weight = 0.0f
            //first weight from input units to 

            for(int i = beg_chunk; i < end_chunk; i++)
            {
                vector<vector<float> > vvf_unitsout; //store units' output for one sample 
                vector<vector<float> > vvf_unitsder; //store units' derivative for one sample 
                vector<vector<float> > vvf_unitsdelta; //store units' delta for one sample

                vector<float> bias_data = data[i];
                bias_data.push_back(1.0f); //add bias term 

                //get external data to accumulate weight 
                compute_one_error(bias_data, target[i], vvf_unitsout, vvf_unitsder);
                get_extern_delta(vvf_unitsout, vvf_unitsder, target[i], vvf_unitsdelta);

                acc_weight_lock.lock();
                acc_weight_ex_data(vvf_unitsout, vvf_unitsdelta, eta, alfa, lambda);
                acc_weight_lock.unlock();
            }

        };

        vector<thread> tids;
        for(int i = 0 ; i < numThreads; i++)
        {
            tids.push_back(thread(accum_chunk, i));
        }

        for(int i = 0 ; i < numThreads; i++)
        {
            tids[i].join();
        }
        update_accummulate_weight((int) (end-beg));
    }

    /**
     * Function implement stochastic back propagation learning algorithm
     * data: raw training input data, not consider bias term 
     * eta: learning rate
     * ep: stopping criteration
     * max_iter: maximum number of epoch
     * alfa: momemtum parameter 
     * lambda: weight decay parameter 
     * */
    float train_stochastic(vector<vector<float> > data, vector<vector<float> > target, 
                        float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        // cout << "Training stochastic...." << endl;
        //check size compatible of target and data 
        if (data.size() != target.size())
        {
            cerr << "Function: back_prop_stoc: data and target size are incompatible";
            return -1.0f;
        }

        //Calculate sum square error of all the training data
        int num_iter = 0;
        float toterr = 0.0f;
        do
        {
            num_iter++;

            //shuffle the data and the target together by shuffling the index 
            vector<int> index_shuffle;
            //shuffle indexes of the data 
            for(int i = 0; i < data.size(); i++)
            {
                index_shuffle.push_back(i);
            }
            std::random_device rd; 
            std::mt19937 g(rd());
            shuffle(index_shuffle.begin(), index_shuffle.end(), g);

            for(int i = 0; i < index_shuffle.size(); i++)
            {
                //comput sum square error before learning
                if(i == index_shuffle.size()-1)
                {
                    toterr = compute_sum_square_multithreads(data, target);
                    if (toterr <= ep) break;
                }
                // cout << toterr << endl;
                #ifdef DEBUG
                cout << "Iter " << num_iter<<  ": Sum square error = " << toterr << endl;
                #elif defined (LOG)
                if(num_iter %50 == 0)
                {
                    cout << num_iter  << "\t" << toterr << endl;
                }
                #endif

                //take the next sample to learn 
                vector<float> bias_sample = data[index_shuffle[i]];
                bias_sample.push_back(1.0f);
                stochastic_learning(bias_sample, target[index_shuffle[i]], eta, alfa, lambda);
            }

            // //compute sum square err 
            // toterr = compute_sum_square_multithreads(data, target);
            // //take random a sample and train 
            // int sid = rand()%data.size(); //get index of the chosen sample 
            // #ifdef DEBUG
            // cout << "Take the sample " << sid << " to train" << endl;
            // #endif
            // vector<float> bias_sample = data[sid];
            // bias_sample.push_back(1.0f);
            // stochastic_learning(bias_sample, target[sid], eta, alfa, lambda);
                    
        }while (num_iter <= max_iter);
        return toterr;
    }

    
    /**
     * Function implement batch learning 
     * data: raw input, not consider bias term yet
     * target: target 
     * ep: stopping criteriation 
     * max_iter: maximum iteration 
     * alfa: momemtum parameter 
     * lambda: weight decay parameter 
     * */
    float train_batch(vector<vector<float> > data, vector<vector<float> > target, 
                    float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
    {
        //check size compatible of target and data 
        if (data.size() != target.size())
        {
            cerr << "Function: back_prop_stoc: data and target size are incompatible";
            return -1.0f;
        }

        //Calculate sum square error of all the training data
        int num_iter = 0;
        float toterr = 0.0f;
        do
        {
            num_iter++;
            //compute sum square err 
            toterr = compute_sum_square_multithreads(data, target);

            #ifdef  DEBUG
            cout << "Sum square error: " << toterr <<  endl;
            #elif defined (LOG)
            if(num_iter %50 == 0)
            {
                cout << num_iter  << "\t" << toterr << endl;
            }
            #endif

            if (toterr <= ep) break;

            // batch_learning(data, target, 0, target.size(), eta, alfa, lambda);
            batch_learning_multithreads(data, target, 0, target.size(), eta, alfa, lambda);
                    
        }while (num_iter <= max_iter);
        return toterr;
    }

    /**
     * Print units' value of the net
     * value = 1 for print units' value 
     * value = 2 for print units' dervivative
     * value = 3 for print units' delta
     * */
    void print_unit_value(int value) // 1 for print value, 2 for print derivative, 3 for print delta
    {
        switch (value)
        {
        case 2:
            cout << "Print dervivative of the net" << endl;
            break;
        case 3:
            cout << "Print delta of the net" << endl;
            break;
        default:
            cout << "Print value of the net" << endl;
            break;
        }
        //input layer 
        for (int i = 0; i < v_inputs.size(); i++)
        {
            if(value == 2) {cout << v_inputs[i]->f_dernet << " ";}
            else if (value == 3) {cout << v_inputs[i]->f_delta << " ";}
            else {cout << v_inputs[i]->f_value << " ";}
        }
        cout << endl;

        //hidden layer 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                if (value == 2) {cout << vv_hidden_units[i][j]->f_dernet << " ";}
                else if(value == 3) {cout << vv_hidden_units[i][j]->f_delta<< " ";}
                else { cout << vv_hidden_units[i][j]->f_value << " ";}
                
            }
            cout << endl;
        }

        //output layer
        for(int i = 0; i < i_n_output_units; i++)
        {
            if(value == 2) { cout << v_outputs[i]->f_dernet << " ";}
            else if (value == 3) { cout << v_outputs[i]->f_delta << " ";}
            else {  cout << v_outputs[i]->f_value << " ";}
        }
        cout << endl;
    }

    /**
     * Function to save model (architecture + weights) to file
     * filename: name of the saved file 
     * */
    void save_model(string filename)
    {
        //open file 
        ofstream file (filename.c_str());

        //first line contains: n_input_units n_ouput_units n_hidden_layers
        file << i_n_input_units << " " << i_n_output_units << " " << i_n_hidden_layers << endl;
        //second line contains number of units for each hidden layer 
        for (int i = 0; i < v_n_hidden_units.size(); i++)
        {
            file << v_n_hidden_units[i] << " ";
        }
        file << endl;

        //next lines store the weights of the net
        //input layer 
        for(int i = 0; i < v_inputs.size(); i++)
        {
            for(int j = 0; j < v_inputs[i]->v_feed_to.size(); j++)
            {
                file << v_inputs[i]->vf_out_weights[j] << " ";
            }
            file << endl;
        }
        
        //hidden layers 
        for (int i = 0 ; i < i_n_hidden_layers; i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                //save the name of the activation function of this unit 
                file << vv_hidden_units[i][j]->act << " ";

                for (int k = 0; k < vv_hidden_units[i][j]->v_feed_to.size(); k++)
                {
                    file << vv_hidden_units[i][j]->vf_out_weights[k] << " ";
                }
                file << endl;
            }
            file << endl;
        }

        for (int i = 0; i < i_n_output_units; i++)
        {
            //save the name of the actiavtion of each output unit
            file << v_outputs[i]->act << " ";
        }
        file << endl;
        file.close();
    }

    /**
     * Function to reset all the network 
     * */
    void reset_mlp()
    {
        //input layer 
        for (int i = 0; i < i_n_input_units; i++)
        {
            v_inputs[i]->reset_unit();
        }

        //hidden layers 
        for (int i = 0; i < i_n_hidden_layers; i++)
        {
            for (int j = 0; j < v_n_hidden_units[i]; j++)
            {
                vv_hidden_units[i][j]->reset_unit();
            }
        }
        
        //output layer 
        for(int i = 0; i < i_n_output_units; i++)
        {
            v_outputs[i]->reset_unit();
        }
    }

    /**
     * Destructor
     * */
    ~MLP()
    {
        for(int i = 0; i < v_inputs.size(); i++)
        {
            delete v_inputs[i];
        }
        for(int i = 0; i < v_outputs.size(); i++)
        {
            delete v_outputs[i];
        }
        for(int i = 0; i < vv_hidden_units.size(); i++)
        {
            for(int j = 0; j < vv_hidden_units[i].size(); j++)
            {
                delete vv_hidden_units[i][j];
            }
        }
    }
};


#endif 