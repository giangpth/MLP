#ifndef UNIT_H
#define UNIT_H

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
#include "Activation.h"

using namespace std;

class Unit 
{
    public:
    int n_input; // number of incomming inputs of this unit 
    int n_output; // number of out going units of this unit, this is equal the number of out going weights. 
    float f_value; // store output of j f(net_j), calculate in feed forward phase
    float f_dernet; // stores f'(net_j), calculate in feed forward phase
    float f_delta; //stores delta of j, calculate in back propagation phase 

    //vf_out_weights, vf_acc_big_delta and vf_momentum have the same size that equal n_output 
    vector<float> vf_out_weights; // outgoing weights from this units to all the units in v_feed_to
    vector<float> vf_acc_big_delta; // store the accumulated big delta for all the weight in v_out_weights, for batch learning version
    vector<float> vf_momentum; // for momentum (stored the old big delta of all the weights in v_out_weights)

    string act; // name of the activation function for this unit 

    vector<Unit*> v_feed_to; // list of all the units of next layer that receive signal from this unit
    vector<Unit*> v_feed_from; // list of all the units of previous layer that send signal to this unit 

    Activation* act_func; // activation function 

    Unit(int _in, int _out, string _act)
    {
        n_output = _out;
        n_input = _in;
        act = _act;
        act_func = new Activation(act);
        f_value = 0.0f;
        f_dernet = 0.0f;
        f_delta = 0.0f;
    }

    /**
     * Function to generate the weights of the unit randomly in range [-0.75, 0.75]
     * Besides also init all the momentum and accumulate big delta (for batch version) with 0.0f
     * */
    void random_weight()
    {
        for (int i = 0; i < n_output; i++)
        {
            #ifndef INTEGER 
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            vf_out_weights.push_back(r*1.5f - 0.75f); 
            #else // generate only integer weight, just for the test 
            float r = static_cast <float> (rand()%10 - 5);
            vf_out_weights.push_back(r);
            #endif
            // init all momentums equal zero
            vf_momentum.push_back(0.0f);
            vf_acc_big_delta.push_back(0.0f);

        }
        #ifdef UNIT_TEST
        cout << "Number of out weights: " << vf_out_weights.size() << endl;
        for (int i = 0; i < n_output; i++)
        {
            cout << vf_out_weights[i] << " ";
        }
        cout << endl;
        #endif
    }

    /**
     * Function to generate the weights of the unit by fan-in method
     * The range of the random is in [-1/a, 1/a] with a = sqrt(fan_in)
     * */
    void fan_in_random_weight()
    {
        int half = 1/(sqrt((float)n_input));

        for(int i = 0; i < n_output; i++)
        {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); // r is in the range [0.0f, 1.0f]
            vf_out_weights.push_back(r*2*half - half); 

            vf_momentum.push_back(0.0f);
            vf_acc_big_delta.push_back(0.0f);
        }
    }

    /**
     * Make this unit connect with the units of next layer 
     * feeds: list of units that receive the signal from this unit as input
     * */
    void set_feed_to_list (vector<Unit*> feeds)
    {
        v_feed_to.clear();
        for (int i = 0; i < feeds.size(); i++)
        {
            v_feed_to.push_back(feeds[i]);
        } 
        #ifdef DEBUG
        cout << "Number of out feed to: " << v_feed_to.size() << endl;
        #endif
    }

    /**
     * Make this unit connect to the unit of previous layers
     * backs: list of units that send input signal to this unit
     * */
    void set_back_prop_list (vector<Unit*> backs)
    {
        v_feed_from.clear();
        for(int i = 0; i < backs.size(); i++)
        {
            v_feed_from.push_back(backs[i]);
        }
        #ifdef DEBUG
        cout << "Number of out back prop: " << v_feed_from.size() << endl;
        #endif
    }
    
    /**
     * Function to reset all the weight of this unit to new random value 
     * */
    void reset_unit()
    {
        vf_out_weights.clear();
        vf_momentum.clear();
        vf_acc_big_delta.clear();
        f_value = 0.0f;
        f_dernet = 0.0f;
        f_delta = 0.0f;
        random_weight();
    }

    ~Unit()
    {
        delete act_func;
    }
}; 

#endif