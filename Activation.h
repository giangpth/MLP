#ifndef ACTIVATION_H
#define ACTIVATION_H

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

using namespace std;


//list of actiavtion functions and their derivative 
    float sigmoid (float sumnet)
    {
        return (1.0f/(1.0f + exp(-sumnet)));
    }
    float der_sigmoid (float sumnet)
    {
        float s = sigmoid(sumnet);
        return s*(1.0f - s);
    }

    float tan_h (float sumnet)
    {
        return ((exp(sumnet) - exp(-sumnet))/(exp(sumnet) + exp(-sumnet)));
    }
    float der_tanh(float sumnet)
    {
        float tem = (exp(sumnet) - exp(-sumnet))/(exp(sumnet) + exp(-sumnet));
        return (1-tem*tem);
    }

    float relu (float sumnet)
    {
        if (sumnet > 0) return sumnet;
        else return 0.0f;
    }
    float der_relu (float sumnet)
    {
        if (sumnet > 0) return 1.0f;
        else return 0.0f;
    }

    float linear(float sumnet)
    {
        return sumnet;
    }
    float der_linear(float sumnet)
    {
        return 1.0f;
    }

    float softplus(float sumnet)
    {
        return log(1.0f +exp(sumnet));
    }
    float der_softplus(float sumnet)
    {
        return (1.0f/(1 + exp(-sumnet)));
    }

    float leakyRelu(float sumnet)
    {
        if (sumnet < 0) return sumnet*0.01f;
        else return sumnet;
    }
    float der_leakyRelu(float sumnet)
    {
        if (sumnet < 0) return 0.01f;
        else return 1.0f;
    }

    float gauss(float sumnet)
    {
        return exp(-sumnet*sumnet);
    }
    float der_gauss(float sumnet)
    {
        return -2.0f*sumnet*exp(-sumnet*sumnet);
    }

    //this is just for test 
    float sign(float sumnet)
    {
        if (sumnet >= 0) return 1.0f;
        else return -1.0f;
    }
    float der_sign(float sumnet)
    {
        #ifdef DEBUG
        cout << "Using sign activation function" << endl;
        #endif
        if (sumnet >= 0) return 1.0f;
        else return -1.0f;
    }


class Activation
{
    public:
    string name;
    function <float(float)> activation;
    function <float(float)> derivative;
    Activation(){}
    Activation(string _name)
    {
        name = _name;
        if (name == "sigmoid") 
        {
            activation = sigmoid;
            derivative = der_sigmoid;
        }
        else if (name == "tanh") 
        {
            activation = tan_h;
            derivative = der_tanh;
        }
        else if (name == "relu") 
        {
            activation = relu;
            derivative = der_relu;
        }
        else if (name == "sign")
        {
            activation = sign;
            derivative = der_sign;
        }
        else if (name == "softplus")
        {
            activation = softplus;
            derivative = der_softplus;
        }
        else if (name == "leakyRelu")
        {
            activation = leakyRelu;
            derivative = der_leakyRelu;
        }
        else if (name == "gauss")
        {
            activation = gauss;
            derivative = der_gauss;
        }
        else if (name == "linear")
        {
            activation = linear;
            derivative = der_linear;
        }
        else
        {
            cerr << "No activation function named " << name << endl;
            exit(-1);
        }
    }
    
    /**
     * Function to compute f(net) of this unit
     * sumnet: the net value
     * */
    float compute(float sumnet)
    {
        return activation(sumnet);
    }

    /**
     * Function to compute f'(net) of this unit
     * sumnet: the net value
     * */
    float computeDer(float sumnet)
    {
        return derivative(sumnet);
    }
};

#endif 