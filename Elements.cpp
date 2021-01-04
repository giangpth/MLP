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

#define MONKS_ROOT "/Users/phamgiang/Study/Master/ThirdSemeter/ML/MLProject/Monks/"

// #define DEBUG 
// #define INTEGER 
// #define TEST_UNIT

mutex sum_err_lock;
mutex acc_weight_lock;


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

//class of unit of the network 
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
            #ifndef INTEGER // just for test 
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            vf_out_weights.push_back(r*1.5f - 0.75f); 
            #else
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

    ~Unit()
    {
        delete act_func;
    }
}; 

class MLP
{
private:   
    int i_n_input_units; // number of input units, remember to consider bias term yet, ex: input has 3 dims so i_n_input_units = 4 because add 1 bias term
    int i_n_output_units; // number of output units, output dimension 
    int i_n_hidden_layers; // number of hidden layers 
    vector<int> v_n_hidden_units; // array of number of hidden units of each hidden layer 

    vector<Unit*> v_inputs;
    vector<Unit*> v_outputs;
    vector< vector<Unit*> > vv_hidden_units;

public:

    /**
     * Constructor
     * _in: number of input units
     * _out: number of ouput units 
     * _hid: number of hidden layers 
     * _hid_units: list of number of units at each hidden layer
     * */
    MLP(int _in, int _out, int _hid, vector<int> _hid_units) //for fully connected 
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
                layer.push_back(new Unit(n_input, n_output, "gauss"));
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
            v_outputs.push_back(new Unit(v_n_hidden_units[i_n_hidden_layers - 1], 0, "sigmoid"));
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
     * vf_one_input: sample to test
     * return: output of the test sample
     * */
    vector<float> get_output(vector<float> vf_one_input)
    {
        //check size compatible 
        if(vf_one_input.size() != i_n_input_units)
        {
            cerr << "Function get_output: input size is incompatible";
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
     * function return error of a sample and also gives back the units' output of the net for this sample
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
        }
        if (vf_one_target.size() != i_n_output_units)
        {
            cerr << "Function computer_one_error: output sizes are incompatible";
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
        }
        if(data[0].size()  != i_n_input_units - 1) // raw data without bias term
        {
            cerr << "Function accummulate_delta_weight: Size of input data is incompatible";
        }
        if (target[0].size() != i_n_output_units)
        {
            cerr << "Function accummulate_delta_weight: Size of output data is incompatible";
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
     * Calculate accumulated weight with external data
     * 
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
     * function accumulate big_delta for all the weight for batch learning 
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
        }
        if(data[0].size()  != i_n_input_units - 1) // raw data without bias term
        {
            cerr << "Function accummulate_delta_weight: Size of input data is incompatible";
        }
        if (target[0].size() != i_n_output_units)
        {
            cerr << "Function accummulate_delta_weight: Size of output data is incompatible";
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
     * print units' value of the net
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

float compute_sum_square_multithreads(MLP* mlp, vector<vector<float> > data, vector<vector<float> >target)
{
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
            localsum += mlp->compute_one_error(data_with_bias, target[i], vvf_unitsout, vvf_unitsder);
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
    return toterr;
}

/**
 * function implement back propagation learning algorithm
 * data: raw training input data, not consider bias term yet, remember to add 1.0f for bias term before feed into the network 
 * eta: learning rate
 * ep: stopping criterion
 * max_iter: maximum iteration
 * alfa: momemtum parameter 
 * lambda: weight decay parameter 
 * */
void train_stochastic(MLP* mlp, vector<vector<float> > data, vector<vector<float> > target, 
                     float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
{
    cout << "Training stochastic...." << endl;
    //check size compatible of target and data 
    if (data.size() != target.size())
    {
        cerr << "Function: back_prop_stoc: data and target size are incompatible";
    }

    //Calculate sum square error of all the training data
    int num_iter = 0;
    float toterr = 0.0f;
    do
    {
        num_iter++;
        //compute sum square err 
        toterr = compute_sum_square_multithreads(mlp, data, target);
        
        #ifdef DEBUG
        cout << "Iter " << num_iter<<  ": Sum square error = " << toterr << endl;
        #else 
        cout << num_iter << "\t" << toterr << endl;
        #endif

        if (toterr <= ep) break;

        //take random a sample and train 
        int sid = rand()%data.size(); //get index of the chosen sample 
        #ifdef DEBUG
        cout << "Take the sample " << sid << " to train" << endl;
        #endif
        vector<float> bias_sample = data[sid];
        bias_sample.push_back(1.0f);
        mlp->stochastic_learning(bias_sample, target[sid], eta, alfa, lambda);
                
    }while (num_iter <= max_iter);
}

void train_batch(MLP* mlp, vector<vector<float> > data, vector<vector<float> > target, 
                 float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
{
    //check size compatible of target and data 
    if (data.size() != target.size())
    {
        cerr << "Function: back_prop_stoc: data and target size are incompatible";
    }

    //Calculate sum square error of all the training data
    int num_iter = 0;
    float toterr = 0.0f;
    do
    {
        num_iter++;
        //compute sum square err 
        toterr = compute_sum_square_multithreads(mlp, data, target);

        #ifdef  DEBUG
        cout << "Sum square error: " << toterr <<  endl;
        #else
        cout << num_iter  << "\t" << toterr << endl;
        #endif

        if (toterr <= ep) break;

        mlp->batch_learning(data, target, 0, target.size(), eta, alfa, lambda);
        // mlp->batch_learning_multithreads(data, target, 0, target.size(), eta, alfa, lambda);
                
    }while (num_iter <= max_iter);
}



void unit_test()
{
    // srand(time(0));

    vector<int> hidden_units{3, 4, 2};

    MLP* mlp = new MLP(3, 2, hidden_units.size(), hidden_units);
    vector<float> one_input {1.3f, 2.6f, 1.0f};
    vector<float> other_input {0.7f, -1.4f, 1.0f};

    vector<float> one_output {1.0f, 0.0f};
    vector<float> other_output {0.0f, 1.0f};

    vector<float> test {1.3f, 2.8f, 1.0f};
    vector<float> test_out {1.0f, 0.2f};

    mlp->feed_forward_stochastic(one_input);
    mlp->back_prop_stochastic(one_output);
    mlp->update_weight_stochastic(0.5f);

    delete mlp;
}



/** function to read file of monks dataset 
 * ftrain: name of the training file
 * ftest: name of the test file
 * fle: name of the learning error 
 * cls: vector of classes of samples
 * atts: vector of samples 
 * name: vector of the id of sample in monks dataset
 */ 
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
            
            // normalize in range [0.0f, 1.0f]
            vector<float> tem{(float)in1/3.0f, (float)in2/3.0f, (float)in3/2.0f, 
                              (float)in4/3.0f, (float)in5/4.0f, (float)in6/2.0f}; //normalize the data
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
        for(int j = 0; j < 6; j++)
        {
            cout << atts[i][j] << " ";
        }
        cout << names[i] << endl;
    }
    #endif 
}

/**
 * Function train mlp with monks dataset 
 * monk_id: id of monks dataset (1, 2, 3)
 * mlp: network 
 * ep: stopping criteation of total sumsquare error
 * max_iter: maximum iteration
 * eta: learning rate
 * alfa: momentum parameter 
 * lambda: weight decay (regularization) parameter
 * */
void train_monk(int monk_id, MLP* mlp, float ep, int max_iter, float eta, float alfa = 0.0f, float lambda = 0.0f)
{
    vector<vector<float> > trcls;
    vector<vector<float> > tratts;
    vector<string> trnames;
    string monks_name = string("monks-")  + to_string(monk_id) + string(".train");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), trcls, tratts, trnames);
    train_batch(mlp, tratts, trcls, ep, max_iter, eta, alfa, lambda);
    // train_stochastic(mlp, tratts, trcls, 0.1f, 50000, 0.03f, 0.0f, 0.0001f);
}

void test_monk(int monk_id, MLP* mlp)
{
    vector<vector<float> > tcls;
    vector<vector<float> > tatts;
    vector<string> tnames;
    string monks_name = string("monks-")  + to_string(monk_id) + string(".test");
    read_file_monks((string(MONKS_ROOT) + monks_name).c_str(), tcls, tatts, tnames);

    int n_corr = 0;
    vector<string> wrong_names;

    for(int i = 0; i < tcls.size(); i++)
    {
        vector<float> bias_data = tatts[i];
        bias_data.push_back(1.0f);
        vector<float> out = mlp->get_output(bias_data);

        int cls;
        if (out[0] >= 0.5f)
        {
            cls = 1;
        }
        else 
        {
            cls = 0;
        }
        if (cls == (int) tcls[i][0])
        {
            n_corr += 1;
        }
        else
        {
            wrong_names.push_back(tnames[i]);
        }
    }
    cout << "Correct: " << n_corr << " over " << tatts.size() << ": " 
         << (float)n_corr/tatts.size() * 100.0f << "%" << endl;
    cout << "List incorrect samples: " << endl;
    for(int i = 0; i < wrong_names.size(); i++)
    {
        cout << wrong_names[i] << endl;
    }
}

void eval_monk(int monks_id)
{
    int in_size = 7; // 6 attributes for data plus one bias term
    int out_size = 1; // binary classification

    //train 
    vector<int> hid_size {10, 8, 4, 2};
    MLP* mlp = new MLP(in_size, out_size, hid_size.size(), hid_size);
    train_monk(monks_id, mlp, 0.1f, 50000, 0.3f, 0.0f, 0.00005f);

    //evaluate 
    test_monk(monks_id, mlp);

    delete mlp;
}


int main(int argc, char** argv)
{
    // unit_test();
    srand(time(NULL));
    eval_monk(2);
    return 0;
}   