# MLP
Implement a feed forward neural network
To implement the multilayer perceptron, I use three classes: MLP, Unit and Activation as depicted in Figure 1. <br>
In which, MLP is the multilayer perceptron which contains multiple units and these units are connected to each other according to the architecture of the network.
Inside each unit, there is an activation function belonging to one of these types: sigmoid, tanh, relu, softplus, leaky relu, gauss and linear. Units also contain the weights that connect the unit with other units of the next layer in the network. With these components, we can construct varied networks with different architectures such as various number of hidden layers or different activation function for each layer.<br>
<br>
The learning flow chart is depicted in Figure 2. When one input comes, the input will be fed forward to the network in feed forward phase.<br>
After this process, I will get the output value opj and the derivative f ′ (netpj ) (with f is the activation function of this unit) for each unit j of the net with the input p.<br>
In the next back propagation phase, the error signal is spread back to all the units of the network, in this phase, the value δpj of each unit j with the input p is calculated.<br>
For the batch learning version, after getting all the needed value (opj and δpj) of the input p for all units of the net, I can compute the value ∆w for the input p and then accumulate this value until all the training input data for one epoch are finished.<br>
Then I update the weights of each unit with the accumulated ∆w.<br>
To apply momentum and regularization, when we calculate the value ∆w, we add the momentum term and the regularization term.<br>
I choose to implement k-fold validation as the validation schema with the number k of folds can be chosen so that it will be suitable for dataset of different sizes.<br>
This program is implemented by C++ with standard libraries. To speed up the running time, I exploit multithreads in computing the mean square error of the training data and also in the accumulating process for batch learning.<br>
