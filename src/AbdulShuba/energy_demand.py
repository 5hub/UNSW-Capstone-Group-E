#!/usr/bin/env python3
"""
energy_demand.py

Group E Team - UNSW Capstone Project

################################################################################
#            UNSW CAPSTONE PROJECT                                             #
#            GROUP - E Team Members                                            #
#            Last Updated Date:  30 Mar 2022                                   #
#            Description: This program is used to define different types of NN #
#                         The hyperparameters of the Neural Networks are also  #
#                         defined in this section. There are 4 different types #
#                         of Neural Networks are defined here. The Architecture#
#                         of these 4 Neural Networks are different. Thus       #
#                         a global parameter is used to control which Architec-#
#                         ture is used when running the program.               # 
################################################################################ 

Energy Demand Forecasting using Neural Networks and Deep Learning

This file is used to create additional variables, functions, classes, etc.
and this code runs with the main program Forecast_Demand_Main.py file 

The default valaues of  trainValSplit, batchSize, epochs, and optimiser, can 
be found on Forecast_Demand_Main.  We can further modify these modify these 
to improve the performance our model. This will help us to predict 
Energy Demand 

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""

import torch
import torch.nn as tnn
import torch.nn.functional as tnnfunc
import torch.optim as toptim
import math
from config import device
import pandas as pd
from torch.autograd import Variable
import torch.utils.data as Data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 
from datetime import datetime



#-----------------------------Global Variable ---------------------------------#
#                 Other global variables used in energy_demand.py              # 
#-------------------------------------------- ---------------------------------#


input_dim = 4  
# This is the input dimension to the neural network.
# At present there will be 4 dimention 
# Temperature, Day, Month, and Time Interval

# Different Architecture are defined below to test with different 
# Neural Network

architecture = 1 # Can have values 1, 2, 3, 4
# This Architecture Value can be changed in order to change the 
# Neural Network Architecture. Below are the details of the value
# 1 represents LSTM Network + 1 fully connected linear layer + Output Layer
# 2 represents LSTM Network + 2 Fully Connected linear Layer + Output Layer
# 3 represents GRU Network + 1 Fully Connected linear Layer + Output Layer
# 4 represents GRU Network + 2 Fully Connected linear Layer + Output Layer

optimiser_choice = 1 # Can have  1 or 2
# The options for optimiser is as mentioned below
# 1  -  SGD Optimiser
# 2  -  ADAM Optimiser

LSTM_GRU_Hidden_Size = 300
# This is the output Size of either the LSTM or GRU Network based on the 
# architecture chosen

LSTM_GRU_Num_Layers = 3
# This is the number of hidden layers in either the LSTM or GRU Network 
# chosen as per the architecture.

Linear_Layer1_output_size = 500
# This is the output size of the first linear Layer which is connected to
# either GNU or LSTM as per the architecture choices made above.

Linear_Layer2_output_size = 120
# This is the output size of the second linear Layer which is connected to
# either GNU or LSTM as per the architecture choices made above.

nn_output_size = 1
# This is the output size of neural network.
# The model output will have predictions it makes in
# the same format as the energy demand dataset.  
# The predictions must be of type float
# so there will be one output which will forecast
# what the energy demand will be.

learning_rate = 0.01 #0.032
# this is the learning rate used in the optimizers

weight_decay = 0.0001
# weight decay used in ADAM Optimiser

training_dataset_filename= 'training_dataset.csv'
#Training dataset path

trainValSplit = 1
# Training and Validation Data set Split
# Need to work on the way to split tensordataset
# in training and Validation Set

batchSize = 1000
# setting the batch size as a global parameter.

epochs = 10 


################################################################################
##### The following determines the loads and processes the data           ######
################################################################################


def upload_data():
    """
    This function is used to load the data
    """
    cwd = os.getcwd()
    training_dataset = pd.read_csv(training_dataset_filename) 
    return training_dataset

def process_data(training_dataset):
    """
    The data is processed and converted in tensors
    """
    training_dataset['TEMPERATURE'] = training_dataset['TEMPERATURE']

    training_dataset['Date_object'] = training_dataset['Date'].apply(pd.to_datetime, format='%d/%m/%Y')

    training_dataset['month'] = training_dataset['Date_object'].apply(lambda x: x.month)

    training_dataset['day'] = training_dataset['Date_object'].apply(lambda x: x.day)

    times = list(training_dataset['Time'][:48])

    training_dataset['time_int'] = training_dataset['Time'].apply(lambda x: times.index(x))

    X = training_dataset[['TEMPERATURE', 'day', 'month', 'time_int']]

    x = torch.tensor(X.values)




    # creating tensor from targets_df 
    Y = training_dataset[['TOTALDEMAND']]
    y = torch.tensor(Y.values)

    # torch can only train on Variable, so convert them to Variable
    x_data, y_data = Variable(x), Variable(y)


    return x_data, y_data


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to this network will be a
    tensor which has the temperature the day month and time of the year. The forward method
    should return an output for the energy demand.
    """

    def __init__(self):
        super(network, self).__init__()


  # The neural network will take the temperature vector as an input.
  # The temperature Vector will have Temperature, Day, Month, Time_Int
  # Thus it is necessary that neural network input layer dimension 
  # remains same as the time vector dimension which is 4.
        # Define the Neural Network Parameters below
        self.input_size = input_dim 
        self.hidden_size = LSTM_GRU_Hidden_Size
        self.num_layers = LSTM_GRU_Num_Layers
        self.linear_output_size1 = Linear_Layer1_output_size
        self.linear_output_size2 = Linear_Layer2_output_size
        self.tnn_output = nn_output_size
        self.tnn_input = None # This will be determined based on architecture

        # the program checks the architecture and then builds the network based on the 
        # architecture values. When the architecture value is equal to 1 or 2 the Recurring Neural 
        # Network (RNN) is created using LSTM and when the architecture value is 3 or 4 the RNN
        # network is created using GRU. 
        if architecture == 1 or architecture == 2 or architecture == 3 or architecture == 4:
            # if proper architecture is choosen then the network is built accordingly
            # change the architecture value to build a specific network
            if architecture == 1 or architecture == 2:
                #------------------------------LSTM NETWORK -------------------------------# 
                # Generating an LSTM Network using the parameters set above.
                # this LSTM Network takes the input size same as the word vector
                # dimension
                self.lstm = tnn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                  num_layers=self.num_layers, batch_first=True)
                #--------------------------------------------------------------------------# 
            else:
                # this is the case for architecture == 3 or architecture == 4
                #------------------------------GRU NETWORK -------------------------------# 
                # Generating a GRU Network using the parameters set above.
                # this GRU Network takes the input size same as the word vector
                # dimension
                self.gru = tnn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                                  num_layers=self.num_layers, batch_first=True)
                #--------------------------------------------------------------------------# 

            # the output from the LSTM / GRU network is passed to a fully connected
            # Linear network. So the input to this linear network is same as the 
            # hidden size of the  Network and the output is defined by the parameter 
            # linear_output_size1
            self.fully_connected_layer1 =  tnn.Linear(self.hidden_size, self.linear_output_size1)

            if architecture == 1 or architecture == 3:
                # In this case the output of the first fully connected layer is fed
                # to the input of the output layer
                self.tnn_input = self.linear_output_size1
            else:
                # In this case the output of the first fully connected layer is fed
                # to the input of the second fully connected layer
                self.fully_connected_layer2 =  tnn.Linear(self.linear_output_size1, self.linear_output_size2 )

                # Further the output of the second fully connected layer is fed
                # to the output layer
                self.tnn_input = self.linear_output_size2


            # The output of the last fully connected 
            # Linear network is fed into the input of the output layer
            # and the output from the output layer is defined by 
            # the parameter tnn_output
            self.output =  tnn.Linear(self.tnn_input , self.tnn_output)

            # Initializing Weight Buffer for fully connected layer
            # which will be activated lated using RELU Activation.
            torch.nn.init.kaiming_normal_(self.fully_connected_layer1.weight.data)

            # finding the standard deviation using the input size of 
            # of the output layer. This will be used to initialise
            # output layer weights
            stdv = 1.0 / math.sqrt(self.tnn_input)      
            self.output.weight.data.uniform_(-stdv, stdv)

            # initialise the bias of the two layers
            self.fully_connected_layer1.bias.data.zero_()
            self.output.bias.data.zero_()

            if architecture == 2 or architecture == 4:
                #Initialise the weights of second linear network.
                # which will be activated lated using RELU Activation.
                torch.nn.init.kaiming_normal_(self.fully_connected_layer2.weight.data)

                #Initialise the bias of second linear network.
                # which will be activated lated using RELU Activation.
                self.fully_connected_layer2.bias.data.zero_()             

        else:
            # this is the case when wrong architecture option is added 
            assert False, 'Wrong Value of Architecture assigned during Global Variable Declaration'



    def forward(self, input):

        # passing the input to the embedding layer.


        if architecture == 1 or architecture == 2:
            # This is the scenario for LSTM Network
            #------------------------------LSTM Forward Propogation---------------------# 
            # hidden state
            h_0 = torch.zeros(self.num_layers, input.size(1), self.hidden_size).to(device) 

            # internal state
            c_0 = torch.zeros(self.num_layers, input.size(1), self.hidden_size).to(device) 

            # Initiate weights before tanh activation
            torch.nn.init.xavier_normal_(h_0) 
            torch.nn.init.xavier_normal_(c_0) 

            # Propagate input through LSTM network
            output, (h_n, c_n) = self.lstm(input, (h_0, c_0)) 
            #lstm with input, hidden, and internal state

            #--------------------------------------------------------------------------#

        elif architecture == 3 or architecture == 4:
            # This is the scenario for GRU Network
            #------------------------------GRU Forward Propogation---------------------# 
            # hidden state
            h_0 = torch.zeros(self.num_layers, input.size(1), self.hidden_size).to(device) 
            # Initiate weights before tanh activation
            torch.nn.init.xavier_normal_(h_0) 
            # Propagate input through GRU network
            output, (h_n) = self.gru(input, (h_0)) 
            #-------------------------------------------------------------------------#


        # reshaping the data and passing it to the 
        # fully connected dense layer 
        out = self.fully_connected_layer1(output[:,-1,:]) 

        # applying relu activation to the output 
        # from the fully connected dense layer
        out = torch.relu(out)

        if architecture == 2 or architecture == 4:
            # This is the case where RNN Network is connected
            # to two fully connected linear dense network
            # reshaping the data and passing it to the 
            # ----second fully connected dense layer ----------------------------------#
            out = self.fully_connected_layer2(out) 

            # applying relu activation to the output 
            # from the second fully connected dense layer
            out = torch.relu(out)
            # -------------------------------------------------------------------------#

        # passing the activated output to the final 
        # output
        out = self.output(out) 


        # The final output is activated using
        #  Softmax. Thus using the log
        # softmax activation with dimension 1
        DemandOutput = torch.log_softmax(out, dim=1)

        return DemandOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The actual demand (DemandTarget)
    and predicted demand output from the network (DemandOutput) 
    will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss_demand = tnnfunc.nll_loss


    def forward(self, DemandOutput, DemandTarget):
        DemandTarget = DemandTarget.to(torch.float).to(device)
        demand_loss = self.loss_rating(DemandOutput, DemandTarget)
        return demand_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################



if optimiser_choice==1 :
    # for choice 1 the SGD Optimiser is called
    optimiser = toptim.SGD(net.parameters(), lr=learning_rate)
elif optimiser_choice==2 :
    # The choice of Option 2 uses ADAM Optimiser
    optimiser = toptim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
else :
    # Wrong value of optimiser_choice assigned.
    assert False, 'Wrong value of optimiser_choice assigned.'
