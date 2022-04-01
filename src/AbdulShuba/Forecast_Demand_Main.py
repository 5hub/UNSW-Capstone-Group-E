#!/usr/bin/env python3
"""
Group E Team - UNSW Capstone Project

################################################################################
#            UNSW CAPSTONE PROJECT                                             #
#            GROUP - E Team Members                                            #
#            Last Updated Date:  30 Mar 2022                                   #
#            Description: This is the main file  which created  the            #
#                         training dataset and splits the training dataset     #
#                         in different batches                                 # 
#                         This further saves the trained model which then      #
#                         can be used to predict demand based on temperature   #   
#                         This program also runs the validation tests and      #
#                         prints the output of the test results                #     
#                                                                              #
################################################################################


Forecast_Demand_Main

This is the main program file
"""

import torch

# Commenting the below line because
# module 'torchtext.data' has no attribute 
# 'Field'
# from torchtext import data

# from torchtext.legacy import data
import torch.utils.data as Data

from config import device
import energy_demand

def main():
    print("Using device: {}"
          "\n".format(str(device)))

    # Load the training dataset
    training_dataset = energy_demand.upload_data()


    # process the data and store in tensor
    x, y = energy_demand.process_data(training_dataset)



    # prepare the dataset using the tensor data
    dataset = Data.TensorDataset(x, y)






    # Allow training on the entire dataset, or split it for training and validation.
    if energy_demand.trainValSplit == 1:
        trainLoader = Data.DataLoader( 
                                dataset= dataset,
                                batch_size=energy_demand.batchSize, 
                                shuffle=True, num_workers=1)
    else:
        # need to handle the split functionality
        trainLoader = Data.DataLoader( 
                                dataset= dataset,
                                batch_size=energy_demand.batchSize, 
                                shuffle=True, num_workers=1)

    # Get model and optimiser from energy_demand.
    net = energy_demand.net.to(device)
    lossFunc = energy_demand.lossFunc
    optimiser = energy_demand.optimiser

    # Train.
    for epoch in range(energy_demand.epochs):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs , DemandTarget = batch


            # converting the inputs to float
            inputs, DemandTarget = inputs.float(), DemandTarget.float()


            # PyTorch calculates gradients by accumulating contributions to them
            # (useful for RNNs).  Hence we must manually set them to zero before
            # calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            DemandOutput = net(inputs)
            
            loss = lossFunc(DemandOutput.view(-1), DemandTarget.view(-1))

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            runningLoss += loss.item()

            if i % 10 == 0:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                      % (epoch + 1, i + 1, runningLoss / 500))
                runningLoss = 0

   

if __name__ == '__main__':
    main()
