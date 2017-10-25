# CNNforCFD
This contains a sample code for the use of convolutional neural network for fluid force prediction of bluff body flows. 
This code is developed on MATLAB 2017b version and uses the MATLAB-Neural Networks Toolbox. To use this code, the users must ensure that they have MATLAB 2017b or a newer version with the Neural Networks Toolbox.

The repository contains the following files:

'CNNforCD.m' - The sample MATLAB code for prediction of the drag coefficient based on CNN.

'TrainingSetSmooth.mat' - Input geometry functions for 13 different bluff bodies.

'TestSet.mat' - Input functions for 14 different bluff bodies that are not included in 'TrainingSetSmooth.mat'.

'CDFOM.mat' - Mean drag coefficients on the bluff bodies in 'TrainingSetSmooth.mat' for a flow with a Reynolds Number = 100. The results are obtained using computational fluid dynamics (CFD) simulations.

To run the program, download all the above files and store them in a single directory.
