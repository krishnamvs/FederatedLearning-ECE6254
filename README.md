# FederatedLearning-ECE6254
Flexible Federated Average Learning Simulator with the following abilities:
  - 2 Common Datasets to run against.
  - Easily change number of Edge Devices in the System.
  - Ability to receive and update only a select number of Edge Devices.
  - Easily change the number of Iterations you want to run the System.
  - Ability to choose if we want to update the Edge Devices.
  - Ability to simulate Malicious/Faulty Edge Devices.
  - Malicious/Faulty Edge Device Detection using MCD.
  - Ability to correct or fix the Malicious/Faulty Edge Devices.

Usage: 
  python3 Final.py --dataset DATASET
DATASET can be 'digits' for MNIST and 'fashion' for Fashion-MNIST.

Optional Flags

  --numberOfEdgeDevices NUMBEROFEDGEDEVICES - Default is 200 Devices.
  
  --percentageOfDevicesToCollectFrom PERCENTAGEOFDEVICESTOCOLLECTFROM - Default is 10%.
  
  --maxIterations MAXITERATIONS - Default is 50.
  
  --updateEdgeWeights | --no-updateEdgeWeights - Default is to always update Weights.
  
  --numberOfFaultyEdgeDevices NUMBEROFFAULTYEDGEDEVICES - Default is 0 Faulty Edge Devices.
  
  --detectOutlier | --no-detectOutlier - Default is to Not Detect Outliers.
  
  --correctFault | --no-correctFault - Default is to Not Correct Outliers. Need "--detectOutlier" for option to have an effect.
  
