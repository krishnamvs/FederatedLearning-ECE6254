from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.covariance import MinCovDet
import numpy as np
import warnings
import random
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='"digits" for MNIST dataset. "fashion" for F-MNIST dataset.')
parser.add_argument('--numberOfEdgeDevices', type=int, default=200, help='Number of Edge Devices in the System. Default is 200.')
parser.add_argument('--percentageOfDevicesToCollectFrom', type=int, default=10, help='Precentage of Edge Devices to Collect Weights and Bias from. Default is 10%.')
parser.add_argument('--maxIterations', type=int, default=50, help='Max Iterations to run the Simulation')
parser.add_argument('--updateEdgeWeights', default=True, action=argparse.BooleanOptionalAction, help='Update Weights of Edge Devices!')
parser.add_argument('--detectOutlier', default=False, action=argparse.BooleanOptionalAction, help='Enable Fault Detection for Edge Devices. We will automatically generate Edge Devices that will be faulty!')
parser.add_argument('--correctFault', default=False, action=argparse.BooleanOptionalAction, help='Correct Faulty Edge Devices.')
parser.add_argument('--numberOfFaultyEdgeDevices', type=int, default=0, help='Number of Faulty Edge Devices in the System. Default is 0.')
args = parser.parse_args()

def fitModel(index, classifier, xtrain, ytrain):
    classifier.fit(xtrain, ytrain)
    return (index, classifier)

def collectClassifer(result):
    global logisticClassfier
    logisticClassfier[result[0]] = result[1]

def detectOutlier(paramlist):
    global args
    global faultyDevices
    locmcd = np.zeros(( args.numberOfEdgeDevices, args.numberOfFaultyEdgeDevices))
    for i in range(0, args.numberOfEdgeDevices):
        for j in range(0, args.numberOfFaultyEdgeDevices):
            locmcd[i][j] = paramlist[i][random.randint(0, 8)][faultyDevices[j]]
    cov = MinCovDet(random_state=0, support_fraction=1).fit(locmcd)
    dist = cov.mahalanobis(locmcd)
    mask = dist > (100000 * np.median(dist))
    return mask

warnings.simplefilter("ignore", category = ConvergenceWarning)

numberOfCommunicatingDevices = int(args.percentageOfDevicesToCollectFrom * args.numberOfEdgeDevices / 100)
classifierAccuracy = np.zeros((args.maxIterations, args.numberOfEdgeDevices))
xTrain = []
yTrain = []
xTest = [[]]
yTest = [[]]
logisticClassfier = []
EdgeDevices = []

faultFixed = False

for i in range(0, args.numberOfEdgeDevices):
    EdgeDevices.append(i)

if args.detectOutlier:
    faultyDevices = random.sample(range(args.numberOfEdgeDevices), args.numberOfFaultyEdgeDevices)

# Store the Entire Data Set
if args.dataset == 'digits':
    dataset = "mnist_784"
elif args.dataset == 'fashion':
    dataset = "Fashion-MNIST"
xTest[0], yTest[0] = fetch_openml(dataset, version = 1, return_X_y = True, as_frame = False)

for i in range(0, args.numberOfEdgeDevices):
    xTrain.append([])
    yTrain.append([])
    xTest.append([])
    yTest.append([])
    xTrain[i], xTest[i+1], yTrain[i], yTest[i+1] = train_test_split(xTest[i], yTest[i], train_size = 100, random_state = 3)
    xTest[i] = []
    yTest[i] = []
    logisticClassfier.append([])
    logisticClassfier[i] = LogisticRegression(max_iter = 2, warm_start = True)

for i in range(0, args.maxIterations):
    xCurrentTest = xTest[random.randint(0, args.numberOfEdgeDevices)]
    yCurrentTest = yTest[random.randint(0, args.numberOfEdgeDevices)]

    # Training on edge devices
    print('Iteration : {}'.format(i))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for j in range(0, args.numberOfEdgeDevices):
            pool.apply_async(fitModel, args = (j, logisticClassfier[j], xTrain[j], yTrain[j]), callback = collectClassifer)
        pool.close()
        pool.join()

    # Corrupt Weights of the Faulty Devicess
    if args.detectOutlier and not faultFixed:
        for j in range(0, args.numberOfFaultyEdgeDevices):
            logisticClassfier[faultyDevices[j]].coef_ = np.random.uniform(-20, 0, [10, 784])

    numberOfDetectedFaultyDevices = 0

    # Dectect Outliers 
    if args.detectOutlier and not faultFixed:
        low = []
        for j in range(0, args.numberOfEdgeDevices):
            low.append(logisticClassfier[j].coef_)
        mask = detectOutlier(low)
        numberOfFaultyDevices = np.sum(mask)
        if args.correctFault:
            numberOfDetectedFaultyDevices = 0
            mask = np.zeros(args.numberOfEdgeDevices)

    # Shuffle the Devices you will get the Weights and Bias from
    random.shuffle(EdgeDevices)

    # Initialize Global Parameters
    global_intercept = np.zeros(10)
    global_coef = np.zeros((10,784))

    # Update Global Parameters from Edge Devices
    if args.detectOutlier and not faultFixed:
        validCommunicatingDevices = numberOfCommunicatingDevices
        for j in range(0, numberOfCommunicatingDevices):
            if EdgeDevices[j] in faultyDevices:
                validCommunicatingDevices = validCommunicatingDevices - 1
        for j in range(0, numberOfCommunicatingDevices):
            if EdgeDevices[j] not in faultyDevices:
                global_intercept = global_intercept + (logisticClassfier[EdgeDevices[j]].intercept_ / validCommunicatingDevices)
                global_coef = global_coef + (logisticClassfier[EdgeDevices[j]].coef_ / validCommunicatingDevices)
    else:
        for j in range(0, numberOfCommunicatingDevices):
            global_intercept = global_intercept + (logisticClassfier[EdgeDevices[j]].intercept_ / numberOfCommunicatingDevices)
            global_coef = global_coef + (logisticClassfier[EdgeDevices[j]].coef_ / numberOfCommunicatingDevices)
    
    # Update Edge Devices with Global Parameters
    if args.updateEdgeWeights:
        if i > (0.7 * args.maxIterations):
            numberOfCommunicatingDevices = args.numberOfEdgeDevices
        for j in range(0, numberOfCommunicatingDevices):
            logisticClassfier[EdgeDevices[j]].intercept_ = global_intercept
            logisticClassfier[EdgeDevices[j]].coef_ = global_coef

    # Correct The Faulty Edge Devices
    if args.correctFault:
        faultFixed = True
        for j in range(0, args.numberOfFaultyEdgeDevices):
            logisticClassfier[faultyDevices[j]].intercept_ = global_intercept
            logisticClassfier[faultyDevices[j]].coef_ = global_coef
    
    for j in range(0, args.numberOfEdgeDevices):
        classifierAccuracy[i][j] = accuracy_score(logisticClassfier[j].predict(xCurrentTest), yCurrentTest)

fileName = args.dataset + '_' + str(args.numberOfEdgeDevices) + '_' + str(args.percentageOfDevicesToCollectFrom) + '_' + str(args.numberOfFaultyEdgeDevices)
if args.updateEdgeWeights:
    fileName = fileName + '_updateWeights'
if args.detectOutlier:
    fileName = fileName + '_detectOutlier'
np.savetxt(fileName + ".csv", classifierAccuracy, delimiter=",")