from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='"digits" for MNIST dataset. "fashion" for F-MNIST dataset.')
parser.add_argument('--maxIterations', type=int, default=50, help='Max Iterations to run the Simulation')
args = parser.parse_args()

warnings.simplefilter("ignore", category = ConvergenceWarning)

# Store the Entire Data Set
if args.dataset == 'digits':
    dataset = "mnist_784"
elif args.dataset == 'fashion':
    dataset = "Fashion-MNIST"
x, y = fetch_openml(dataset, version = 1, return_X_y = True, as_frame = False)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 3)
logisticClassfier = LogisticRegression(max_iter = 2, warm_start = True)
classifierAccuracy = np.zeros(args.maxIterations)

for i in range(0, args.maxIterations):
    logisticClassfier.fit(xTrain, yTrain)
    classifierAccuracy[i] = accuracy_score(logisticClassfier.predict(xTest), yTest)

np.savetxt("baseline" + args.dataset + ".csv", classifierAccuracy, delimiter=",")