import os
import struct
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from multiclass_svm import testSVM

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def parseData(dataSet):
    retValue = read(dataset = dataSet)
    X_train = []
    Y_train = []
    for val in retValue:
        X_train.append(val[1].flatten())
        Y_train.append(val[0])

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    return (X_train, Y_train)

def getData():
    (X_train, Y_train) = parseData("training")
    (X_test, Y_test) = parseData("testing")
    return (X_train, Y_train, X_test, Y_test)


def getBinaryClassifierWeights(X_train, Y_train, c=0.1):
    assert len(np.unique(Y_train)) == 2, "binary classifier can't do multiclassification"
    clf = svm.LinearSVC(C=c)
    clf.fit(X_train, Y_train)
    return clf.coef_[0]

def one_vs_rest_classifier(X_train, Y_train, X_test, Y_test):

    labels = np.unique(Y_train)
    C_values = [0.01, 0.1, 1.0, 10]
    bestAccuracy, bestC = 0.0, 0
    for c in C_values:
        print('For the c value of:', c)
        accuracy = 0.0
        for trainIndex, testIndex in StratifiedKFold(random_state=0).split(X_train, Y_train):
            xTrain = X_train[trainIndex]
            xTest = X_train[testIndex]
            yTrain = Y_train[trainIndex]
            yTest = Y_train[testIndex]
    
            print('For new fold')

            weights = np.zeros((len(labels),784))
            for k in labels:
#                print('Calculating weights for',k)
                y = 2*np.array(map(int, yTrain==k), dtype=int) - 1 
                #all labels are -1 or 1
                weights[k,:] = getBinaryClassifierWeights(xTrain,y,c)
            accuracy += testSVM(xTest, yTest, weights, labels)
        print "Average Accuracy for all three folds:", accuracy/3.0
        if accuracy/3.0 > bestAccuracy:
            bestC = c
            bestAccuracy = accuracy/3.0
            
    print 'Taking the best value of C:', bestC
    
    weights = np.zeros((len(labels), 784))
    for k in labels:
        y = 2*np.array(map(int, Y_train==k), dtype=int) - 1
        weights[k, :] = getBinaryClassifierWeights(X_train, y, bestC)
    accuracy = testSVM(X_test, Y_test, weights, labels)
    print 'Final Test accuracy =',accuracy
    
if __name__ == "__main__":
    ## Here using one-vs-all algorithm predict the labels for X_test
    ## You just need to implement the logic of one-vs-all. For each binary classifiers use "getBinaryClassifierWeights" method.
    ## Cross validate on the training set to find out the value of c (parameter for binary classifiers) that generates the best score and then with this c, test on X-test and recoed the accuracy
    ## accuacy can be calculated by "accuracy_score(Y_test, predict_Y)"

    (X_train, Y_train, X_test, Y_test) =  getData()
    one_vs_rest_classifier(X_train, Y_train, X_test, Y_test)
