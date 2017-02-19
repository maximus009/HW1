import os
import struct
import numpy as np
from scipy.sparse import rand


def read(dataset="training", path="."):
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
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arraysra
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


def getNormData(dataType):
    x = []
    y = []
    for val in read(dataset=dataType):
        x.append(np.divide(val[1].flatten(), 255))
        y.append(val[0])

    return np.asarray(x), np.asarray(y)


def computeObjectiveValue(weight, C, xTrain, yTrain):
    total_part_1 = 0
    for i in range(weight.shape[0]):
        total_part_1 += np.dot(weight[i].T, (weight[i]))
    total_part_1 = 0.5 * total_part_1
    y_labels = np.unique(yTrain)
    total_part_2 = 0
    for i in range(len(xTrain)):
        predicted_y = -1
        max_y_hat = -np.inf
        for j in y_labels:
            y_hat = np.dot(weight[j].T, xTrain[i])
            if j != yTrain[i]:
                y_hat += 1
            if max_y_hat < y_hat:
                max_y_hat = y_hat
                predicted_y = j
        total_part_2 += (max_y_hat - np.dot(weight[yTrain[i]].T, xTrain[i]))
    total_part_2 = C * total_part_2
    return total_part_1 + total_part_2


def trainSVM(xTrain, yTrain, C=1, numEpoch=100):
    # xTrain, yTrain: data
    # numEpoch: number of iteration
    learning_rate = 0.01
    feature_size = xTrain.shape[1]
    y_labels = np.unique(yTrain)
    weight = np.zeros((len(y_labels), feature_size))
    for epoch in range(numEpoch):
        loss = 0
        for i in range(len(xTrain)):
            predicted_y = -1
            max_y_hat = -np.inf
            for j in y_labels:
                y_hat = weight[j].dot(xTrain[i])
                if j != yTrain[i]:  # mismatch
                    y_hat += 1
                if max_y_hat < y_hat:
                    max_y_hat = y_hat
                    predicted_y = j
            # loss can be used for debugging
            if predicted_y != yTrain[i]:
                loss += 1
            # update weight
            #  =========================================
            # You need to implement the update rules here
            #  =========================================

    print('%d\t%f\t%f' % (epoch, ((loss / len(xTrain)) * 100), computeObjectiveValue(weight, C, xTrain, yTrain)))

    return weight


def testSVM(xTest, yTest, weight, y_labels):
    feature_size = xTest.shape[1]
    accurate = 0
    for i in range(len(xTest)):
        predicted_y = -1
        max_y_hat = -np.inf
        for j in y_labels:
            y_hat = weight[j].dot(xTest[i])
            if max_y_hat < y_hat:
                max_y_hat = y_hat
                predicted_y = y_labels[j]
        if predicted_y == yTest[i]:
            accurate += 1.0

    print('Accuracy: ', (accurate / float(len(xTest))) * 100.0)


def toLiblinear(data, label, fileName):
    fp = open(fileName,'w')
    feature_size = data.shape[1]
    for i,y in enumerate(label):
        fp.write(str(y+1))
        for j in range(feature_size):
            if data[i][j]!=0:
                fp.write(' '+str(j+1)+":"+str(data[i][j]))
        fp.write('\n')
    fp.close()

    
def crossV(X_train, Y_train, X_test, Y_test):
    
    labels = np.unique(Y_train)
    C_values = [0.01]#, 0.1, 1.0, 10]
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

            weights = trainSVM(xTrain, yTrain, numEpoch=10)
            accuracy += testSVM(xTest, yTest, weights, labels)
        print "Average Accuracy for all three folds:", accuracy/3.0
        if accuracy/3.0 > bestAccuracy:
            bestC = c
            bestAccuracy = accuracy/3.0
            
    print('Taking the best value of C:', bestC)
    
    weights = trainSVM(X_train, Y_train, C=bestC)
    accuracy = testSVM(X_test, Y_test, weights, labels)
    print('Final Test accuracy =',accuracy)
    

if __name__ == '__main__':

    X_train, Y_train = getNormData("training")
    X_test, Y_test = getNormData("testing")
    crossV(X_train, Y_train, X_test, Y_test)
