import numpy as np
import scipy.spatial.distance as dist
import math

class KNN(object):

    def __init__(self):
        self._bags = None
        self._bag_predictions = None
        self._labels = None
        self._full_bags = None
        self._DM = None

    def fit(self, train_bags, train_labels, **kwargs):

        self._bags = train_bags
        self._labels = train_labels
        self._K = kwargs['k']

    def predict(self, Testbags):

        train_bags = self._bags
        full_bags = self._bags+Testbags
        #print(full_bags)
        pred_labels = np.array([])
        self._DM = self.DistanceMatrix(train_bags, Testbags)
        #print("Hi")
        #print(self._DM)
        #print(self._K)
        #print("Train bag")
        #print(train_bags)
        #print("Printing labels")
        #print(self._labels)

        for i in range(0, len(self._DM)):
            arr = np.array( self._DM[i] )
            ind = arr.argsort()[:self._K]
            #print("Array")
            #print(arr)
            #print("Indices of k minimum values")
            #print(ind)
            relevant_test_labels = []
            for j in range(0, len(ind)):
                relevant_test_labels.append(self._labels[ind[j]][0])
            #print("All labels")
            #print(relevant_test_labels)
            relevant_test_labels.sort()
            #print("Sorted labels")
            #print(relevant_test_labels)
            label_out = relevant_test_labels[math.floor(self._K / 2)]
            pred_labels = np.append(pred_labels,label_out)


    def DistanceMatrix (self,train_bags, test_bags): 
        w, h = len(train_bags), len(test_bags)
        Matrix = [[0 for x in range(w)] for y in range(h)] 
        count=0 
        for i in range(0, len(test_bags)):
                for j in range(0, len(train_bags)):
                    Matrix[i][j] = _min_hau_bag(test_bags[i], train_bags[j])
                    
        return Matrix

def _min_hau_bag(X,Y):    
    Hausdorff_distance = max( min((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
                               min((min([list(dist.euclidean(x, y) for x in X) for y in Y])))
                              )
    return Hausdorff_distance



