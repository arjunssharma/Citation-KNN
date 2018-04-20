import numpy as np
import scipy.spatial.distance as dist
import math

class CitationKNN(object):

    def __init__(self):
        self._bags = None
        self._bag_predictions = None
        self._labels = None
        self._full_bags = None
        self._DM = None

    def fit(self, train_bags, train_labels, **kwargs):
        self._bags = train_bags
        self._labels = train_labels
        self._no_of_references = kwargs['references']
        self._no_of_citers = kwargs['citers']

    def predict(self, Testbags):
        train_bags = self._bags
        #full_bags = self._bags+Testbags
        #print(full_bags)
        pred_labels = np.array([])
        self._DM = self.DistanceMatrixCKNN(train_bags)
        #print("Hi")
        #print(self._DM)
        #print(self._K)
        #print("Train bag")
        #print(train_bags)
        #print("Printing labels")
        #print(self._labels)

        for i in range(0, len(Testbags)):

            citers = []
            references = []
            
            #print("train bags")
            #print(train_bags)

            distances = []

            for j in range(0, len(train_bags)):
                distance = _min_hau_bag(Testbags[i], train_bags[j])
                distances.append(distance)
                self._DM[j].append(distance)

            self._DM.append(distances)
            last = len(self._DM) - 1
            self._DM[last].append(0)
            #print("Distance matrix")            
            #print(self._DM)
            
            arr = np.array( self._DM[last] )
            references = arr.argsort()[:self._no_of_references + 1]

            index = np.argwhere(references==last)
            references = np.delete(references, index)

            #print("References")
            #print(references)

            for j in range(0, len(self._DM) - 1):
                arr = np.array( self._DM[j] )
                neighbors = arr.argsort()[:self._no_of_citers + 1]
                if last in neighbors:
                    citers.append(j)

            #print("Citers")
            #print(citers)

            relevant_test_labels = []
            for j in range(0, len(references)):
                relevant_test_labels.append(self._labels[references[j]][0])
            for j in range(0, len(citers)):
                relevant_test_labels.append(self._labels[citers[j]][0])

            #print("All labels")
            #print(relevant_test_labels)
            
            relevant_test_labels.sort()

            label_out = relevant_test_labels[int(math.floor( (len(references) + len(citers) - 1) / 2))]
            pred_labels = np.append(pred_labels,label_out)

            self._DM.pop()
            for j in range(0, len(self._DM)):
                self._DM[j].pop()

        return pred_labels

    def DistanceMatrixCKNN (self, full_bag):
        w, h = len(full_bag), len(full_bag)
        Matrix = [[0 for x in range(w)] for y in range(h)] 
        count=0        
        for i in range(0, len(full_bag)):
                for j in range(0, len(full_bag)):
                    #print(str(i)+","+str(j))
                    Matrix[i][j] = _min_hau_bag(full_bag[i], full_bag[j])
                    
        return Matrix

def _min_hau_bag(X,Y):
    
    Hausdorff_distance = max( min((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
                               min((min([list(dist.euclidean(x, y) for x in X) for y in Y])))
                              )
    return Hausdorff_distance



