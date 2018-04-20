import sys,os
from sklearn.utils import shuffle
import random as rand
from cross_validation import cross_validation
sys.path.append(os.path.realpath('..\data'))
from data import load_data

#Import Algorithms 
from CitationKNN import CitationKNN
from KNN import KNN


print('Started')

#Load Data 
#bags,labels,_ = load_data('musk1_scaled')
#bags,labels,_ = load_data('musk2_scaled')
bags,labels,_ = load_data('data_gauss')


#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))

#Number of Folds 
folds=4

for k in range(2, 11):
    citationknn_classifier = CitationKNN() 
    parameters_citationknn = {'references': k, 'citers': k+2}
    knn_classifier = KNN();
    parameters_knn = {'k': k}
    accuracy_model1, accuracy_model2 = cross_validation(bags=bags,labels=labels, model1=citationknn_classifier, model2=knn_classifier, folds=folds, parameters1=parameters_citationknn, parameters2 = parameters_knn)
    print("k=" + str(2)+", references="+str(2)+", citers="+str(2+2))
    print('knn accuracy='+str(accuracy_model2)+", citation knn accuracy = "+str(accuracy_model1))