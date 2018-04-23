from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import timeit

def cross_validation(bags, labels, model_1, model_2, folds, parameters_1 = {}, parameters_2 = {}):  
    skf = StratifiedKFold(labels.reshape(len(labels)), n_folds = folds)
    results_accuracy_model_1 = []
    results_accuracy_model_2 = []
    fold = 0
    for train_index, test_index in skf:
        X_train = [bags[i] for i in train_index]
        Y_train = labels[train_index]
        X_test  = [bags[i] for i in test_index]
        Y_test  = labels[test_index]
        
        if len(parameters_1) > 0: 
            model_1.fit(X_train, Y_train, **parameters_1)
        if len(parameters_2) > 0: 
            model_2.fit(X_train, Y_train, **parameters_2)
        else: 
            model_1.fit(bags, labels)
            model_2.fit(bags, labels)
        print("Starting")
        predictions_1 = model_1.predict(X_test)
        predictions_2 = model_2.predict(X_test)
        #print("End prediction in mil_cross_val. Prediction result from model 1:")
        #print(predictions_1)
        #print("End prediction in mil_cross_val. Prediction result from model 2:")
        #print(predictions_2)
        if (isinstance(predictions_1, tuple)):
            predictions_1 = predictions_1[1]
        if (isinstance(predictions_2, tuple)):
            predictions_2 = predictions_2[1]	
			
        #Calculation of Accuracy
        accuracy_model_1 = np.average(Y_test.T == np.sign(predictions_1)) 
        accuracy_model_2 = np.average(Y_test.T == np.sign(predictions_2))
        results_accuracy_model_1.append(100 * accuracy_model_1)
        results_accuracy_model_2.append(100 * accuracy_model_2)
        fold = fold + 1
        
    return np.mean(results_accuracy_model_1), np.mean(results_accuracy_model_2)