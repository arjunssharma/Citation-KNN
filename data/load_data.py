from scipy.io import loadmat
import numpy as np

def load_data(data):
    
    if data == 'musk1_scaled':  
        f_bag = 'data/musk1_scaled/Bag2_mus_escal.mat'
        f_labels = 'data/musk1_scaled/bagI_mus_escal.mat'
        X_g = loadmat('data/musk1_scaled/X_mus_escal.mat')
    elif data == 'data_gauss': 
        f_bag = 'data/gauss_data/bag_g.mat'
        f_labels = 'data/gauss_data/bagI_gauss.mat'
        X_g = loadmat('data/gauss_data/X_g.mat')
    
        
    bag_g = loadmat(f_bag)
    labels = loadmat(f_labels)
    
	try: 
        Bag = bag_g['Bag2']
    except KeyError: 
        Bag = bag_g['Bag']
    
	labels = labels['bagI']
    X = X_g['X']
    Bag = np.squeeze(Bag - 1)
    nrobags = max(Bag+1)
    bags = []

    for i in range(0,nrobags): 
        index = np.where( Bag == i )
        bag = X[index]
        bags.append(bag)
    
	return bags,labels,X