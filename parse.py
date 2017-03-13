import numpy as np

def target(X):
    target = np.zeros([X.shape[0],3],dtype=float)
    target[:,0] = X.interest_level=='high' 
    target[:,1] = X.interest_level=='medium' 
    target[:,2] = X.interest_level=='low' 
    return target

def get_ids(X):
    ids = X['listing_id']
    return ids

def parse_0(X):
    X = X[['latitude','longitude','bathrooms','bedrooms','price']]
    return X







