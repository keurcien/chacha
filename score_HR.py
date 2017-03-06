from sklearn.model_selection import cross_val_score
import numpy as np
def compute_score(clf, X, y):
    xval = cross_val_score(clf, X, y, cv=5)
    return np.mean(xval)







