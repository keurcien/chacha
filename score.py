
import numpy as np

def compute_score(y_true, y_pred):
    assert(y_true.shape == y_pred.shape),'shape of predicted interest non compatible'
    logVec = y_true*np.log(np.clip(y_pred,1-1e-15,1e-15))
    return -np.sum( logVec, axis=(0 , 1))/y_true.shape[0]







