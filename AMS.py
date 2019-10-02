from sklearn.metrics import make_scorer
import numpy as np

def compute_AMS(y_true, y_pred):
    # true positive
    s = np.count_nonzero((y_pred==True) & (y_true==True))
    
    # false positive
    b = np.count_nonzero((y_pred==True) & (y_true==False))
    
    br = 10

    radicand = 2 *((s + b + br) * np.log (1.0 + s/(b + br)) -s)
    return np.sqrt(radicand)

AMS = make_scorer(compute_AMS)