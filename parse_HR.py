import pandas as pd

def parse_0(X):
    ''' Benchmark model'''
    target = X.left
    X = X[['satisfaction_level', 'time_spend_company','promotion_last_5years']]
    return X, target

def parse_1(X):
    target = X.left
    to_del = ['left', 'sales', 'salary']
    for col in to_del : 
        del X[col]
    return X, target

def parse_2(X):
    target = X.left
    to_dummies = ['sales','salary']
    for dum in to_dummies:
        class_dummies = pd.get_dummies(X[dum], prefix='split_'+dum)
        X = X.join(class_dummies)
        del X[dum]
    to_del = ['left']
    for col in to_del : del X[col]
    return X, target



