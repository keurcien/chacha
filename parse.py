import numpy as np
import pandas as pd

def target(X):
    target = X["interest_level"]
    #target = np.zeros([X.shape[0],3],dtype=float)
    #target[:,0] = X.interest_level=='high' 
    #target[:,1] = X.interest_level=='medium' 
    #target[:,2] = X.interest_level=='low' 
    return target

def parse_0(X):
    X = X[['latitude','longitude','bathrooms','bedrooms','price']]
    return X

def parse_1(X):
    #Source: https://www.kaggle.com/aikinogard/two-sigma-connect-rental-listing-inquiries/random-forest-starter-with-numerical-features
    X["num_photos"] = X["photos"].apply(len)
    X["num_features"] = X["features"].apply(len)
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    X = X[["bathrooms", "bedrooms", "latitude", "longitude", "price",
                 "num_photos", "num_features", "num_description_words",
                 "created_year", "created_month", "created_day"]]
    return X

def parse_2(X):
    #Source: https://www.kaggle.com/aikinogard/two-sigma-connect-rental-listing-inquiries/random-forest-starter-with-numerical-features
    X["num_photos"] = X["photos"].apply(len)
    X["num_features"] = X["features"].apply(len)
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    X = X[["bathrooms", "bedrooms", "latitude", "longitude", "price",
                 "num_photos", "num_features", "num_description_words",
                 "created_year", "created_month", "created_day"]]
    return X













