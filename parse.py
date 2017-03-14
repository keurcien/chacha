import numpy as np
import pandas as pd
from collections import Counter

def target(X):
    target = X["interest_level"]
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
    #From parse_1, with most common feature   

    X["num_photos"] = X["photos"].apply(len)
    X["num_features"] = X["features"].apply(len)
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X["created"] = pd.to_datetime(X["created"])
    X["created_year"] = X["created"].dt.year
    X["created_month"] = X["created"].dt.month
    X["created_day"] = X["created"].dt.day
    
    entriesToParse = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
                      "num_photos", "num_features", "num_description_words",
                      "created_year", "created_month", "created_day"]
    
    # Looking at the most common features in an subset
    size_subset = 200
    n_features = 20
    rows = np.random.choice(X.index.values, size_subset)
    sampled_features = X["features"].ix[rows]
    allFeatures=[]
    for row in rows:
        allFeatures.extend(sampled_features.loc[[row]].values[0])
    features_to_count = (word for word in allFeatures if word[:1])
    features_count = Counter(features_to_count)
    common_features = map(lambda x:x[0],features_count.most_common(n_features))
    common_features_rows= map(lambda x:'with_'+x.lower().replace(" ","_"),common_features)
   
    for common_feature,common_feature_row in zip(common_features,common_features_rows):
        X[common_feature_row] = X["features"].apply(lambda x: any(common_feature in feature for feature in x))   
    X = X[entriesToParse+common_features_rows]
    return X













