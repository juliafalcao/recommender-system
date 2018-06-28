# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs, typing, os

from data_manipulation import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

pd.set_option('display.float_format', lambda x: '%.7f' % x) # display floats with 6 decimal digits

"""
dataset (after data cleaning)
"""
artists = pd.read_csv("../data/cleaned/artists.csv", header = 0, index_col = "Unnamed: 0")

with codecs.open("../data/cleaned/tags.csv", encoding = "utf-8", errors = "replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep = ",", header = 0, index_col = "Unnamed: 0")
f.close()

user_artists = pd.read_csv("../data/cleaned/user_artists.csv", header = 0, index_col = "Unnamed: 0")
all_users = set(user_artists["user_id"])

# user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
# user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)

user_tagged_artists = pd.read_csv("../data/cleaned/user_tagged_artists.csv", header = 0, index_col = "Unnamed: 0")
uta = user_tagged_artists

"""
Function that receives an user ID and prints the top artist recommendations for that user.
"""
def recommend_for_user(user: int, n = 20):
    # check if user exists in dataset
    if user not in all_users:
        print(f"ERROR: User {user} is not in the dataset.")
        exit()
    
    print(f"USER: {user}")

    # get user table from .csv file or generate it
    if not os.path.exists(f"../data/user-tables/user_{user}_table.csv"):
        build_user_table(user)
    
    user_table = pd.read_csv(f"../data/user-tables/user_{user}_table.csv", header = 0, index_col = "Unnamed: 0")

    """
    machine learning!!
    """
    user_table.sort_values(by = "artist_id", inplace = True)
    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(user_table, test_size = 0.2)

    # columns: 0 = artist_id, 1 through -2 = all tags (features), -1 = listen_% (target)
    X_train: pd.DataFrame = train.iloc[:, 1:-1] # all tags
    y_train: pd.Series = train.iloc[:, -1] # only target
    X_test: pd.DataFrame = test.iloc[:, 1:-1] # all tags
    y_test: pd.Series = test.iloc[:, -1] # only target

    print(f"\n\n-- training models for user {user} --")
    print(f"train data: {len(train)} samples")
    print(f"test data: {len(test)} samples")

    """ K-Nearest Neighbors """
    k = 7
    KNN = KNeighborsRegressor(n_neighbors = k)
    print(f"\n-- K-Nearest Neighbors (k = {k}) --")
    KNN.fit(X_train, y_train)    
    knn_test = KNN.predict(X_test)

    # metrics
    knn_mae = mean_absolute_error(y_test, knn_test)
    print(f"KNN mean absolute error: {knn_mae}")

    print("test results:")
    knn_results = test[["artist_id", "listen_%"]].copy()
    knn_results.rename(columns = {"listen_%": "target"})
    knn_results["prediction"] = knn_test
    knn_results["error"] = abs(knn_results["listen_%"] - knn_results["prediction"])
    knn_results.sort_values(by = "artist_id", inplace = True)
    print(knn_results.to_string())

    # graph
    """
    x_data = test["artist_id"]
    plt.scatter(x_data, y_test, color = "darkturquoise", label = "Test data")
    plt.scatter(x_data, knn_test, color = "crimson", label = "KNN prediction")
    plt.title(f"KNN (user {user})")
    plt.xlabel("Artist")
    plt.ylabel("Listen %")
    plt.legend()
    # plt.show()
    """

    """ Random Forest Regressor """
    print("\n-- Random Forest Regressor --")
    RF = RandomForestRegressor()
    RF.fit(X_train, y_train)
    rf_test = RF.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_test)
    print(f"RF mean absolute error: {rf_mae}")


    """ Ada Boosted Decision Tree Regressor """
    print("\n-- Ada Boosted Decision Tree Regressor --")
    DT = DecisionTreeRegressor(criterion = "mse")
    ABDT = AdaBoostRegressor(DecisionTreeRegressor(criterion = "mse"), n_estimators = 100)

    DT.fit(X_train, y_train)
    ABDT.fit(X_train, y_train)
    dt_test = DT.predict(X_test)
    abdt_test = ABDT.predict(X_test)

    dt_mae = mean_absolute_error(y_test, dt_test)
    abdt_mae = mean_absolute_error(y_test, abdt_test)
    print(f"DT mean absolute error: {dt_mae}")
    print(f"ABDT mean absolute error: {abdt_mae}")

    if (abdt_mae >= dt_mae): # if ABDT performed worse than single DT, abandon it
        print("Using single-estimator DT")
        ABDT = DT
        abdt_test = dt_test
    
    else:
        print("Using Ada-Boosted DT")

    print("ABDT Test Results:")
    abdt_results = test[["artist_id", "listen_%"]].copy()
    abdt_results.rename(columns = {"listen_%": "target"})
    abdt_results["prediction"] = abdt_test
    abdt_results["error"] = abs(abdt_results["listen_%"] - abdt_results["prediction"])
    abdt_results.sort_values(by = "artist_id", inplace = True)
    print(abdt_results.to_string())

    # predicting for all artists by merging recommendations from all 3 methods

    artists_table = pd.read_csv("../data/final_artists_table.csv", header = 0, index_col = "Unnamed: 0")

    knn_prediction = KNN.predict(artists_table.iloc[:, 1:]) # all artists, all tag columns
    rf_prediction = KNN.predict(artists_table.iloc[:, 1:])
    abdt_prediction = ABDT.predict(artists_table.iloc[:, 1:])

    recommendations = pd.DataFrame.from_dict({"artist_id": artists_table["artist_id"], "KNN": knn_prediction, "RF": rf_prediction, "ABDT": abdt_prediction})
    recommendations = recommendations.merge(artists[["artist_id", "name"]], on = "artist_id")
    print(recommendations.head().to_string())
    recommendations.rename(columns = {"artist_id": "ID", "name": "Artist"}, inplace = True)
    recommendations["Avg"] = (recommendations["KNN"] + recommendations["RF"] + recommendations["ABDT"]) / 3
    recommendations["New?"] =  ~(recommendations["ID"].isin(user_table["artist_id"]))
    recommendations.sort_values(by = "Avg", ascending = False, inplace = True)
    recommendations = recommendations[["ID", "Artist", "New?", "KNN", "RF", "ABDT", "Avg"]]

    print(f"\nTOP {n} RECOMMENDATIONS FOR USER {user}")
    print(recommendations.head(n).to_string())

"""
liked_artists = user_artists[["user_id", "artist_id"]].merge(artists[["artist_id", "name"]], on = "artist_id")
liked_artists.rename(columns = {"name": "artist_name"}, inplace = True)
liked_artists.sort_values(by = "user_id", inplace = True)
print(liked_artists.to_string())
"""

# testing with random generated users
user = np.random.choice(list(all_users))
# recommend_for_user(user)

recommend_for_user(47)
