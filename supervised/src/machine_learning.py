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


def print_test_results(test: pd.DataFrame, y_prediction: pd.Series) -> None:
    print("Test results:")
    results = test[["artist_id", "listen_%"]].copy()
    results.rename(columns = {"listen_%": "target"}, inplace = True)
    results["prediction"] = y_prediction
    results["error"] = abs(results["target"] - results["prediction"])
    results = results.merge(artists[["artist_id", "name"]], on = "artist_id")
    results.sort_values(by = "artist_id", inplace = True)
    results = results[["artist_id", "name", "target", "prediction", "error"]]

    print(results.to_string())

"""
Function that receives an user ID and prints the top artist recommendations for that user.
"""
def recommend_for_user(user: int, n = 20):
    # check if user exists in dataset
    if user not in all_users:
        print(f"ERROR: User {user} is not in the dataset.")
        exit()

    # get user table from .csv file or generate it
    if not os.path.exists(f"../data/generated-tables/user_{user}_table.csv"):
        build_user_table(user)
    
    user_table = pd.read_csv(f"../data/generated-tables/user_{user}_table.csv", header = 0, index_col = "Unnamed: 0")

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

    print(f"\n\n-- Training models for user {user} --")
    print(f"train data: {len(train)} samples")
    print(f"test data: {len(test)} samples")

    """ K-Nearest Neighbors """
    k = 3
    KNN = KNeighborsRegressor(n_neighbors = k, weights = "distance")

    print(f"\n-- K-Nearest Neighbors (k = {k}) --")
    KNN.fit(X_train, y_train)    
    knn_test = KNN.predict(X_test)

    knn_mae = mean_absolute_error(y_test, knn_test)
    print(f"KNN mean absolute error: {knn_mae}")
    print_test_results(test, knn_test)

    """ Random Forest """
    print("\n-- Random Forest --")
    RF = RandomForestRegressor(n_estimators = 15)
    RF.fit(X_train, y_train)
    rf_test = RF.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_test)
    print(f"RF mean absolute error: {rf_mae}")
    print_test_results(test, rf_test)


    """ Ada-Boosted Decision Tree """
    print("\n-- Ada-Boosted Decision Tree --")
    DT = DecisionTreeRegressor()
    ABDT = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = 100)

    DT.fit(X_train, y_train)
    ABDT.fit(X_train, y_train)
    dt_test = DT.predict(X_test)
    abdt_test = ABDT.predict(X_test)

    dt_mae = mean_absolute_error(y_test, dt_test)
    abdt_mae = mean_absolute_error(y_test, abdt_test)
    print(f"DT mean absolute error: {dt_mae}")
    print(f"ABDT mean absolute error: {abdt_mae}")

    if (abdt_mae >= dt_mae): # if ABDT performed worse than single DT, abandon it
        print("(Will use single Decision Tree for recommendations.)")
        abdt_label = "Single Decision Tree"
        ABDT = DT
        abdt_test = dt_test
    
    else:
        print("(Will use Ada-Boosted Decision Tree for recommendations.)")

    print_test_results(test, abdt_test)

    print("\nFinal results:")
    print(f"method  mean absolute error")
    print(f"KNN:    {knn_mae}")
    print(f"RF:     {rf_mae}")
    print(f"(AB)DT: {abdt_mae}")

    best = "KNN" if (min(knn_mae, rf_mae, abdt_mae) == knn_mae) else "RF" if (min(knn_mae, rf_mae, abdt_mae) == rf_mae) else "(AB)DT" if (min(knn_mae, rf_mae, abdt_mae) == abdt_mae) else "?"
    print(f"BEST: {best}")

    # predicting for all artists by merging recommendations from all 3 methods

    artists_table = pd.read_csv("../data/generated-tables/final_artists_table.csv", header = 0, index_col = "Unnamed: 0")

    knn_prediction = KNN.predict(artists_table.iloc[:, 1:]) # all artists, all tag columns
    rf_prediction = KNN.predict(artists_table.iloc[:, 1:])
    abdt_prediction = ABDT.predict(artists_table.iloc[:, 1:])

    recommendations = pd.DataFrame.from_dict(data = {"artist_id": artists_table["artist_id"], "KNN": knn_prediction, "RF": rf_prediction, "(AB)DT": abdt_prediction})
    recommendations = recommendations.merge(artists[["artist_id", "name"]], on = "artist_id")
    recommendations.rename(columns = {"artist_id": "ID", "name": "Artist"}, inplace = True)
    recommendations["Avg"] = (recommendations["KNN"] + recommendations["RF"] + recommendations["(AB)DT"]) / 3
    recommendations["New?"] =  ~(recommendations["ID"].isin(user_table["artist_id"]))
    recommendations.sort_values(by = "Avg", ascending = False, inplace = True)
    recommendations = recommendations[["ID", "Artist", "New?", "KNN", "RF", "(AB)DT", "Avg"]]

    print(f"\nTOP {n} RECOMMENDATIONS FOR USER {user}")
    print(recommendations.head(n).to_string())

    # TODO: graph

    return


liked_artists = user_artists[["user_id", "artist_id"]].copy()
liked_artists = liked_artists.merge(artists[["artist_id", "name"]], on = "artist_id")
liked_artists.rename(columns = {"name": "artist_name"}, inplace = True)
liked_artists.sort_values(by = "user_id", inplace = True)

# testing with random generated users
user = np.random.choice(list(all_users))
recommend_for_user(user, n = 5)