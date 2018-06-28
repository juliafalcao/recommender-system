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
def recommend_for_user(user: int):
    # check if user exists in dataset
    if user not in all_users:
        print(f"ERROR: User {user} is not in the dataset.")
        exit()

    # get user table from .csv file or generate it
    if not os.path.exists(f"../data/user-tables/user_{user}_table.csv"):
        build_user_table(user)
    
    user_table = pd.read_csv(f"../data/user-tables/user_{user}_table.csv", header = 0, index_col = "Unnamed: 0")

    print(user_table.head(50))

    """
    machine learning!!
    """
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
    KNN.fit(X_train, y_train)

    print(f"\n-- K-Nearest Neighbors (k = {k}) --")
    
    knn_test = KNN.predict(X_test)

    # metrics
    knn_mae = mean_absolute_error(y_test, knn_test)
    print(f"mean absolute error: {knn_mae}")

    print("test results: ")
    comparison = test[["artist_id", "listen_%"]].copy()
    comparison.rename(columns = {"listen_%": "target"})
    comparison["prediction"] = knn_test
    comparison["error"] = abs(comparison["listen_%"] - comparison["prediction"])
    comparison.sort_values(by = "artist_id", inplace = True)
    print(comparison.to_string())

    # graph
    x_data = test["artist_id"]
    plt.scatter(x_data, y_test, color = "darkturquoise", label = "Test data")
    plt.scatter(x_data, knn_test, color = "crimson", label = "KNN prediction")
    plt.title(f"KNN (user {user})")
    plt.xlabel("Artist")
    plt.ylabel("Listen %")
    plt.legend()
    # plt.show()

    """ Decision Tree """
    DT = DecisionTreeRegressor()
    DT.fit(X_train, y_train)

    print(f"-- Decision Tree --")

    dt_test = DT.predict(X_test)

    # metrics
    dt_mae = mean_absolute_error(y_test, dt_test)
    print(f"DT mean absolute error: {dt_mae}")

    print("test results: ")
    comparison = test[["artist_id", "listen_%"]].copy()
    comparison.rename(columns = {"listen_%": "target"})
    comparison["prediction"] = dt_test
    comparison["error"] = abs(comparison["listen_%"] - comparison["prediction"])
    comparison.sort_values(by = "artist_id", inplace = True)
    print(comparison.to_string())

    # predicting for all artists

    artists_table = pd.read_csv("../data/final_artists_table.csv", header = 0, index_col = "Unnamed: 0")
    
    knn_prediction = KNN.predict(artists_table.iloc[:, 1:]) # all artists, all tag columns
    knn_recommendation = pd.DataFrame({"artist_id": artists_table["artist_id"], "KNN prediction": knn_prediction})
    knn_recommendation = knn_recommendation.merge(artists[["artist_id", "name"]], on = "artist_id")
    knn_recommendation.sort_values(by = "KNN prediction", ascending = False, inplace = True)

    dt_prediction = DT.predict(artists_table.iloc[:, 1:])
    dt_recommendation = pd.DataFrame({"artist_id": artists_table["artist_id"], "DT prediction": dt_prediction})
    dt_recommendation = dt_recommendation.merge(artists[["artist_id", "name"]], on = "artist_id")
    dt_recommendation.sort_values(by = "DT prediction", ascending = False, inplace = True)

    print(dt_recommendation.head(50).to_string(), end = "\n\n")
    print(dt_recommendation.head(50).to_string())


"""
liked_artists = user_artists.merge(artists[["artist_id", "name"]], on = "artist_id")
liked_artists.sort_values(by = ["user_id", "listen_count"], ascending = [True, False], inplace = True)
print(liked_artists[liked_artists["name"] == "Ana Carolina"])
"""

recommend_for_user(2) # TODO: test with random users
