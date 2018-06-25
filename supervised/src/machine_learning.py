import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs, typing, os

from data_manipulation import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import confusion_matrix


"""
dataset (after data cleaning)
"""
artists = pd.read_csv("../data/cleaned_artists.csv", header = 0, index_col = "Unnamed: 0")

with codecs.open("../data/cleaned_tags.csv", encoding = "utf-8", errors = "replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep = ",", header = 0, index_col = "Unnamed: 0")
f.close()

user_artists = pd.read_csv("../data/cleaned_user_artists.csv", header = 0, index_col = "Unnamed: 0")
all_users = set(user_artists["user_id"])

# user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
# user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)

user_tagged_artists = pd.read_csv("../data/cleaned_user_tagged_artists.csv", header = 0, index_col = "Unnamed: 0")
uta = user_tagged_artists


"""
Function that receives an user ID and prints the top 5 artist recommendations for that user.
"""
def recommend_for_user(user: int):
    # check if user exists in dataset
    if user not in all_users:
        print(f"ERROR: User {user} is not in the dataset.")
        exit()

    """
    get user table from csv file (cached) or generate it
    """
    if not os.path.exists(f"user-tables/user_{user}_table.csv"):
        build_user_table(user)
    
    user_table = pd.read_csv(f"user-tables/user_{user}_table.csv", header = 0, index_col = "Unnamed: 0")


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

    # KNN
    KNN = KNeighborsRegressor()
    KNN.fit(X_train, y_train)
    knn_predictions = KNN.predict(X_test)

    comparison = X_test
    comparison["true"] = y_test
    comparison["prediction"] = knn_predictions
    comparison["error"] = abs(comparison["true"] - comparison["prediction"])

    mse = mean_squared_error(y_test, knn_predictions)
    print(f"mean squared error: {mse}")

    print(comparison.head(10))

    """
    predicting for all artists
    """
    artists_table = pd.read_csv("../data/final_artists_table.csv", header = 0, index_col = "Unnamed: 0")

    # print(artists_table.iloc[:5, :25].to_string())
    
    knn_all_predictions = KNN.predict(artists_table)
    recommendations = pd.DataFrame({"artist_id": artists_table["artist_id"], "predicted_listen_%": knn_all_predictions})
    recommendations = recommendations.merge(artists[["artist_id", "name"]], on = "artist_id")

    recommendations.sort_values(by = "predicted_listen_%", ascending = False, inplace = True)
    print(recommendations.head(20).to_string())


recommend_for_user(2)

        

