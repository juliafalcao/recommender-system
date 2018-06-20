import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import typing

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


"""
dataset
"""
artists = pd.read_table("../data/artists.dat", sep = "\t", header = 0, names = ["artist_id", "name", "url", "image"])
artists = artists.drop(["image"], axis = 1)

with codecs.open("../data/tags.dat", encoding = "utf-8", errors = "replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep = "\t", header = 0, names = ["tagid", "tag"])
f.close()

user_artists = pd.read_table("../data/user_artists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "listen_count"])

user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tagid", "day", "month", "year"])
user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)
uta = user_tagged_artists

"""
final generated table
user_id / tag1 / tag2 / tag3 / tag4 / tag5 / artist_id
"""
data = pd.read_csv("final.csv", header = 0, names = ["user_id", "tag1", "tag2", "tag3", "tag4", "tag5", "artist_id"])

"""
Function that replaces tag and artist IDs with names for better visualization of the data.
"""
def visualization(data: pd.DataFrame) -> pd.DataFrame:
    data = data.merge(tags, left_on = "tag1", right_on = "tagid")
    data = data.drop("tag1", axis = 1).rename(columns = {"tag": "tag1"})
    data = data.merge(tags, left_on = "tag2", right_on = "tagid")
    data = data.drop("tag2", axis = 1).rename(columns = {"tag": "tag2"})
    data = data.merge(tags, left_on = "tag3", right_on = "tagid")
    data = data.drop("tag3", axis = 1).rename(columns = {"tag": "tag3"})
    data = data.merge(tags, left_on = "tag4", right_on = "tagid")
    data = data.drop("tag4", axis = 1).rename(columns = {"tag": "tag4"})
    data = data.merge(tags, left_on = "tag5", right_on = "tagid")
    data = data.drop("tag5", axis = 1).rename(columns = {"tag": "tag5"})
    data = data.merge(artists, on = "artist_id")
    data = data.drop("artist_id", axis = 1).rename(columns = {"name": "artist_name"})
    data = data[["user_id", "tag1", "tag2", "tag3", "tag4", "tag5", "artist_name"]]
    data.sort_values(by = "user_id")
    
    return data


visual = visualization(data)

"""
machine learning!!
"""
train: pd.DataFrame
test: pd.DataFrame
train, test = train_test_split(data, test_size = 0.2)
X_train: pd.DataFrame = train.iloc[:, 0:-1] # features (user_id and tags) # TODO: only tags maybe?
y_train: pd.Series = train.iloc[:, -1] # only artist
X_test: pd.DataFrame = test.iloc[:, 0:-1] # features
y_test: pd.Series = test.iloc[:, -1] # only artist

# KNN
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN.fit(X_train, y_train)
knn_predictions = KNN.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN accuracy: {knn_accuracy}")

# Decision Tree (MEMORY ERROR)
"""
DT = DecisionTreeClassifier(max_depth = 20)
DT.fit(X_train, y_train)
dt_predictions = DT.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"DT accuracy: {dt_accuracy}")
"""

# Random Forest (MEMORY ERROR)
"""
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
rf_predictions = RF.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"RF accuracy: {rf_accuracy}")
"""

# Gaussian Naive-Bayes (MEMORY ERROR)
"""
GNB = GaussianNB()
GNB.fit(X_train, y_train)
gnb_predictions = GNB.predict(X_test)

gnb_accuracy = accuracy_score(y_test, gnb_predictions)
print(f"GNB accuracy: {gnb_accuracy}")
"""
