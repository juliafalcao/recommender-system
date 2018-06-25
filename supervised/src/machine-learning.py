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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


"""
dataset
"""
artists = pd.read_table("../data/artists.dat", sep = "\t", header = 0, names = ["artist_id", "name", "url", "image"])
artists = artists.drop(["image"], axis = 1)

with codecs.open("../data/tags.dat", encoding = "utf-8", errors = "replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep = "\t", header = 0, names = ["tag_id", "tag"])
f.close()

user_artists = pd.read_table("../data/user_artists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "listen_count"])

user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)
uta = user_tagged_artists

"""
final generated table
user_id / tag1 / tag2 / tag3 / tag4 / tag5 / artist_id
"""
data = pd.read_csv("final.csv", header = 0)


"""
machine learning!!
"""
train: pd.DataFrame
test: pd.DataFrame
train, test = train_test_split(data, test_size = 0.3)

# columns: 0 = user_id, 1 = artist_id, 2... = tags
X_train: pd.DataFrame = train.iloc[:, 2:] # all tags
y_train: pd.Series = train.iloc[:, 1] # only artist
X_test: pd.DataFrame = test.iloc[:, 2:] # all tags
y_test: pd.Series = test.iloc[:, 1] # only artist


# KNN
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
knn_predictions = KNN.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN accuracy: {knn_accuracy}")


# Decision Tree (MEMORY ERROR)
"""
DT = DecisionTreeClassifier(criterion = "entropy")
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


# Multi-Layer Perceptron (MLP)
"""
MLP = MLPClassifier()
MLP.fit(X_train, y_train)
mlp_predictions = MLP.predict(X_test)

mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"MLP accuracy: {mlp_accuracy}")
"""