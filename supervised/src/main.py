import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import typing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


"""
read dataset into dataframes

artists: id / name / url
tags: id / tag
user_artists: user_id / artist_id / listen_count
user_tagged_artists: user_id / artist_id / tag_id / (day / month / year)

ps.: all indexes are default (unnamed, starting from 0)
"""

artists = pd.read_table("../data/artists.dat", sep = "\t", header = 0, names = ["id", "name", "url", "image"])
artists = artists.drop(["image"], axis = 1)

with codecs.open("../data/tags.dat", encoding = "utf-8", errors = "replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep = "\t", header = 0, names = ["id", "tag"])
f.close()

user_artists = pd.read_table("../data/user_artists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "listen_count"])

user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)

"""
print("\nARTISTS")
print(artists.head().to_string())
print("\nTAGS")
print(tags.head().to_string())
print("\nUSER ARTISTS")
print(user_artists.head().to_string())
print("\nUSER TAGGED ARTISTS")
print(user_tagged_artists.head().to_string())
"""


"""
intermediate tables
"""

"""
tag_weights_per_artist: artist_id / tag_id / (user_count / total_taggings) / tag_weight
based on how many users tagged an artist as a certain tag out of all users who tagged that artist
user_count = users who tagged artist_id as tag_id
total_taggings = total users who tagged artist_id as any tag
tag_weight = user_count / total_taggings
"""
uta = user_tagged_artists
total_users: int = uta["user_id"].nunique()
tag_weights_per_artist: pd.DataFrame = uta[["artist_id", "tag_id"]]
user_counts: pd.Series = uta.groupby(["artist_id", "tag_id"])["user_id"].count()
user_counts = pd.DataFrame(user_counts)
tag_weights_per_artist = tag_weights_per_artist.merge(user_counts, on = ["artist_id", "tag_id"])
tag_weights_per_artist.rename(columns = {"user_id": "user_count"}, inplace = True)
tag_weights_per_artist.drop_duplicates(keep = "first", inplace = True)
tag_weights_per_artist.sort_values(by = "artist_id", inplace = True)

artist_tagged_times: pd.Series = uta.groupby(["artist_id"])["user_id"].count()
artist_tagged_times = pd.DataFrame(artist_tagged_times)
tag_weights_per_artist = tag_weights_per_artist.merge(artist_tagged_times, on = "artist_id")
tag_weights_per_artist.rename(columns = {"user_id": "total_taggings"}, inplace = True)
tag_weights_per_artist["tag_weight"] = tag_weights_per_artist["user_count"] / tag_weights_per_artist["total_taggings"]
tag_weights_per_artist.drop(["user_count", "total_taggings"], axis = 1, inplace = True)

"""
total_listen_counts: user_id / listen_count
listen_count = sum of listen counts for all the artists that user_id listens to
"""
# ABANDONED FOR NOW
"""
total_listen_counts = user_artists.groupby(["user_id"])["listen_count"].sum()
total_listen_counts = pd.DataFrame(total_listen_counts).reset_index()
"""

"""
tag_listen_counts: user_id / tag_id / listen_count
listen_count = sum of listen counts for the artists user_id listens to which are tagged as tag_id
"""
# ABANDONED FOR NOW
"""
tag_listen_counts = user_tagged_artists[["user_id", "tag_id", "artist_id"]].merge(user_artists, on = ["user_id", "artist_id"])
tag_listen_counts.sort_values(by = ["user_id", "tag_id"], inplace = True)
tag_listen_counts = tag_listen_counts.groupby(["user_id", "tag_id"])["listen_count"].sum()
tag_listen_counts = pd.DataFrame(tag_listen_counts).reset_index()
# print(tag_listen_counts)
"""

"""
tag_weights_per_user: user_id / tag_id / (tag_listen_count / user_listen_count) / tag_weight
based on how many times user_id has listened to artists who are tagged as tag_id
"""
# ABANDONED FOR NOW
"""
tag_listen_counts.rename(columns = {"listen_count": "tag_listen_count"}, inplace = True)
tag_weights_per_user = tag_listen_counts.merge(total_listen_counts, on = "user_id")
tag_weights_per_user.rename(columns = {"listen_count": "user_listen_count"}, inplace = True)
tag_weights_per_user["tag_weight"] = tag_weights_per_user["tag_listen_count"] / tag_weights_per_user["user_listen_count"]
tag_weights_per_user.drop(["tag_listen_count", "user_listen_count"], axis = 1, inplace = True)
"""

# ANALYSIS: artists that have less than 5 tags (yikes)
"""
tag_counts = tag_weights_per_artist.groupby(["artist_id"])["tag_id"].count()
tag_counts = pd.DataFrame(tag_counts).reset_index()
tag_counts.rename(columns = {"tag_id": "tag_count"}, inplace = True)
# print(tag_counts[tag_counts["tag_count"] < 5])
"""

"""
best_tags_per_artist: artist_id / tag_weight
in order of highest to lower tag weight
"""
best_tags_per_artist = tag_weights_per_artist.sort_values(by = ["artist_id", "tag_weight"])
best_tags_per_artist = best_tags_per_artist.reset_index().drop("index", axis = 1)

"""
function that returns the 5 highest weighted tags for an artist
as a tuple filled with zeroes in case the artist has less than 5 tags
"""
def get_top_tags(artist: int) -> tuple:
    top_tags = best_tags_per_artist.loc[best_tags_per_artist["artist_id"] == artist]
    
    if len(top_tags) > 5:
        top_tags = best_tags_per_artist.ix[0:4, "tag_id"]
    
    else:
        top_tags = best_tags_per_artist.ix[0:(len(top_tags)-1), "tag_id"]
        top_tags = top_tags.append(pd.Series([0] * (5 - len(top_tags))))
        # fill with zeroes when artist has less than 5 tags
        # TODO: maybe negative integers so the algorithm won't think all zeroes are the same tag?
    
    return tuple(top_tags)

"""
function that receives an user_id and returns three tables
columns: artist_id / tag1 / tag2 / tag3 / tag4 / tag5 / listen_%
all_artists_for_user -> all existing artists and listen_%, NaN when the user hasn't listened to them
liked_artists -> only artists the user knows and listen_% (for training)
unknown_artists -> only artists the user doesn't know and listen_% = NaN (for recommending new artists)
"""
def user_x_all_artists_table(user: int) -> pd.DataFrame:
    all_artists = pd.DataFrame({"artist_id": artists["id"]})

    all_artists_for_user = user_artists.loc[user_artists["user_id"] == user]
    total_listen_count = all_artists_for_user["listen_count"].sum()
    all_artists_for_user["listen_%"] = all_artists_for_user["listen_count"].apply(lambda row: row / total_listen_count)
    # TODO: deal with setting with copy warning??
    all_artists_for_user.drop(["user_id", "listen_count"], axis = 1, inplace = True)
    liked_artist_ids = all_artists_for_user["artist_id"] # before merge
    all_artists_for_user = all_artists_for_user.merge(all_artists, how = "right", on = "artist_id")
    
    all_artists_for_user["top_tags"] = all_artists_for_user["artist_id"].apply(get_top_tags)
    all_artists_for_user[["tag1", "tag2", "tag3", "tag4", "tag5"]] = all_artists_for_user["top_tags"].apply(pd.Series)
    all_artists_for_user.drop("top_tags", axis = 1, inplace = True)
    all_artists_for_user = all_artists_for_user[["artist_id", "tag1", "tag2", "tag3", "tag4", "tag5", "listen_%"]]

    liked_artists = all_artists_for_user[all_artists_for_user["artist_id"].isin(liked_artist_ids)]
    unknown_artists = all_artists_for_user - liked_artists
    return all_artists_for_user, liked_artists, unknown_artists


def recommend_for_user(user: int):
    # TODO: check if user exists

    all_artists, liked_artists, _ = user_x_all_artists_table(user)

    # split into training and testing sets
    """
    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(liked_artists, test_size = 0.2)
    X_train: pd.DataFrame = train.iloc[:, 0:-1] # all features
    y_train: pd.Series = train.iloc[:, -1] # only target
    X_test: pd.DataFrame = test.iloc[:, 0:-1] # all features
    y_test: pd.Series = test.iloc[:, -1] # only target
    """

    # KNN
    """
    KNN = KNeighborsRegressor()
    KNN.fit(X_train, y_train) # train model
    knn_predictions = KNN.predict(X_test)
    """

    X_train = liked_artists.iloc[:, 0:-1] # features
    y_train = liked_artists.iloc[:, -1] # target

    # Decision Tree
    DT = DecisionTreeRegressor()
    DT.fit(X_train, y_train)

    X_predict = all_artists.iloc[:, 0:-1]
    y_real = all_artists.iloc[:, -1] # all listen_%, including NaN

    # predicting for all artists, known or not
    dt_predictions = DT.predict(X_predict)
    
    return dt_predictions, y_real


# all_users = user_artists["user_id"].drop_duplicates(keep = "first")

# predict for all artists, one user
all_artists = artists["id"]
predictions, y_real = recommend_for_user(846)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xlabel("artist")
ax.set_ylabel("listen %")

ax.scatter(all_artists, y_real, color = "darkviolet", s = 30, label = "real values", alpha = 0.5)
ax.scatter(all_artists, predictions, color = "black", s = 5, label = "DT predictions")

ax.legend()
plt.savefig("dt_predictions.png")
plt.show()


# NOTE: training with few artists and predicting for 17k of them is Bad
