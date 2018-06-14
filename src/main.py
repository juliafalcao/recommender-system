import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # eventually
import codecs
import typing

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
total_listen_counts = user_artists.groupby(["user_id"])["listen_count"].sum()
total_listen_counts = pd.DataFrame(total_listen_counts).reset_index()

"""
tag_listen_counts: user_id / tag_id / listen_count
listen_count = sum of listen counts for the artists user_id listens to which are tagged as tag_id
"""
tag_listen_counts = user_tagged_artists[["user_id", "tag_id", "artist_id"]].merge(user_artists, on = ["user_id", "artist_id"])
tag_listen_counts.sort_values(by = ["user_id", "tag_id"], inplace = True)
tag_listen_counts = tag_listen_counts.groupby(["user_id", "tag_id"])["listen_count"].sum()
tag_listen_counts = pd.DataFrame(tag_listen_counts).reset_index()
# print(tag_listen_counts)

"""
tag_weights_per_user: user_id / tag_id / (tag_listen_count / user_listen_count) / tag_weight
based on how many times user_id has listened to artists who are tagged as tag_id
"""
tag_listen_counts.rename(columns = {"listen_count": "tag_listen_count"}, inplace = True)
tag_weights_per_user = tag_listen_counts.merge(total_listen_counts, on = "user_id")
tag_weights_per_user.rename(columns = {"listen_count": "user_listen_count"}, inplace = True)
tag_weights_per_user["tag_weight"] = tag_weights_per_user["tag_listen_count"] / tag_weights_per_user["user_listen_count"]
tag_weights_per_user.drop(["tag_listen_count", "user_listen_count"], axis = 1, inplace = True)


# tests (completely failed)
"""
total_listen_count = int(total_listen_counts[total_listen_counts["user_id"] == user]["listen_count"])
artist_weight = listen_count / total_listen_count
print(f"original weight: {artist_weight}")

"""


"""
final table
user_id / artist_id / tag1 / tag2 / tag3 / tag4 / tag5 / listen_count
tags 1-5: tag IDs of tags chosen by heighest weights for the artist
for each prediction, use only final[final["user_id"] == user]
"""

final = user_artists[["user_id", "artist_id"]].sort_values(by = "user_id")


# TESTING: artists that have less than 5 tags
tag_counts = tag_weights_per_artist.groupby(["artist_id"])["tag_id"].count()
tag_counts = pd.DataFrame(tag_counts).reset_index()
tag_counts.rename(columns = {"tag_id": "tag_count"}, inplace = True)
# print(tag_counts[tag_counts["tag_count"] < 5])

# find highest rated tags for each artist
best_tags_per_artist = tag_weights_per_artist.sort_values(by = ["artist_id", "tag_weight"])
best_tags_per_artist = best_tags_per_artist.reset_index().drop("index", axis = 1)

def recommend_for_user(user: int, n: int = 5):
    # TODO: check if user exists

    artists = user_artists.loc[user_artists["user_id"] == user]
    artists = artists.drop("user_id", axis = 1)
    total_listen_count = artists["listen_count"].sum()
    artists["listen_%"] = artists["listen_count"] / total_listen_count
    
    # print(artists.head())
    
    # print(best_tags_per_artist)
    artists["top_tags"] = artists.apply(get_top_tags)
    # setar colunas tag1 até tag5
    print(artists)


def get_top_tags(artist: int) -> tuple:
    top_tags = best_tags_per_artist[best_tags_per_artist["artist_id"] == artist]
    top_tags = best_tags_per_artist.ix[0:4, "tag_id"] if len(top_tags) > 5 else best_tags_per_artist.ix[0:(len(top_tags)-1), "tag_id"]
    return tuple(top_tags)


    # user_artists: user_id / artist_id / listen_count
    # user_tagged_artists: user_id / artist_id / tag_id / (day / month / year)


recommend_for_user(2)
