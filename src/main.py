import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # eventually
import codecs

"""
read dataset into dataframes

artists: id / name / url
tags: id / tag
user_artists: user_id / artist_id / listen_count
user_tagged_artists: user_id / artist_id / tag_id / day / month / year
"""

artists = pd.read_table("../data/artists.dat", sep = "\t", header = 0, index_col = 0, names = ["id", "name", "url", "image"])
# index column: "id"
artists = artists.drop(["image"], axis = 1)

with codecs.open("../data/tags.dat", encoding = "utf-8", errors = "replace") as f: # to avoid encoding error
    tags = pd.read_table(f, sep = "\t", header = 0, index_col = 0, names = ["id", "tag"])
    # index column: "id"
f.close()

user_artists = pd.read_table("../data/user_artists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "listen_count"])
# index column: default

user_tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
user_tagged_artists = user_tagged_artists.drop(["day", "month", "year"], axis = 1)
# index column: default

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

tag_weights: artist_id / tag_id / user_count / tag_weight
where tag_weight = (users who tagged artist_id as tag_id) / (total users who tagged artist_id)

"""

uta = user_tagged_artists
total_users: int = uta["user_id"].nunique()
tag_weights: pd.DataFrame = uta[["artist_id", "tag_id"]]
user_counts: pd.Series = uta.groupby(["artist_id", "tag_id"])["user_id"].count()
user_counts = pd.DataFrame(user_counts)
tag_weights = tag_weights.merge(user_counts, on = ["artist_id", "tag_id"])
tag_weights.rename(columns = {"user_id": "user_count"}, inplace = True)
tag_weights.drop_duplicates(keep = "first", inplace = True)
tag_weights.sort_values(by = "artist_id", inplace = True)

artist_tagged_times: pd.Series = uta.groupby(["artist_id"])["user_id"].count()
artist_tagged_times = pd.DataFrame(artist_tagged_times)
tag_weights = tag_weights.merge(artist_tagged_times, on = "artist_id")
tag_weights.rename(columns = {"user_id": "total_taggings"}, inplace = True)
tag_weights["tag_weight"] = tag_weights["user_count"] / tag_weights["total_taggings"]

print(tag_weights)
