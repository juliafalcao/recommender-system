import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import typing

pd.set_option('display.float_format', lambda x: '%.6f' % x) # display floats with 6 decimal digits


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
data cleaning
"""

# removing least used tags
tag_use_threshold = 6
uta = user_tagged_artists
tags_to_keep = pd.DataFrame(uta.groupby("tag_id")["artist_id"].count()).reset_index()
tags_to_keep.rename(columns = {"artist_id": "use_count"}, inplace = True)
tags_to_keep.sort_values(by = "use_count", ascending = True, inplace = True)
tags_to_keep = tags_to_keep[tags_to_keep["use_count"] >= tag_use_threshold]
tags_to_keep = tags_to_keep["tag_id"]
uta = uta[uta["tag_id"].isin(tags_to_keep)] # remove from user_tagged_artists table
user_tagged_artists = uta # update original

# removing least tagged artists
tag_count_threshold = 4
uta = user_tagged_artists
artists_to_keep = pd.DataFrame(uta.groupby("artist_id")["tag_id"].count()).reset_index()
artists_to_keep.rename(columns = {"tag_id": "tag_count"}, inplace = True)
artists_to_keep = artists_to_keep[artists_to_keep["tag_count"] > tag_count_threshold]
artists_to_keep = artists_to_keep["artist_id"]
uta = uta[uta["artist_id"].isin(artists_to_keep)] # removing from user_tagged_artists table
user_tagged_artists = uta
user_artists = user_artists[user_artists["artist_id"].isin(artists_to_keep)] # removing from user_artists table

