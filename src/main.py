import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # eventually
import codecs

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
print(tag_weights_per_user)




# tests

user = 2
artist = 51
listen_count = 13883

total_listen_count = int(total_listen_counts[total_listen_counts["user_id"] == user]["listen_count"])
artist_weight = listen_count / total_listen_count
print(f"original weight: {artist_weight}")

artist_new_weight = 0
tags = 0

for row in uta[uta["artist_id"] == artist].iterrows():
    row = row[1]
    
    bonus = 2 if row["user_id"] == user else 1
    tag = row["tag_id"]
    twpa = tag_weights_per_artist
    tag_weight_for_artist = twpa[(twpa["artist_id"] == artist) & (twpa["tag_id"] == tag)]["tag_weight"]
    twpu = tag_weights_per_user
    tag_weight_for_user = twpu[(twpu["user_id"] == user) & (twpa["tag_id"] == tag)]["tag_weight"]
    if tag_weight_for_user.empty: tag_weight_for_user = 0

    artist_new_weight += bonus * float(tag_weight_for_user) * float(tag_weight_for_artist)
    tags += 1
    

print(f"calculated weight: {artist_new_weight}")