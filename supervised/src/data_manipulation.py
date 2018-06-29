# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs, typing, os

pd.set_option('display.float_format', lambda x: '%.7f' % x) # display floats with 6 decimal digits

"""
read dataset into dataframes

artists: artist_id / name / url
tags: tag_id / tag
user_artists: user_id / artist_id / listen_count
user_tagged_artists: user_id / artist_id / tag_id / (day / month / year)

ps.: all indexes are default (unnamed, starting from 0)
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
print("\nARTISTS")
print(artists.head().to_string())
print("\nTAGS")
print(tags.head().to_string())
print("\nUSER ARTISTS")
print(user_artists.head().to_string())
print("\nUSER TAGGED ARTISTS")
print(uta.head().to_string())
"""


"""
data cleaning (0)
removing artists that haven't been tagged by any users
"""
print("-- preparing data -- ")

print(f"\ncleaning: removing artists that haven't been tagged or listened to by any users")
before = len(artists)
artists = artists[artists["artist_id"].isin(uta["artist_id"])]
print(f"\tartists: from {before} to {len(artists)}")

before = len(user_artists)
user_artists = user_artists[user_artists["artist_id"].isin(artists["artist_id"])]
print(f"\tuser-artist pairs: from {before} to {len(user_artists)}")

before = len(uta)
uta = uta[(uta["artist_id"].isin(artists["artist_id"])) & (uta["artist_id"].isin(user_artists["artist_id"]))]
print(f"\tuser-tag-artist pairs: from {before} to {len(uta)}")


"""
data cleaning (1)
removing least used tags
"""
tag_uses = uta.groupby("tag_id")[["user_id", "artist_id"]].count()
tag_uses.drop("user_id", axis = 1, inplace = True)
tag_uses.rename(columns = {"artist_id": "uses"}, inplace = True)
avg_tag_uses = int(tag_uses["uses"].mean())
tags_to_keep = tag_uses[tag_uses["uses"] >= avg_tag_uses]
tags_to_keep = tags_to_keep.reset_index()
tags_to_keep = tags_to_keep["tag_id"]

print(f"\ncleaning: removing least used tags (below {avg_tag_uses} uses)")

before = len(tags)
tags = tags[tags["tag_id"].isin(tags_to_keep)] # remove from tags dataframe
print(f"\ttags: from {before} to {len(tags)}")

before = len(uta)
uta = uta[uta["tag_id"].isin(tags["tag_id"])] # remove from uta dataframe
print(f"\tuser-tag-artist tuples: from {before} to {len(uta)}")

before = len(artists)
artists = artists[artists["artist_id"].isin(uta["artist_id"])] # remove from artists dataframe
print(f"\tartists: from {before} to {len(artists)}")

"""
data cleaning (2)
removing least tagged artists
"""
tags_per_artist = uta[["artist_id", "tag_id"]].drop_duplicates(keep = "first")
tags_per_artist = tags_per_artist.groupby("artist_id")["tag_id"].count()
tags_per_artist = tags_per_artist.reset_index().rename(columns = {"tag_id": "tag_count"})
avg_tags_per_artist = int(tags_per_artist["tag_count"].mean())
artists_to_keep = tags_per_artist[tags_per_artist["tag_count"] >= avg_tags_per_artist]
artists_to_keep = artists_to_keep["artist_id"]

print(f"\ncleaning: removing least tagged artists (below {avg_tags_per_artist} tags)")

before = len(uta)
uta = uta[uta["artist_id"].isin(artists_to_keep)] # remove from uta dataframe
print(f"\tuser-tag-artist tuples: from {before} to {len(uta)}")

beore = len(tags)
tags = tags[tags["tag_id"].isin(uta["tag_id"])] # remove from tags dataframe
print(f"\ttags: from {before} to {len(tags)}")

before = len(user_artists)
user_artists = user_artists[user_artists["artist_id"].isin(artists_to_keep)] # remove from user_artists dataframe
print(f"\tuser-artist pairs: from {before} to {len(user_artists)}")

before = len(artists)
artists = artists[artists["artist_id"].isin(artists_to_keep)] # remove from artists dataframe
print(f"\tartists: from {before} to {len(artists)}")

"""
saving cleaned dataframes
"""
# remove non-utf8 chars
tags["tag"] = tags["tag"].map(lambda x: x.encode('unicode-escape').decode('utf-8'))
artists["name"] = artists["name"].map(lambda x: x.encode('unicode-escape').decode('utf-8'))
tags.to_csv("../data/cleaned/tags.csv")
artists.to_csv("../data/cleaned/artists.csv")
uta.to_csv("../data/cleaned/user_tagged_artists.csv")
user_artists.to_csv("../data/cleaned/user_artists.csv")


"""
intermediate table
artist_tags: artist_id / tag_id
all tags used for each artist (to help filling final dataframes)
"""
artist_tags = uta[["artist_id", "tag_id"]].drop_duplicates(keep = "first")
artist_tags.sort_values(by = ["artist_id", "tag_id"], inplace = True)

at = pd.DataFrame(artist_tags.groupby("artist_id")["tag_id"].count())
at.reset_index(inplace = True)
print(f"media: {at['tag_id'].mean()}")

"""
function that generates the final table for one given user
(table used to train and test the machine learning algorithms)

user_table: artist_id / (932 tags)* / listen_%
    * tag IDs as columns, value is 1 if the artist has it and 0 otherwise
    listen_%: listen count of the user to a certain artist, out of their total listen count
              to measure "how much the user likes the artist"

ps.: function saves dataframe to a .csv file in /data/generated-tables
"""
def build_user_table(user: int) -> None:
    all_tags: list = [str(t) for t in tags["tag_id"]]

    # create table, create tag columns and fill with zeroes
    user_table: pd.DataFrame = user_artists[user_artists["user_id"] == user]
    user_table.drop("user_id", axis = 1, inplace = True)
    user_table = user_table.reindex(columns = user_table.columns.tolist() + all_tags, fill_value = 0)
    user_table.set_index("artist_id", inplace = True)
    # set artist_id as index temporarily, in order to use .at[artist, tag]
    
    artists = artist_tags[artist_tags["artist_id"].isin(user_table.index)]
    # artists: artist_id / tag_id   (for every artist the user listens to)

    # iterate through artist-tag pairs and mark 1 in final table where artist_id has tag_id
    for row in artists.itertuples(index = False):
        user_table.at[row.artist_id, str(row.tag_id)] = 1

    user_table.reset_index(inplace = True)

    # calculate and store listen percentages
    total_listen_count = user_table["listen_count"].sum()
    user_table["listen_%"] = user_table["listen_count"] / total_listen_count
    user_table.drop("listen_count", axis = 1, inplace = True)

    # save .csv file
    user_table.to_csv(f"../data/generated-tables/user_{user}_table.csv")

    return

"""
function that generates the final table for all the artists
(table used to generate new recommendations using an already trained algorithm)

artists_table: artist_id / (932 tags)*
    * tag IDs as columns, value is 1 if the artist has it and 0 otherwise
"""
def build_final_artists_table():
    all_tags: list = [str(t) for t in tags["tag_id"]]

    # create table, create tag columns and fill with zeroes
    artists_table = artists[["artist_id", "name"]]
    artists_table = artists_table.reindex(columns = artists_table.columns.tolist() + all_tags, fill_value = 0)
    artists_table.drop("name", axis = 1, inplace = True)
    artists_table.set_index("artist_id", inplace = True) # set artist_id as index temporarily

    # iterate through artist-tag pairs and mark 1 in final table where artist_id has tag_id
    for row in artist_tags.itertuples(index = False):
        artists_table.at[row.artist_id, str(row.tag_id)] = 1

    artists_table.reset_index(inplace = True)

    # save .csv file
    artists_table.to_csv("../data/generated-tables/final_artists_table.csv")

    print("\n-- final artists table generated --")

    return


print("\n-- data preparation finished --")
