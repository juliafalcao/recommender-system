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
print(user_tagged_artists.head().to_string())
"""


"""
data cleaning (1)
removing least used tags
"""
tag_uses = uta.groupby("tag_id")["user_id", "artist_id"].count()
tag_uses.drop("user_id", axis = 1, inplace = True)
tag_uses.rename(columns = {"artist_id": "count"}, inplace = True)
avg_tag_uses = int(tag_uses["count"].mean())
tags_to_keep = tag_uses[tag_uses["count"] >= avg_tag_uses]
tags_to_keep = tags_to_keep.reset_index()
tags_to_keep = tags_to_keep["tag_id"]

print(f"data cleaning: removing least used tags (below {avg_tag_uses} uses)")
before = len(tags)
tags = tags[tags["tag_id"].isin(tags_to_keep)]
tags.to_csv("../data/cleaned_tags.csv")

before = len(uta)
uta = uta[uta["tag_id"].isin(tags_to_keep)]
print(f"from {before} to {len(uta)} user-tag-artist tuples", end = "\n\n")
uta.to_csv("../data/cleaned_user_tagged_artists.csv")


"""
intermediate tables
"""

"""
    tag_popularities: tag_id / tag / uses / popularity
    uses of tag_id out of all taggings done
    popularity = uses / total_taggings where a tagging is one (user, artist, tag) tuple
"""
"""
total_taggings = len(uta)
tag_popularities = uta.groupby("tag_id")[["user_id", "artist_id"]].count()
tag_popularities = pd.DataFrame(tag_popularities).reset_index().drop(columns = "artist_id", axis = 1)
tag_popularities.rename(columns = {"user_id": "uses"}, inplace = True)
tag_popularities["popularity"] = tag_popularities["uses"] / total_taggings
tag_popularities.sort_values(by = "popularity", ascending = False, inplace = True)
tag_popularities = tag_popularities.merge(tags, on = "tag_id")
tag_popularities = tag_popularities[["tag_id", "tag", "uses", "popularity"]]
# print(tag_popularities.head())
"""

"""
    artist_tags: artist_id / tag_id
    all tags used for each artist
"""
artist_tags = uta[["artist_id", "tag_id"]].drop_duplicates(keep = "first")
# artist_tags = artist_tags.merge(tag_popularities, on = "tag_id")
# artist_tags.sort_values(by = ["artist_id", "popularity"], ascending = [True, False], inplace = True)
# artist_tags.drop("uses", axis = 1, inplace = True)
artist_tags.sort_values(by = ["artist_id", "tag_id"], inplace = True)

"""
data cleaning (2)
removing least tagged artists
"""
tags_per_artist = artist_tags.groupby("artist_id")["tag_id"].count()
tags_per_artist = tags_per_artist.reset_index().rename(columns = {"tag_id": "tag_count"})
avg_tags_per_artist = int(tags_per_artist["tag_count"].mean())

artists_to_keep = tags_per_artist[tags_per_artist["tag_count"] >= avg_tags_per_artist]
artists_to_keep = artists_to_keep["artist_id"]

print(f"data cleaning: removing least tagged artists (below {avg_tags_per_artist} tags per artist)")
before = len(artists)
artists = artists[artists["artist_id"].isin(artists_to_keep)]
print(f"from {before} to {len(artists)} artists")
artists.to_csv("../data/cleaned_artists.csv")

before = len(uta)
uta = uta[uta["artist_id"].isin(artists_to_keep)]
print(f"from {before} to {len(uta)} user-tag-artist tuples")
uta.to_csv("../data/cleaned_user_tagged_artists.csv")

before = len(user_artists)
user_artists = user_artists[user_artists["artist_id"].isin(artists_to_keep)]
print(f"from {before} to {len(user_artists)} user-artist pairs")
user_artists.to_csv("../data/cleaned_user_artists.csv")


"""
final table for one user: artist_id / (932 tags)* / listen_%
tags as columns where the value is 1 if the artist has it and 0 otherwise
"""
def build_user_table(user: int) -> pd.DataFrame:
    user_table = user_artists[user_artists["user_id"] == user]
    user_table.drop("user_id", axis = 1, inplace = True)

    all_tags: list = [str(t) for t in tags["tag_id"]]
    user_table = user_table.reindex(columns = user_table.columns.tolist() + all_tags, fill_value = 0)
    user_table.set_index("artist_id", inplace = True) # set artist_id as index temporarily, in order to use .at[artist, "tag"]
    
    # artists: artist_id / tag_id   (for every artist the user listens to)
    artists = artist_tags[artist_tags["artist_id"].isin(user_table.index)]
    
    """
    iterate through artist tag pairs and mark 1 in final table where artist_id has tag_id
    """
    for row in artists.itertuples(index = False):
        user_table.at[row.artist_id, str(row.tag_id)] = 1

    user_table.reset_index(inplace = True)

    # listen %
    total_listen_count = user_table["listen_count"].sum()
    user_table["listen_%"] = user_table["listen_count"] / total_listen_count
    user_table.drop("listen_count", axis = 1, inplace = True)

    user_table.to_csv(f"/user-tables/user_{user}_table.csv")

"""
final table for all artists: artist_id / (932 tags)
"""

artists_table = artists[["artist_id", "name"]]
all_tags: list = [str(t) for t in tags["tag_id"]]
artists_table = artists_table.reindex(columns = artists_table.columns.tolist() + all_tags, fill_value = 0)
artists_table.set_index("artist_id", inplace = True)
artists_table.drop("name", axis = 1, inplace = True)

for row in artist_tags.itertuples(index = False):
    artists_table.at[row.artist_id, str(row.tag_id)] = 1

artists_table = artists_table.reset_index()
artists_table.to_csv("../data/final_artists_table.csv")

# TODO: change dtype of tag columns to int
