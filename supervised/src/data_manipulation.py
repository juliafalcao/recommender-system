# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import typing

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
experimental data analysis
"""
tag_uses = uta.groupby("tag_id")["user_id", "artist_id"].count()
tag_uses.drop("user_id", axis = 1, inplace = True)
tag_uses.rename(columns = {"artist_id": "count"}, inplace = True)
avg_tag_uses = int(tag_uses["count"].mean())
# least_used_tags = tag_uses[tag_uses["count"] < avg_tag_uses]
# least_used_tags = least_used_tags.reset_index()
# least_used_tags = least_used_tags["tag_id"]
tags_to_keep = tag_uses[tag_uses["count"] >= avg_tag_uses]
tags_to_keep = tags_to_keep.reset_index()
tags_to_keep = tags_to_keep["tag_id"]

"""
data cleaning
"""

# removing least used tags
print("data cleaning")
print(f"before: {len(tags)} tags, {len(uta)} uta")
tags = tags[tags["tag_id"].isin(tags_to_keep)]
uta = uta[uta["tag_id"].isin(tags_to_keep)]
print(f"after: {len(tags)} tags, {len(uta)} uta")
print()


"""
intermediate tables
"""

"""
    tag_popularities: tag_id / tag / uses / popularity
    uses of tag_id out of all taggings done
    popularity = uses / total_taggings where a tagging is one (user, artist, tag) tuple
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
    artist_tags: artist_id / tag_id / tag / popularity
    all tags used for each artist, from most to least popular
"""
artist_tags = uta[["artist_id", "tag_id"]].drop_duplicates(keep = "first")
artist_tags = artist_tags.merge(tag_popularities, on = "tag_id")
artist_tags.sort_values(by = ["artist_id", "popularity"], ascending = [True, False], inplace = True)
artist_tags.drop("uses", axis = 1, inplace = True)

"""
    final table: user_id / artist_id / (all tags)
    each tag's value is 0 on 1, whether the artist has the tag or not
"""

"""
def get_artist_tags(artist: int) -> list:
    return list(artist_tags[artist_tags["artist_id"] == artist]["tag_id"])
"""

final = user_artists[["user_id", "artist_id"]]
all_tags = [str(t) for t in tags["tag_id"]]
final = final.reindex(columns = final.columns.tolist() + all_tags, fill_value = 0)

# final.to_csv("final2.csv")
