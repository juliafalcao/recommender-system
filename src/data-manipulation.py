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

tags_per_artist = uta.groupby("artist_id")["tag_id"].count()
tags_per_artist = pd.DataFrame(tags_per_artist).reset_index().rename(columns = {"tag_id": "tag_count"})
print(f"Average tags per artist: {int(tags_per_artist['tag_count'].mean())}")
less_than_4 = len(tags_per_artist[tags_per_artist['tag_count'] < 4])
print(f"Artists with less than 4 tags: {less_than_4} ({less_than_4 / len(artists)}%)")
print()

"""
fig = plt.figure()
ax = fig.add_axes([0.11, 0.11, 0.8, 0.8])
ax.set_axisbelow(True)
ax.yaxis.grid(color = "lightgray")
ax.set_xlabel("Tags per artist")
ax.set_ylabel("Artists")
ax.hist(tags_per_artist["tag_count"], range = [1, 5], bins = 5, rwidth = 0.8, color = "c")
plt.savefig("../data-analysis/tags-per-artist-5.png")
"""

"""
data cleaning
"""
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
"""


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
    tag_weights: user_id / tag_id / listen_count / weight
    how much user_id has listened to artists tagged as tag_id
    weight = total listen count of artists tagged as tag_id, out of user_id's total listen count
"""
tag_weights = uta.merge(user_artists, on = ["user_id", "artist_id"])
tag_weights.sort_values(by = "tag_id", ascending = False, inplace = True)
tag_weights = tag_weights.groupby(["user_id", "tag_id"])["listen_count"].sum()
tag_weights = pd.DataFrame(tag_weights).reset_index()
tag_weights.rename(columns = {"listen_count": "tag_listen_count"}, inplace = True)
user_listen_counts = user_artists.groupby("user_id")["listen_count"].sum()
user_listen_counts = pd.DataFrame(user_listen_counts).reset_index().rename(columns = {"listen_count": "user_listen_count"})
tag_weights = tag_weights.merge(user_listen_counts, on = "user_id")
tag_weights["weight"] = tag_weights["tag_listen_count"] / tag_weights["user_listen_count"]
tag_weights.drop(["user_listen_count", "tag_listen_count"], axis = 1, inplace = True)
tag_weights = tag_weights.merge(tags, on = "tag_id")
tag_weights.sort_values(by = ["user_id", "weight"], ascending = [True, False], inplace = True)
tag_weights = tag_weights[["user_id", "tag_id", "tag", "weight"]]
# print(tag_weights.head().to_string())


"""
    analysis to defend the dataset
    dedicated to raphael leardini & raffael paranhos
"""
artist_tag_names = user_tagged_artists.merge(artists, on = "artist_id")
artist_tag_names = artist_tag_names.merge(tags, on = "tag_id")
artist_tag_names = artist_tag_names.drop(columns = ["user_id", "tag_id", "artist_id", "url"])
artist_tag_names.rename(columns = {"name": "artist_name"}, inplace = True)
artist_tag_names.drop_duplicates(keep = "first", inplace = True)
artist_tag_names = artist_tag_names[["tag", "artist_name"]]
# print(artist_tag_names[artist_tag_names["tag"] == "the beatles"])

"""
    top_tags: user_id / tag_id / tag / popularity / weight / score
    top tags per (user, artist) pair
    score = popularity * weight
"""
top_tags = tag_weights[["user_id", "tag_id", "tag", "weight"]].merge(tag_popularities[["tag_id", "popularity"]], on = "tag_id")
top_tags = top_tags[["user_id", "tag_id", "tag", "popularity", "weight"]]
top_tags.sort_values(by = ["user_id", "weight"], ascending = [True, False], inplace = True)
top_tags["score"] = top_tags["weight"] * top_tags["popularity"]
# top_tags["score_avg"] = (top_tags["weight"] + top_tags["popularity"]) / 2
top_tags.sort_values(by = ["user_id", "score"], ascending = [True, False], inplace = True)
# print(top_tags.head(30).to_string())

"""
    artist_tags: artist_id / tag_id / tag / popularity
    all tags used for each artist, from most to least popular
"""
artist_tags = uta[["artist_id", "tag_id"]].drop_duplicates(keep = "first")
artist_tags = artist_tags.merge(tag_popularities, on = "tag_id")
artist_tags.sort_values(by = ["artist_id", "popularity"], ascending = [True, False], inplace = True)
artist_tags.drop("uses", axis = 1, inplace = True)
# print(artist_tags)

"""
    final table: user_id / tag1 / tag2 / tag3 / tag4 / tag5 / artist_id
    top 5 tags: chosen by most popular and most relevant to the user
                if less than 5: filled with popular tags, regardless of importance to user
"""

# ?
available_tags = artist_tags[["artist_id", "tag_id", "popularity"]]
available_tags = available_tags.merge(top_tags[["user_id", "tag_id", "score"]], on = "tag_id")

"""
Function that returns a tuple of the top 5 tags for an (user, artist) pair.
Criteria for choosing the tags:
1. tags the artist has and the user likes, ranked by score
2. tags the artist has, ranked by popularity
3. if still less than 5, fill with 0
"""
def get_top_tags(user: int, artist: int) -> tuple:
    top_5_tags = top_tags[top_tags["user_id"] == user]
    top_5_tags = top_5_tags[top_5_tags["tag_id"].isin(artist_tags[artist_tags["artist_id"] == artist]["tag_id"])]

    if len(top_5_tags) >= 5:
        return tuple(top_5_tags["tag_id"][:5])
    
    else:
        # user doesn't know 5 tags the artist has:
        # fill with most popular tags for the artist
        top_5_tags = list(top_5_tags["tag_id"])
        all_artist_tags = artist_tags[artist_tags["artist_id"] == artist][["artist_id", "tag_id", "tag", "popularity"]] # remover "tag"
        all_artist_tags.sort_values(by = "popularity", ascending = False, inplace = True)
        all_artist_tags = list(all_artist_tags["tag_id"])
        remaining_tags = [tag for tag in all_artist_tags if tag not in top_5_tags]
        top_5_tags.extend(remaining_tags)

        if len(top_5_tags) >= 5:
            return top_5_tags[:5]
        
        else: # if the artist's other tags weren't enough:
            missing = 5 - len(top_5_tags)
            top_5_tags.extend([0] * missing) # fill with zeroes
            return top_5_tags[:5]

# print(get_top_tags(2095, 18680)) # artist has 4 tags, user likes 3
# print(top_tags[top_tags["user_id"] == 2095])
# print(artist_tags[artist_tags["artist_id"] == 18680])

final = user_artists[["user_id", "artist_id"]]
final = final.merge(artists, on = "artist_id").drop("url", axis = 1).rename(columns = {"name": "artist_name"})
final.sort_values(by = "user_id", inplace = True)
final["top_5_tags"] = final.apply(lambda row: get_top_tags(row.user_id, row.artist_id), axis = 1)
final[["tag1", "tag2", "tag3", "tag4", "tag5"]] = final["top_5_tags"].apply(pd.Series)
final.drop("top_5_tags", axis = 1, inplace = True)
final = final[["user_id", "tag1", "tag2", "tag3", "tag4", "tag5", "artist_id"]]
final.to_csv("final.csv")
