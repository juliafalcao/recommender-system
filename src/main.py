import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # (?)
import codecs

"""
ler dataset e colocar em dataframes
"""

artists = pd.read_table("../data/artists.dat", sep = "\t", header = 0, index_col = 0, names = ["id", "name", "url", "image"])
# index column: "id"

with codecs.open("../data/tags.dat", encoding = "utf-8", errors = "replace") as f: # to avoid encoding error
    tags = pd.read_table(f, sep = "\t", header = 0, index_col = 0, names = ["id", "tag"])
    # index column: "id"
f.close()

user_artists = pd.read_table("../data/user_artists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "count"])
# index column: default

tagged_artists = pd.read_table("../data/user_taggedartists.dat", sep = "\t", header = 0, names = ["user_id", "artist_id", "tag_id", "day", "month", "year"])
# index column: default