import codecs
import pandas as pd

class tagN:
    def __init__(self, nome, qtd):
         self.nome = nome
         self.qtf = qtd

uta = pd.read_table("../data/user_taggedartists-timestamps.dat", sep="\t", header=0, names=["userID", "artistsID", "tagID","timestamp"])
uta = uta.drop(["timestamp"], axis=1)

with codecs.open("../data/tags.dat", encoding="utf-8", errors="replace") as f:
    # to avoid encoding error b/c of non utf-8 characters
    tags = pd.read_table(f, sep="\t", header=0, names=["tagID", "value"])
f.close()

artists = pd.read_table("../data/artists.dat", sep="\t", header=0, names=["artistsID", "name", "url", "image"])
artists = artists.drop(["image","url"], axis=1)

utajoin = pd.merge(uta,tags,on=["tagID","tagID"])

utajoin2 = pd.merge(utajoin,artists,on=["artistsID","artistsID"])

utajoin2 = utajoin2.drop(["userID","tagID","artistsID"],axis=1)

def achatags(artista):
    listatags = []
    listatagsfinal = []
    qtd = []
    for index, row in utajoin2.iterrows():
        if row["name"] == artista:
            if row["value"] not in listatags:
                listatags.append(row["value"])
                qtd.append(1)
            else:
                indice = listatags.index(row["value"])
                qtd[indice] = qtd[indice]+1

    for i in range(10):
        if qtd:
            maior = max(qtd)
            ind_maior = qtd.index(maior)
            maior_tag = listatags[ind_maior]
            listatagsfinal.append(maior_tag)
            qtd.remove(maior)
            listatags.remove(maior_tag)


    return(listatagsfinal)

def produzDataFrame():
    for index, row in artists.iterrows():
        print(row["name"])
        print(achatags(row["name"]))

produzDataFrame()
