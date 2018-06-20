import codecs
import pandas as pd
from sklearn.cluster import KMeans

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

    return listatagsfinal


utaAgrupado = utajoin2.groupby(["name", "value"]).size().reset_index(name='count')
col = ['Artista','Tag1','Tag2','Tag3','Tag4','Tag5','Tag6','Tag7','Tag8','Tag9','Tag10']


def produzdf():
    resp = pd.DataFrame(columns=col)
    i = 1
    for index, row in artists.iterrows():
        print(i)
        utaagrupado2 = utaAgrupado[utaAgrupado["name"] == row["name"]].sort_values(by=['count'], ascending=[False])
        lista_tags = utaagrupado2["value"].tolist()
        lista_tags = lista_tags[:10]
        lista_tags.insert(0, row["name"])
        #lista_tags_df = pd.DataFrame([lista_tags])
        if len(lista_tags) == 11:
            resp.loc[len(resp)] = lista_tags
        #resp = resp.append(lista_tags_df, ignore_index=True)  # ignoring index is optional
        i = i + 1
    resp.to_csv("../unsupervised/data/lista_tags_virgula.csv", sep=',')


def kmeans_lista_tags():
    lista_tags = pd.read_csv("../unsupervised/data/lista_tags_virgula.csv", sep=',')
    lista_tags = lista_tags.drop(["ID"], axis=1)
    lt_array = lista_tags.values
    print(lt_array)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(lt_array)
    print(kmeans)


#produzdf()
kmeans_lista_tags()