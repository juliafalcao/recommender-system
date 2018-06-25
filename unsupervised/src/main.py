import codecs
from random import randint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
col = ['Artista', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'Tag6', 'Tag7', 'Tag8', 'Tag9', 'Tag10']



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


def produzdf2():
    print(uta)
    tag_popularities = uta.groupby("tagID")[["userID", "artistsID"]].count()
    tag_popularities = pd.DataFrame(tag_popularities).reset_index().drop(columns="artistsID", axis=1)
    tag_popularities.rename(columns={"userID": "uses"}, inplace=True)
    tag_popularities["popularity"] = tag_popularities["uses"]
    tag_popularities.sort_values(by="popularity", ascending=False, inplace=True)
    tag_popularities = tag_popularities.merge(tags, on="tagID")
    tag_popularities = tag_popularities[["tagID", "value", "uses", "popularity"]]
    tag_popularities.to_csv("../unsupervised/data/toptags.csv", sep=',')


def produzdf3(ntags):
    top_tags = pd.read_csv("../data/toptags.csv", sep=',')
    top_tags = top_tags.drop(["Unnamed: 0"],axis=1)
    top_tags = top_tags.head(n=ntags)
    return top_tags


col4 = ['Artista','rock', 'pop', 'alternative', 'electronic', 'indie', 'female vocalists', '80s', 'dance', 'alternative rock', 'classic rock', 'british', 'indie rock', 'singer-songwriter', 'hard rock', 'experimental', 'metal', 'ambient', '90s', 'new wave', 'seen live', 'chillout', 'hip-hop', 'punk', 'folk', 'electronica', 'rnb', 'instrumental', 'heavy metal', 'soul', 'acoustic', 'progressive rock', '70s', 'jazz', 'soundtrack', 'male vocalists', 'industrial', 'trip-hop', 'metalcore', 'rap', 'synthpop', 'american', 'hardcore', 'indie pop', 'pop rock', '00s', 'britpop', 'post-punk', '60s', 'punk rock', 'blues', 'psychedelic', 'downtempo', 'beautiful', 'sexy', 'thrash metal', 'idm', 'post-rock', 'electro', 'awesome', 'love', 'mellow', 'cover', 'death metal', 'female vocalist', 'post-hardcore', 'brazilian', 'amazing', 'pop punk', 'country', 'ebm', 'progressive metal', 'emo', 'hip hop', 'piano', 'screamo', 'trance', 'funk', 'classical', 'nu metal', 'favorites', 'melodic death metal', 'gothic', 'grunge', 'house', 'german', 'female', 'canadian', 'power metal', 'techno', 'ballad', 'uk', '<3', 'love at first listen', 'cool', 'french', 'catchy', 'deathcore', 'japanese', 'synth pop', 'sad', 'latin', 'usa', 'shoegaze', 'reggae', 'brasil', 'black metal', 'epic', 'gothic metal', 'electropop', 'world', 'favorite', 'blues rock', 'mpb', 'minimal', 'oldies', 'progressive', 'lounge', 'covers', 'darkwave', 'disco', 'classic', 'symphonic metal', 'new age', 'avant-garde', 'guitar', 'dream pop', 'atmospheric', 'lo-fi', 'noise', 'psychedelic rock', 'fun', 'happy', 'j-pop', 'alternative metal', 'swedish', 'industrial metal', 'christian', 'soft rock', 'chill', 'remix', 'romantic', 'favorite songs', 'english', 'ska', 'rock n roll', 'spanish', 'rock and roll', 'urban', 'j-rock', 'grindcore', 'russian', 'legend', 'sweet', 'dreamy', 'dark electro', 'garage rock', 'polish', 'futurepop', 'christian rock', 'gothic rock', 'doom metal', 'hair metal', 'melancholic', 'disney', 'hot', 'powerpop', 'glam rock', 'relax', 'party', 'relaxing', 'dark ambient', 'perfect', 'brazil', 'new romantic', 'italian', 'icelandic', 'emocore', '2008', 'mathcore', 'love songs', 'love it', 'fucking awesome', 'ethereal', 'guilty pleasures', '50s', 'southern rock', 'electroclash', 'speed metal', 'australian', 'dub', 'easy listening', 'bossa nova', 'favourites', 'guilty pleasure', 'cute', 'christmas', 'diva', 'brutal death metal', 'psytrance', 'male vocalist']
def produzdf4():
    resp = pd.DataFrame(columns=col4)
    i = 1
    top_tags = produzdf3(200)
    print(top_tags["value"].tolist())
    for index, row in artists.iterrows():
        tagsbool = [row["name"]]
        utaagrupado2 = utaAgrupado[utaAgrupado["name"] == row["name"]]
        lista_tags = utaagrupado2["value"].tolist()
        for indext, rowt in top_tags.iterrows():
            if rowt["value"] in lista_tags:
                tagsbool.append(1)
            else:
                tagsbool.append(0)
        if sum(tagsbool[1:]) > 0:
            resp.loc[len(resp)] = tagsbool
        print(i)
        i=i+1

    print(resp.to_string)
    resp.to_csv("../data/tagsbool200semzeros.csv", sep=',')


def kmeans_artista_tags(usuario_tags,ntags,ncluster):

    artistas = pd.read_csv("../data/tagsbool"+str(ntags)+"semzeros.csv", sep=',')
    artistas = artistas.drop(["Unnamed: 0"], axis=1)
    artistas = artistas.append(usuario_tags,ignore_index=True)
    artistas = artistas.drop(["Artista"], axis=1)
    art_array = artistas.values
    kmeans = KMeans(n_clusters=100, random_state=0).fit(art_array)
    #distance = kmeans.fit_transform(art_array)
    #np.set_printoptions(threshold=np.nan)
    artistas = pd.read_csv("../data/tagsbool" + str(ntags) + "semzeros.csv", sep=',')
    artistas = artistas.drop(["Unnamed: 0"], axis=1)
    artistas = artistas.append(usuario_tags,ignore_index=True)
    artistas["cluster"] = kmeans.labels_
    artistas = artistas.drop(artistas.columns[1:51], axis=1)
    user_id = usuario_tags["Artista"].tolist()[0]
    artistas.sort_values(by=['cluster']).to_csv("../output/recomendacao-id"+str(user_id)+"-c"+str(ncluster)+"-t"+str(ntags)+".csv",sep=",")
    user = artistas[artistas["Artista"]==user_id]
    user_c = user["cluster"].tolist()[0]
    cluster_user = artistas[artistas["cluster"]==user_c]
    cluster_user = cluster_user.reset_index(drop=True)
    recomendacoes = []
    i = 0
    while i <= 4:
        r = randint(0, len(cluster_user.index)-2)
        if r not in recomendacoes:
            recomendacoes.append(r)
            i += 1

    print(cluster_user.iloc[recomendacoes])

def usuario_top_tags():
    usuarios = pd.read_table("../data/user_artists.dat", sep="\t", header=0, names=["userID", "artistsID", "weight"])
    ua = pd.merge(usuarios,artists,on=['artistsID',"artistsID"])
    tags = pd.read_csv("../data/tagsbool50.csv", sep=',')
    tags = tags.drop(["Unnamed: 0"], axis=1)
    uat = pd.merge(ua, tags, how='inner', left_on='name', right_on='Artista')
    uat = uat.sort_values(by=["userID", "weight"], ascending=[True, False])
    uat = uat.drop(["name"],axis=1)
    uat.to_csv("../data/usuario_artista_tag.csv", sep=',')

def recomenda(id,ntags,ncluster):
    usuario = pd.read_csv("../data/usuario_artista_tag.csv", sep=',')
    usuario = usuario.drop(["Unnamed: 0"], axis=1)
    usuario = usuario[usuario["userID"] == id]
    usuario = usuario.drop(["artistsID","weight","Artista"], axis=1)
    usuario = usuario.reset_index(drop=True)
    usuario_tags = usuario.loc[[0]]
    new_columns = usuario_tags.columns.tolist()
    new_columns[0] = 'Artista'
    usuario_tags.columns = new_columns
    kmeans_artista_tags(usuario_tags,ntags,ncluster)
    #artistas = kmeans_artista_tags(ntags)

#print(utaAgrupado)
#produzdf()
#produzdf3()
#produzdf4()

#kmeans_artista_tags(50)
recomenda(2,50,100)