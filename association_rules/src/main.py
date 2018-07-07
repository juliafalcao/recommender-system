
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def produzcsv_associacao(): #depracated
    ua = pd.read_table("../data/user_artists.dat", sep="\t", header=0,
                        names=["userID", "artistsID", "weight"])

    artists = pd.read_table("../data/artists.dat", sep="\t", header=0,
                            names=["artistsID", "name", "url", "image"])

    artists = artists.drop(["image", "url"], axis=1)
    ua_artists = pd.merge(ua, artists, on=["artistsID", "artistsID"])

    ua_artists.groupby('userID')['name'].apply(','.join).reset_index().to_csv("../output/associacao.csv",sep=",")

def tiraaspas(): #depracated
    with open("../data/associacao.csv", 'r',encoding="utf8") as fh:
        with open("../data/associao2.csv", 'w',encoding="utf8") as dest:
            for linha in fh:
                dest.write(linha[:-2]+"\n")


def produzcsv_associacao2(): #funcao de pre processamento utilizada
    ua = pd.read_csv("../data/user_artists.csv", sep=",", header=0,
                        names=["userID", "artistsID", "weight"])

    artists = pd.read_table("../data/artists.csv", sep=",", header=0,
                            names=["artistsID", "name", "url"])

    artists = artists.drop(["url"], axis=1)
    ua_artists = pd.merge(ua, artists, on=["artistsID", "artistsID"])

    ua_artists = ua_artists.groupby('userID')['name'].apply(','.join).reset_index()
    lista_artistas = artists["name"].tolist()
    with open("../data/user_artists_boolean.csv", 'w', encoding="utf8") as dest:
        for index, row in ua_artists.iterrows():
            resp = "{}".format(row['userID'])
            for item in lista_artistas:
                if row['name'].find(item) != -1:
                    resp += ",1"
                else:
                    resp += ",0"
            resp += "\n"
            dest.write(resp)



a = pd.read_csv("../data/user_artists_boolean.csv", sep=",", header=0,)
a = a.drop(["user"],axis=1)
frequent_itemsets = apriori(a, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values(['confidence','lift'],ascending=[0,0]).to_csv("../output/regras_supp_01.csv")

