# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(DF):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    for c in DF.columns:
        c_min = DF[c].min()
        c_max = DF[c].max()
        for i in range(DF[c].size):
            DF[c][i] = (DF[c][i] - c_min) / (c_max - c_min)
    return DF

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    d = 0.0
    for i in range(v1.size):
        d += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return math.pow(d, 0.5)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(DF):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    d = dict()
    for c in DF.columns:
        d[c] = DF[c].mean()
    return pd.DataFrame(data=[d])

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(DF):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    res = 0.0
    for i in range(DF.shape[0]):
        d = dist_vect(DF.iloc[i],centroide(DF).iloc[0])
        res += d * d
    return res


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,DF):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    indices = [i for i in range(DF.shape[0])]
    indices_exemples = rd.sample(indices, K)
    res = pd.DataFrame(columns=[i for i in df.columns])
    for i in range(K):
        res = pd.concat([res, DF.loc[[indices_exemples[i]]]])
    return res


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(Exe,Centres):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    indice = 0
    dist_min = dist_vect(Exe, Centres.iloc[0])
    for i in range (1,Centres.shape[0]):
        dist_courant = dist_vect(Exe, Centres.iloc[i])
        if(dist_min > dist_courant):
            indice = i
            dist_min = dist_courant
    return indice

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(Base,Centres):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    matrice = dict()
    for i in range(len(centroides)):
        matrice[i] = list()
    for i in range(Base.shape[0]):
        matrice[plus_proche(Base.iloc[i], Centres)].append(i)
    return matrice

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(Base,U):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    res = pd.DataFrame()
    for k, v in U.items():
        tmp = pd.DataFrame(columns=[i for i in Base.columns])
        for i in v:
            tmp = pd.concat([tmp, Base.loc[[i]]])
        res = pd.concat([res, centroide(tmp)], ignore_index=True)
    return res
# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(Base, U):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    res = pd.DataFrame()
    for k, v in U.items():
        tmp = pd.DataFrame(columns=[i for i in Base.columns])
        for i in v:
            tmp = pd.concat([tmp, Base.loc[[i]]])
        res = pd.concat([res, centroide(tmp)], ignore_index=True)
    return res
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, Base, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    raise NotImplementedError("Please Implement this method")
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    raise NotImplementedError("Please Implement this method")
# -------