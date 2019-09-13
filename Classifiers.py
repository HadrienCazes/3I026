# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import math

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        score = 0.0
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) * dataset.getY(i) >= 0:
                score += 1.0
        return score/dataset.size()

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = np.random.rand(input_dimension) 
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(self.w, x)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        tab_dist = list()
        for i in range(self.dataset.size()):
            x2 = self.dataset.getX(i)
            dist = math.sqrt( (x[0]-x2[0])**2 + (x[1]-x2[1])**2 )
            tab_dist.append(dist)
            
        tab_index = np.argsort(tab_dist)
        
        somme_type = 0
        for i in range(self.k):
            somme_type += self.dataset.getY(tab_index[i])[0]
          
        return somme_type

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.dataset = labeledSet

# ---------------------------
