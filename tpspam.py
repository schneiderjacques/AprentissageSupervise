import numpy as np
import os
import math

def lireMail(fichier, dictionnaire):
	""" 
	Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
	"""
	f = open(fichier, "r",encoding="ascii", errors="surrogateescape")
	mots = f.read().split(" ")
	
	x = [False] * len(dictionnaire) 

	# x[i] représente la presence du mot d'indice i du dictionnaire dans le message
	for i, mot in enumerate(dictionnaire):
		if mot in mots:
			x[i] = True
	
	f.close()
	return x

def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	mots = [x for x in mots if len(x) >= 3] #4. Consigne, on ne garde que les mots de 3 lettres et plus
	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
	f.close()
	return mots[:-1]

def apprendBinomial(dossier, fichiers, dictionnaire):
	"""
	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
	Retourne un vecteur b de paramètres 
		
	"""
	freq = np.zeros_like(dictionnaire, dtype=np.int32) # compte / accu la présence des mots dans l'ensemble des mails spams (ou hams)
	
	for fichier in fichiers: 
		v = lireMail("/".join([dossier, fichier]), dictionnaire) # vecteur présence mots
		freq += np.array(v, dtype=np.int32) # cast booleen en int, si la valeur d'un indice est True, +1 dans freq

	b = freq / len(fichiers)
	return b


def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
		
	"""
	
	return False  # à modifier...
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""
	fichiers = os.listdir(dossier)
	for fichier in fichiers:
		print("Mail " + dossier+"/"+fichier)		

		
		# à compléter...

	return 0  # à modifier...


############ programme principal ############

dossier_spams = "baseapp/spam" # à vérifier
dossier_hams = "baseapp/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = len(fichiersspams)
mHam = len(fichiershams)

# Chargement du dictionnaire:
dictionnaire = charge_dico("dictionnaire1000en.txt")
print(dictionnaire)

# Apprentissage des bspam et bham:
print("apprentissage de bspam...")
bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
print("apprentissage de bham...")
bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam / (mSpam + mHam)
Pham = 1 - Pspam

# Calcul des erreurs avec la fonction test():


