import numpy as np
import os
import math


def lireMail(fichier, dictionnaire):
    """
    Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
    """
    f = open(fichier, "r", encoding="ascii", errors="surrogateescape")
    mots = f.read().lower().split(" ")  # Conversion en minuscules pour une comparaison insensible à la casse

    x = [False] * len(dictionnaire)

    # x[i] représente la presence du mot d'indice i du dictionnaire dans le message
    for i, mot in enumerate(dictionnaire):
        if mot in mots:
            x[i] = True

    f.close()
    return x


def charge_dico(fichier):
    f = open(fichier, "r")
    mots = f.read().lower().split("\n")  # Conversion en minuscules pour une comparaison insensible à la casse
    mots = [x for x in mots if len(x) >= 3]  # Exclusion des mots de moins de 3 lettres
    print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
    f.close()
    return mots[:-1]


def apprendBinomial(dossier, fichiers, dictionnaire):
    """
    Fonction d'apprentissage d'une loi binomiale à partir des fichiers d'un dossier
    Retourne un vecteur b de paramètres
    """
    freq = np.zeros_like(dictionnaire, dtype=np.int32)

    for fichier in fichiers:
        v = lireMail("/".join([dossier, fichier]), dictionnaire)
        freq += np.array(v, dtype=np.int32)

    # Appliquer le lissage
    e = 1
    b = (freq + e) / (len(fichiers) + e * 2)

    return b


def prediction(x, Pspam, Pham, bspam, bham):
    """
    Prédit si un mail représenté par un vecteur booléen x est un spam
    à partir du modèle de paramètres Pspam, Pham, bspam, bham.
    Retourne True si c'est un spam, False sinon (c'est-à-dire un ham).
    """
    # Conversion du vecteur x en un array numpy pour faciliter les calculs
    x = np.array(x)

    # bjSPAM = nombre de SPAM contenant le mot j / nombre de SPAM
    # bjHAM = nombre de HAM contenant le mot j / nombre de HAM

    # Calcul des probabilités a posteriori
    # P(Y = SPAM | X = x) = (1 / P(X = x)) * P(Y = SPAM) * PRODUIT_PI de j = 1 à d de (bjSPAM)^x^j * (1 - bjSPAM)^(1 - x^j)
    proba_spam = Pspam * np.prod(np.power(bspam, x) * np.power(1 - bspam, 1 - x))
    # P(Y = HAM | X = x) = (1 / P(X = x)) * P(Y = HAM) * PRODUIT_PI de j = 1 à d de (bjHAM)^x^j * (1 - bjHAM)^(1 - x^j)
    proba_ham = Pham * np.prod(np.power(bham, x) * np.power(1 - bham, 1 - x))

    # Retourne True (SPAM) si P(Y = SPAM | X = x) > P(Y = HAM | X = x), sinon retourne False (HAM)
    return proba_spam > proba_ham





def test(dossier, isSpam, Pspam, Pham, bspam, bham):
    fichiers = os.listdir(dossier)
    nb_erreur = 0

    for fichier in fichiers:  # On parcours chaque fichier du dossier
        x = lireMail(dossier + "/" + fichier, dictionnaire)  # On lit le fichier
        isSpamPrediction = prediction(x, Pspam, Pham, bspam, bham)  # On fait une prediction
        type_mail = "SPAM" if isSpam else "HAM"

        if (isSpam and not isSpamPrediction) or (not isSpam and isSpamPrediction):
            # Si c'est un spam et que la prediction est un ham ou si c'est un ham et que la prediction est un spam
            # Il y a donc une erreur
            nb_erreur += 1
            erreur_msg = "*** erreur ***"
        else:
            erreur_msg = ""
        type_prediction = "SPAM" if isSpamPrediction else "HAM"
        print(type_mail + " " + dossier + "/" + fichier + " identifié comme un " + type_prediction + " " + erreur_msg)

    taux_erreur = nb_erreur / len(fichiers)
    return taux_erreur


############ programme principal ############

dossier_spams = "baseapp/spam"
dossier_hams = "baseapp/ham"

dossier_test_spams = "basetest/spam"
dossier_test_hams = "basetest/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = 300  # len(fichiersspams)
mHam = 300  # len(fichiershams)

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

print("La probabilité a priori d'un spam est de " + str(Pspam * 100) + " %")
print("La probabilité a priori d'un ham est de " + str(Pham * 100) + " %")

# Calcul des erreurs avec la fonction test():
print("calcul des erreurs sur les spams...")

erreur_spam = test(dossier_test_spams, True, Pspam, Pham, bspam, bham)

print("calcul des erreurs sur les hams...")
erreur_hams = test(dossier_test_hams, False, Pspam, Pham, bspam, bham)

print("Erreur de test sur " + str(mSpam) + " SPAM    : " + str(round(erreur_spam * 100)) + " %")
print("Erreur de test sur " + str(mHam) + " HAM    : " + str(round(erreur_hams * 100)) + " %")

erreur_globale = (erreur_spam * mSpam + erreur_hams * mHam) / (mSpam + mHam)
print("Erreur de test globale sur " + str((mSpam + mHam)) + " : " + str(round(erreur_globale * 100)) + " %")
