import numpy as np
import os
import math
from numpy import log1p
import string
import pickle


def lireMail(fichier, dictionnaire):
    """
    Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
    """
    # Est insensible à la casse et aux accents
    # Les mots de plus de 3 lettres sont pris en compte
    # On enlève les mots qui finissent par des ponctuations on passe de 24% d'erreur à 22% d'erreur pour les spam
    # En enlevant les \n et \r on passe de 22% à 18 pour les SPAM et de 2% à 1% pour les HAM

    f = open(fichier, "r", encoding="ascii", errors="surrogateescape")
    mots = f.read().lower()  # Conversion en minuscules pour une comparaison insensible à la casse
    mots = mots.replace('\n', ' ').replace('\r', ' ')
    mots = mots.split(" ")

    translator = str.maketrans("", "", string.punctuation)
    mots = [mot.translate(translator) for mot in mots]
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

    # Après des recherches on peut réduire l'impact de l'imprécision numérique lors du calcul des probabilités a posteriori.
    # En utilisant la fonction log1p qui calcule log(1+x) de manière plus précise pour les petits x.

    # Conversion du vecteur x en un array numpy pour faciliter les calculs
    x = np.array(x)

    # Calcul des logarithmes des probabilités a priori
    log_Pspam = np.log(Pspam)
    log_Pham = np.log(Pham)

    # Calcul des logarithmes des rapports de vraisemblance pour chaque mot présent et absent
    log_ratios_present = np.log((bspam) / (bham))
    log_ratios_absent = np.log(((1 - bspam)) / ((1 - bham)))

    # Calcul du logarithme du rapport de probabilités a posteriori
    log_ratio_posteriori = log_Pspam - log_Pham + np.sum(log_ratios_present * x) + np.sum(log_ratios_absent * (1 - x))

    # Calcul des probabilités a posteriori P(Y=SPAM | X=x) et P(Y=HAM | X=x)
    log1p_exp_ratio_diff = log1p(np.exp(-log_ratio_posteriori))
    Pspam_x = 1 / (1 + np.exp(log1p_exp_ratio_diff))
    Pham_x = 1 - Pspam_x

    isSpam = log_ratio_posteriori > 0

    return isSpam, Pspam_x, Pham_x


def test(dossier, isSpam, Pspam, Pham, bspam, bham):
    fichiers = os.listdir(dossier)
    nb_erreur = 0
    indexSpam = 0
    indexHam = 0
    for fichier in fichiers:  # On parcours chaque fichier du dossier
        x = lireMail(dossier + "/" + fichier, dictionnaire)  # On lit le fichier
        isSpamPrediction, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)
        type_mail = "SPAM" if isSpam else "HAM"

        if (isSpam and not isSpamPrediction) or (not isSpam and isSpamPrediction):
            # Si c'est un spam et que la prediction est un ham ou si c'est un ham et que la prediction est un spam
            # Il y a donc une erreur
            nb_erreur += 1
            erreur_msg = "*** erreur ***"
        else:
            erreur_msg = ""
        type_prediction = "SPAM" if isSpamPrediction else "HAM"
        print(type_mail + " numéro ", end="")
        if isSpam:
            print(indexSpam, end="")
        else:
            print(indexHam, end="")
        print(" : P(Y=SPAM | X=x) = " + str(Pspam_x) + ", P(Y=HAM | X=x) = " + str(Pham_x))
        print("    => identifié comme un " + type_prediction + " " + erreur_msg)
        if isSpam:
            indexSpam += 1
        else:
            indexHam += 1

    taux_erreur = nb_erreur / len(fichiers)
    return taux_erreur


def testClassifieur(dossier, classifieur, isSpam):
    fichiers = os.listdir(dossier)
    nb_erreur = 0
    indexSpam = 0
    indexHam = 0
    for fichier in fichiers:  # On parcours chaque fichier du dossier
        x = lireMail(dossier + "/" + fichier, classifieur['dictionnaire'])  # On lit le fichier
        isSpamPrediction, Pspam_x, Pham_x = prediction(x, classifieur['Pspam'], classifieur['Pham'],
                                                       classifieur['bspam'], classifieur['bham'])
        type_mail = "SPAM" if isSpam else "HAM"

        if (isSpam and not isSpamPrediction) or (not isSpam and isSpamPrediction):
            # Si c'est un spam et que la prediction est un ham ou si c'est un ham et que la prediction est un spam
            # Il y a donc une erreur
            nb_erreur += 1
            erreur_msg = "*** erreur ***"
        else:
            erreur_msg = ""
        type_prediction = "SPAM" if isSpamPrediction else "HAM"
        print(type_mail + " numéro ", end="")
        if isSpam:
            print(indexSpam, end="")
        else:
            print(indexHam, end="")
        print(" : P(Y=SPAM | X=x) = " + str(Pspam_x) + ", P(Y=HAM | X=x) = " + str(Pham_x))
        print("    => identifié comme un " + type_prediction + " " + erreur_msg)
        if isSpam:
            indexSpam += 1
        else:
            indexHam += 1

    taux_erreur = nb_erreur / len(fichiers)
    return taux_erreur


def enregistreClassifieur(classifieur, nom):  # enregistre le classifieur dans un fichier avec son nom
    with open(nom + ".pkl", "wb") as f:
        pickle.dump(classifieur, f)


def chargeClassifieur(nom):  # charge le classifieur depuis un fichier avec son nom
    try:
        with open(nom + ".pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Le fichier " + nom + ".pkl n'a pas été trouvé.")
        return None


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
"""print("calcul des erreurs sur les spams...")

erreur_spam = test(dossier_test_spams, True, Pspam, Pham, bspam, bham)

print("calcul des erreurs sur les hams...")
erreur_hams = test(dossier_test_hams, False, Pspam, Pham, bspam, bham)

print("Erreur de test sur " + str(mSpam) + " SPAM    : " + str(round(erreur_spam * 100)) + " %")
print("Erreur de test sur " + str(mHam) + " HAM    : " + str(round(erreur_hams * 100)) + " %")

erreur_globale = (erreur_spam * mSpam + erreur_hams * mHam) / (mSpam + mHam)
print("Erreur de test globale sur " + str((mSpam + mHam)) + " : " + str(round(erreur_globale * 100)) + " %")"""

# Améliorations

classifieur = {}
classifieur["dictionnaire"] = dictionnaire
classifieur["Pspam"] = Pspam
classifieur["Pham"] = Pham
classifieur["bspam"] = bspam
classifieur["bham"] = bham
classifieur["mSpam"] = mSpam
classifieur["mHam"] = mHam

# Enregistrement du classifieur dans un fichier classifieur.pkl
enregistreClassifieur(classifieur, "classifieur")

# Chargement du classifieur à partir du fichier classifieur.pkl
classifieur = chargeClassifieur("classifieur")

# Calcul des erreurs avec la fonction testClassifieur():
print("[Amelioration] calcul des erreurs sur les spams...")
erreur_spam = testClassifieur(dossier_test_spams, classifieur, True)

print("[Amelioration] calcul des erreurs sur les hams...")
erreur_hams = testClassifieur(dossier_test_hams, classifieur, False)
print("[Amelioration] Erreur de test sur " + str(mSpam) + " SPAM    : " + str(round(erreur_spam * 100)) + " %")
print("[Amelioration] Erreur de test sur " + str(mHam) + " HAM    : " + str(round(erreur_hams * 100)) + " %")
erreur_globale = (erreur_spam * mSpam + erreur_hams * mHam) / (mSpam + mHam)
print("[Amelioration] Erreur de test globale sur " + str((mSpam + mHam)) + " : " + str(round(erreur_globale * 100)) + " %")