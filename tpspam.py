import numpy as np
import os
import math


def lireMail(fichier, dictionnaire):
    """
    Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
    """
    f = open(fichier, "r", encoding="ascii", errors="surrogateescape")
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
    mots = [x for x in mots if len(x) >= 3]  # 4. Consigne, on ne garde que les mots de 3 lettres et plus
    print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
    f.close()
    return mots[:-1]


def apprendBinomial(dossier, fichiers, dictionnaire):
    """
    Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
    Retourne un vecteur b de paramètres

    """
    freq = np.zeros_like(dictionnaire,
                         dtype=np.int32)  # compte / accu la présence des mots dans l'ensemble des mails spams (ou hams)

    for fichier in fichiers:
        v = lireMail("/".join([dossier, fichier]), dictionnaire)  # vecteur présence mots
        freq += np.array(v, dtype=np.int32)  # cast booleen en int, si la valeur d'un indice est True, +1 dans freq

    b = freq / len(fichiers)
    return b


def prediction(x, Pspam, Pham, bspam, bham):
    """
    	Prédit si un mail représenté par un vecteur booléen x est un spam
    	à partir du modèle de paramètres Pspam, Pham, bspam, bham.
    	Retourne True ou False.

    """
    epsilon = 1e-10  # Petite valeur pour éviter les erreurs de domaine

    log_pSpam = math.log(Pspam)
    log_pHam = math.log(Pham)
    for i, present in enumerate(x):
        if present:
            log_pSpam += math.log(bspam[i] + epsilon)
            log_pHam += math.log(bham[i] + epsilon)
        else:
            log_pSpam += math.log(1 - bspam[i] + epsilon)
            log_pHam += math.log(1 - bham[i] + epsilon)

    return log_pSpam > log_pHam




def test(dossier, isSpam, Pspam, Pham, bspam, bham):
    fichiers = os.listdir(dossier)
    nb_erreur = 0

    for fichier in fichiers: #On parcours chaque fichier du dossier
        x = lireMail(dossier + "/" + fichier, dictionnaire) #On lit le fichier
        isSpamPrediction = prediction(x, Pspam, Pham, bspam, bham) #On fait une prediction
        type_mail = "SPAM" if isSpam else "HAM"

        if (isSpam and not isSpamPrediction) or (not isSpam and isSpamPrediction):
            #Si c'est un spam et que la prediction est un ham ou si c'est un ham et que la prediction est un spam
            #Il y a donc une erreur
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

mSpam = 300 #len(fichiersspams)
mHam = 300 #len(fichiershams)

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
print("calcul des erreurs sur les spams...")
erreur_spam = test(dossier_test_spams, True, Pspam, Pham, bspam, bham)

print("calcul des erreurs sur les hams...")
erreur_hams = test(dossier_test_hams, False, Pspam, Pham, bspam, bham)

print("Erreur de test sur " + str(mSpam) + " SPAM    : " + str(erreur_spam*100) + " %")
print("Erreur de test sur " + str(mHam) + " HAM    : " + str(erreur_hams*100) + " %")

erreur_globale = (erreur_spam * mSpam + erreur_hams * mHam) / (mSpam + mHam)
print("Erreur de test globale sur " + str(mSpam + mHam) + " : " + str(round(erreur_globale*100)) + " %")

