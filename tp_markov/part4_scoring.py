"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 4 : Évaluation du modèle (Score & Perplexité)
-----------------------------------------------------
Métriques utilisées
-------------------
• Log-vraisemblance (log-likelihood) :
      LL = Σ log P(c_{i+1} | c_i)

  → On utilise le logarithme pour éviter le sous-dépassement numérique
    (underflow) lors de la multiplication de très petites probabilités.

• Perplexité (standard NLP) :
      PP = exp( -LL / n )

  → Plus la perplexité est faible, mieux le modèle "comprend" le texte.
"""

import math
import numpy as np
from part2_preprocessing import CHAR2IDX, preprocess
from part3_order1_model  import build_transition_matrix


def log_likelihood(text: str, P: np.ndarray) -> float:
    """
    Calcule la log-vraisemblance d'un texte sous le modèle P.

    Paramètres
    ----------
    text : texte prétraité
    P    : matrice de probabilités (VOCAB_SIZE × VOCAB_SIZE)

    Retourne
    --------
    LL   : float  (valeur négative – plus proche de 0 = meilleur)
    """
    ll = 0.0
    count = 0
    for k in range(len(text) - 1):
        c_curr = text[k]
        c_next = text[k + 1]
        if c_curr in CHAR2IDX and c_next in CHAR2IDX:
            i = CHAR2IDX[c_curr]
            j = CHAR2IDX[c_next]
            prob = P[i, j]
            # Éviter log(0) – ne devrait pas arriver avec lissage
            if prob > 0:
                ll += math.log(prob)
                count += 1
    return ll, count


def perplexity(text: str, P: np.ndarray) -> float:
    """
    Calcule la perplexité d'un texte sous le modèle P.

    PP = exp( -LL / n )   où n est le nombre de transitions évaluées.
    """
    ll, n = log_likelihood(text, P)
    if n == 0:
        return float("inf")
    return math.exp(-ll / n)


def score_model(text: str, P: np.ndarray, label: str = "") -> dict:
    """
    Évalue le modèle sur un texte et affiche un résumé complet.

    Retourne un dictionnaire avec les métriques.
    """
    ll, n = log_likelihood(text, P)
    pp    = math.exp(-ll / n) if n > 0 else float("inf")

    # Score normalisé (fraction de tentatives correctes selon l'énoncé)
    # On utilise ici la probabilité moyenne comme indicateur
    avg_prob = math.exp(ll / n) if n > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  Évaluation : {label}")
    print(f"{'='*50}")
    print(f"  Transitions évaluées : {n}")
    print(f"  Log-vraisemblance    : {ll:.2f}")
    print(f"  Perplexité           : {pp:.4f}")
    print(f"  Probabilité moyenne  : {avg_prob:.6f}")

    return {"log_likelihood": ll, "perplexity": pp, "n_transitions": n}


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # ---- Chargement des textes ----
    try:
        with open("train_clean.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("test_clean.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
    except FileNotFoundError:
        print("[AVERTISSEMENT] Fichiers non trouvés – utilisation de textes de démo.")
        DEMO_TRAIN = (
            "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
            "at the mouth of the Bou Regreg river. The city has a rich history."
        ) * 15
        DEMO_TEST = (
            "Casablanca is the largest city of Morocco and its main economic hub. "
            "The city is known for the Hassan II Mosque, one of the largest mosques."
        ) * 10
        train_text = preprocess(DEMO_TRAIN)
        test_text  = preprocess(DEMO_TEST)

    # ---- Chargement ou reconstruction de la matrice ----
    try:
        P = np.load("transition_matrix_order1.npy")
        print("[INFO] Matrice chargée depuis 'transition_matrix_order1.npy'")
    except FileNotFoundError:
        print("[INFO] Reconstruction de la matrice…")
        P = build_transition_matrix(train_text)

    # ---- Évaluations ----
    # 1. Même texte que l'entraînement (devrait avoir la perplexité la plus basse)
    score_train = score_model(train_text, P, "Texte d'entraînement (Rabat)")

    # 2. Texte de test – même domaine (un peu moins bon)
    score_test  = score_model(test_text, P,  "Texte de test (Casablanca)")

    # 3. Texte totalement aléatoire (devrait avoir la plus haute perplexité)
    import random, string
    gibberish = "^" + "".join(random.choices(string.ascii_lowercase + " ", k=500)) + "$"
    score_gib  = score_model(gibberish, P, "Texte aléatoire (bruit)")

    # ---- Comparaison ----
    print("\n\n--- Comparaison des perplexités ---")
    print(f"  Entraînement : {score_train['perplexity']:.4f}")
    print(f"  Test         : {score_test['perplexity']:.4f}")
    print(f"  Aléatoire    : {score_gib['perplexity']:.4f}")
    print("\n  → Plus la perplexité est faible, mieux le modèle modélise le texte.")
