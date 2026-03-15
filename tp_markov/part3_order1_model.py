"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 3 : Modèle de Markov d'ordre 1 (caractères)
----------------------------------------------------
Construction de la matrice de transition M (27×27) puis de la
matrice de probabilités P avec lissage de Laplace (add-1).

Algorithme
----------
1. Construire M[i][j] = nombre de fois que la lettre j suit la lettre i
2. Appliquer le lissage de Laplace : M[i][j] += 1  (évite les prob. nulles)
3. Normaliser chaque ligne → matrice stochastique P
   P[i][j] = M[i][j] / somme_ligne_i
"""

import numpy as np
from part2_preprocessing import VOCAB, CHAR2IDX, VOCAB_SIZE, preprocess


def build_transition_matrix(text: str, smoothing: float = 1.0) -> np.ndarray:
    """
    Construit la matrice de probabilités de transition d'ordre 1.

    Paramètres
    ----------
    text      : texte prétraité (avec marqueurs ^ et $)
    smoothing : valeur du lissage de Laplace (défaut = 1)

    Retourne
    --------
    P : np.ndarray de forme (VOCAB_SIZE, VOCAB_SIZE)
        P[i, j] = P(lettre_j | lettre_i)
    """
    # ---- Étape 1 : Matrice de comptage ----
    M = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=float)

    for k in range(len(text) - 1):
        char_curr = text[k]
        char_next = text[k + 1]
        # On ignore les caractères hors vocabulaire
        if char_curr in CHAR2IDX and char_next in CHAR2IDX:
            i = CHAR2IDX[char_curr]
            j = CHAR2IDX[char_next]
            M[i, j] += 1.0

    # ---- Étape 2 : Lissage de Laplace ----
    M += smoothing

    # ---- Étape 3 : Normalisation (matrice stochastique) ----
    row_sums = M.sum(axis=1, keepdims=True)
    # Éviter la division par zéro (ne devrait pas arriver avec smoothing > 0)
    row_sums[row_sums == 0] = 1.0
    P = M / row_sums

    return P


def top_transitions(P: np.ndarray, n: int = 10) -> None:
    """Affiche les n transitions les plus probables."""
    transitions = []
    for i in range(VOCAB_SIZE):
        for j in range(VOCAB_SIZE):
            transitions.append((VOCAB[i], VOCAB[j], P[i, j]))
    transitions.sort(key=lambda x: -x[2])
    print(f"\nTop-{n} transitions les plus fréquentes :")
    print(f"{'De':^6} {'Vers':^6} {'Prob':^10}")
    print("-" * 25)
    for src, dst, prob in transitions[:n]:
        src_disp = repr(src) if src == " " else src
        dst_disp = repr(dst) if dst == " " else dst
        print(f"  {src_disp:^6} → {dst_disp:^6}  {prob:.4f}")


def verify_stochastic(P: np.ndarray) -> bool:
    """Vérifie que chaque ligne de P somme bien à 1.0."""
    row_sums = P.sum(axis=1)
    ok = np.allclose(row_sums, 1.0, atol=1e-6)
    if ok:
        print("[OK] La matrice est bien stochastique (chaque ligne somme à 1).")
    else:
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        print(f"[ERREUR] Lignes non normalisées : indices {bad}")
    return ok


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # Charger le texte prétraité (généré par partie 2)
    try:
        with open("train_clean.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
    except FileNotFoundError:
        print("[AVERTISSEMENT] train_clean.txt non trouvé – utilisation d'un texte de démo.")
        DEMO = (
            "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
            "at the mouth of the Bou Regreg river. The city has a rich history dating back "
            "to ancient times and is known for its beautiful architecture." * 10
        )
        train_text = preprocess(DEMO)

    # Construction de la matrice de transition
    print("Construction de la matrice de transition (ordre 1)…")
    P = build_transition_matrix(train_text, smoothing=1.0)

    # Vérification
    verify_stochastic(P)

    # Affichage des top transitions
    top_transitions(P, n=10)

    # Exemples de probabilités
    print("\nExemples de probabilités de transition :")
    for src, dst in [("t", "h"), ("h", "e"), ("e", " "), ("q", "u")]:
        i, j = CHAR2IDX[src], CHAR2IDX[dst]
        print(f"  P('{dst}' | '{src}') = {P[i, j]:.4f}")

    # Sauvegarde de la matrice
    np.save("transition_matrix_order1.npy", P)
    print("\n[OK] Matrice sauvegardée dans 'transition_matrix_order1.npy'")
