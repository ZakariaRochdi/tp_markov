"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 5 : Génération de texte
-------------------------------
Algorithme de génération séquentielle :
  1. Démarrer depuis le marqueur '^'
  2. Échantillonner le prochain caractère selon P(· | état courant)
  3. Ajouter au résultat et avancer
  4. S'arrêter au marqueur '$' ou lorsque max_length est atteint

Stratégies d'échantillonnage disponibles
-----------------------------------------
• full_sampling : échantillon depuis la distribution complète
• top_k         : limiter au top-k caractères les plus probables
• greedy        : choisir toujours le caractère le plus probable (k=1)
"""

import numpy as np
import random
from part2_preprocessing import VOCAB, CHAR2IDX, VOCAB_SIZE, preprocess
from part3_order1_model  import build_transition_matrix


def generate_text(P: np.ndarray,
                  max_length: int = 200,
                  strategy: str = "full",
                  k: int = 5,
                  seed: int = None) -> str:
    """
    Génère du texte en parcourant la chaîne de Markov.

    Paramètres
    ----------
    P          : matrice de transition (VOCAB_SIZE × VOCAB_SIZE)
    max_length : nombre maximal de caractères générés
    strategy   : 'full' | 'top_k' | 'greedy'
    k          : taille du top-k (utilisé seulement si strategy='top_k')
    seed       : graine aléatoire pour la reproductibilité

    Retourne
    --------
    Texte généré (str), sans les marqueurs ^ et $
    """
    if seed is not None:
        np.random.seed(seed)

    result   = []
    current  = "^"               # Toujours démarrer au marqueur de début
    idx_end  = CHAR2IDX["$"]

    for _ in range(max_length):
        i = CHAR2IDX[current]
        probs = P[i].copy()      # Distribution sur le prochain caractère

        # --- Stratégie d'échantillonnage ---
        if strategy == "greedy":
            # Toujours choisir le caractère le plus probable
            next_idx = int(np.argmax(probs))

        elif strategy == "top_k":
            # Garder seulement les k caractères les plus probables
            top_k_idx = np.argsort(probs)[-k:]       # indices des k meilleurs
            top_k_probs = probs[top_k_idx]
            top_k_probs /= top_k_probs.sum()          # renormaliser
            next_idx = int(np.random.choice(top_k_idx, p=top_k_probs))

        else:  # "full" – distribution complète
            next_idx = int(np.random.choice(VOCAB_SIZE, p=probs))

        # Arrêt sur le marqueur de fin
        if next_idx == idx_end:
            break

        next_char = VOCAB[next_idx]
        result.append(next_char)
        current = next_char

    return "".join(result)


def demo_generation(P: np.ndarray, n_samples: int = 3) -> None:
    """Compare les trois stratégies de génération."""
    print("\n" + "="*60)
    print("  GÉNÉRATION DE TEXTE – Comparaison des stratégies")
    print("="*60)

    strategies = [
        ("Échantillonnage complet (full)", "full",   5),
        ("Top-k  (k=5)",                  "top_k",  5),
        ("Top-k  (k=10)",                 "top_k", 10),
        ("Greedy (k=1)",                  "greedy", 1),
    ]

    for name, strat, k in strategies:
        print(f"\n--- {name} ---")
        for i in range(n_samples):
            text = generate_text(P, max_length=150, strategy=strat, k=k, seed=42 + i)
            print(f"  [{i+1}] {text}")


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # ---- Chargement du texte d'entraînement ----
    try:
        with open("train_clean.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
    except FileNotFoundError:
        print("[AVERTISSEMENT] train_clean.txt non trouvé – utilisation d'un texte de démo.")
        DEMO = (
            "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
            "at the mouth of the Bou Regreg river. The city has a rich history dating back "
            "to ancient times and is known for its beautiful architecture. "
            "The Hassan Tower and the Kasbah of the Udayas are major landmarks."
        ) * 20
        train_text = preprocess(DEMO)

    # ---- Chargement ou reconstruction de la matrice ----
    try:
        P = np.load("transition_matrix_order1.npy")
    except FileNotFoundError:
        P = build_transition_matrix(train_text)

    # ---- Démonstration ----
    demo_generation(P, n_samples=3)
