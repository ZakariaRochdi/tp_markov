"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 6 : Modèle de Markov d'ordre N (ordre supérieur)
---------------------------------------------------------
Pour un modèle d'ordre n, l'état est un n-uplet de caractères :
    P(c_{t+1} | c_t, c_{t-1}, …, c_{t-n+1})

Avantages
---------
• Capture des dépendances plus longues → texte généré plus cohérent
• Exemple : avec ordre 3 et l'état "ens", les états suivants possibles
  sont "ns ", "nsa", "nsb", …, "nsz"

Inconvénients
-------------
• L'espace d'états explose : |V|^n états possibles
• Matrice très creuse pour les grands ordres → on utilise un dict
• Risque d'overfitting (mémorisation du texte d'entraînement)

Implémentation
--------------
On utilise des dictionnaires Python (hashmap) au lieu d'un tableau numpy
pour gérer efficacement la sparsité.
"""

import math
import numpy as np
from collections import defaultdict, Counter
from part2_preprocessing import VOCAB, CHAR2IDX, preprocess


# -----------------------------------------------------------------------
# Construction du modèle d'ordre N
# -----------------------------------------------------------------------

def build_ngram_model(text: str, order: int = 3, smoothing: float = 1.0) -> dict:
    """
    Construit un modèle de Markov d'ordre `order` avec un dictionnaire.

    Paramètres
    ----------
    text     : texte prétraité (avec marqueurs ^ et $)
    order    : nombre de caractères dans chaque état (n-gram de contexte)
    smoothing: lissage de Laplace ajouté à chaque compte

    Retourne
    --------
    model : dict  { context_str : {next_char : probability} }
    """
    # 1. Comptage des transitions
    counts = defaultdict(Counter)   # counts[contexte][suivant] = nombre

    # Ajouter (order - 1) marqueurs de début pour initialiser le contexte
    padded = "^" * (order - 1) + text

    for k in range(len(padded) - order):
        context  = padded[k: k + order]        # n caractères de contexte
        next_ch  = padded[k + order]           # caractère suivant

        # Ignorer si caractères hors vocabulaire
        if all(c in CHAR2IDX for c in context) and next_ch in CHAR2IDX:
            counts[context][next_ch] += 1.0

    # 2. Normalisation avec lissage de Laplace
    model = {}
    vocab_list = list(CHAR2IDX.keys())

    for context, counter in counts.items():
        # Ajouter le lissage sur tous les caractères du vocabulaire
        total = sum(counter.values()) + smoothing * len(vocab_list)
        model[context] = {}
        for ch in vocab_list:
            model[context][ch] = (counter.get(ch, 0.0) + smoothing) / total

    return model


# -----------------------------------------------------------------------
# Génération de texte avec le modèle d'ordre N
# -----------------------------------------------------------------------

def generate_ngram(model: dict,
                   order: int = 3,
                   max_length: int = 300,
                   seed: int = None) -> str:
    """
    Génère du texte à partir du modèle d'ordre `order`.

    Paramètres
    ----------
    model      : dictionnaire { context : {next_char : prob} }
    order      : ordre du modèle
    max_length : longueur maximale du texte généré
    seed       : graine aléatoire

    Retourne
    --------
    Texte généré (str) sans les marqueurs ^ et $
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialisation : contexte = (order) marqueurs de début
    context = "^" * order
    result  = []

    for _ in range(max_length):
        # Récupérer la distribution depuis le modèle
        if context in model:
            dist      = model[context]
            chars     = list(dist.keys())
            probs     = list(dist.values())
            # Renormaliser (précaution numérique)
            total     = sum(probs)
            probs     = [p / total for p in probs]
            next_char = np.random.choice(chars, p=probs)
        else:
            # Contexte inconnu → choisir uniformément dans le vocabulaire
            next_char = np.random.choice(list(CHAR2IDX.keys()))

        if next_char == "$":
            break

        result.append(next_char)
        # Faire glisser la fenêtre de contexte
        context = context[1:] + next_char

    return "".join(result)


# -----------------------------------------------------------------------
# Évaluation du modèle d'ordre N
# -----------------------------------------------------------------------

def log_likelihood_ngram(text: str, model: dict, order: int) -> tuple:
    """Calcule la log-vraisemblance d'un texte sous le modèle d'ordre N."""
    padded = "^" * (order - 1) + text
    ll     = 0.0
    n      = 0
    for k in range(len(padded) - order):
        context = padded[k: k + order]
        next_ch = padded[k + order]
        if context in model and next_ch in model[context]:
            prob = model[context][next_ch]
            if prob > 0:
                ll += math.log(prob)
                n  += 1
    return ll, n


def perplexity_ngram(text: str, model: dict, order: int) -> float:
    """Calcule la perplexité du modèle d'ordre N sur un texte."""
    ll, n = log_likelihood_ngram(text, model, order)
    return math.exp(-ll / n) if n > 0 else float("inf")


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # ---- Chargement du texte d'entraînement ----
    try:
        with open("train_clean.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("test_clean.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
    except FileNotFoundError:
        print("[AVERTISSEMENT] Fichiers non trouvés – utilisation de textes de démo.")
        DEMO_TRAIN = (
            "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
            "at the mouth of the Bou Regreg river. The city has a rich history dating back "
            "to ancient times and is known for its beautiful architecture. "
            "The Hassan Tower is one of the most famous symbols of Rabat."
        ) * 30
        DEMO_TEST = (
            "Casablanca is the largest city of Morocco and its main economic hub. "
            "The city is known for the Hassan II Mosque, one of the largest mosques."
        ) * 15
        train_text = preprocess(DEMO_TRAIN)
        test_text  = preprocess(DEMO_TEST)

    # ---- Comparaison des ordres 1, 2, 3 ----
    print("Construction et comparaison des modèles d'ordres 1, 2, 3…\n")

    results = {}
    for order in [1, 2, 3]:
        model = build_ngram_model(train_text, order=order, smoothing=1.0)
        pp_train = perplexity_ngram(train_text, model, order)
        pp_test  = perplexity_ngram(test_text,  model, order)
        results[order] = {"model": model, "pp_train": pp_train, "pp_test": pp_test}
        print(f"Ordre {order}  |  Perplexité entraînement : {pp_train:8.3f}  |  test : {pp_test:8.3f}")

    # ---- Génération de texte pour chaque ordre ----
    print("\n\n=== Textes générés par ordre ===")
    for order in [1, 2, 3]:
        model = results[order]["model"]
        print(f"\n--- Ordre {order} ---")
        for i in range(2):
            gen = generate_ngram(model, order=order, max_length=200, seed=42 + i)
            print(f"  [{i+1}] {gen}")
