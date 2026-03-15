"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 7 : Modèle de Markov au niveau des mots (Bonus)
-------------------------------------------------------
Au lieu de traiter les caractères comme états, on traite les MOTS.
Cela capture des dépendances syntaxiques de plus haut niveau.

Avantages
---------
• Les transitions mot-à-mot capturent la syntaxe ("the city" → "is")
• Meilleure cohérence lexicale

Inconvénients
-------------
• Vocabulaire très grand → matrice très creuse
• Beaucoup plus de données nécessaires
• Transitions inconnues fréquentes → besoin de lissage plus agressif

Connexion aux LLMs modernes
----------------------------
Les modèles comme GPT utilisent une tokenisation par sous-mots (BPE)
qui est un compromis entre niveau caractère et niveau mot.
Vocabulaire typique : 32 000 – 100 000 tokens.
"""

import math
import numpy as np
from collections import defaultdict, Counter
import re


# -----------------------------------------------------------------------
# Prétraitement au niveau des mots
# -----------------------------------------------------------------------

SPECIAL_START = "<START>"
SPECIAL_END   = "<END>"
UNKNOWN       = "<UNK>"


def tokenize(text: str, max_vocab: int = 500) -> tuple:
    """
    Tokenise un texte brut en liste de mots et construit le vocabulaire.

    Paramètres
    ----------
    text      : texte brut (non prétraité au niveau caractère)
    max_vocab : taille maximale du vocabulaire (tokens les plus fréquents)

    Retourne
    --------
    tokens    : list[str] – séquence de tokens
    vocab     : set[str]  – vocabulaire retenu
    """
    # Nettoyage minimal : minuscules, garder lettres et espaces
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()

    if not words:
        raise ValueError("Le texte est vide après tokenisation.")

    # Construire le vocabulaire limité aux max_vocab mots les plus fréquents
    freq  = Counter(words)
    vocab = {w for w, _ in freq.most_common(max_vocab)}
    vocab.update([SPECIAL_START, SPECIAL_END, UNKNOWN])

    # Remplacer les mots hors vocabulaire par <UNK>
    tokens = [w if w in vocab else UNKNOWN for w in words]
    tokens = [SPECIAL_START] + tokens + [SPECIAL_END]

    return tokens, vocab


# -----------------------------------------------------------------------
# Modèle de Markov d'ordre 1 au niveau des mots
# -----------------------------------------------------------------------

def build_word_model(tokens: list, vocab: set, smoothing: float = 1.0) -> dict:
    """
    Construit le modèle de transition entre mots avec lissage.

    Retourne
    --------
    model : dict { mot_source : {mot_suivant : probabilité} }
    """
    counts = defaultdict(Counter)

    for k in range(len(tokens) - 1):
        counts[tokens[k]][tokens[k + 1]] += 1.0

    model = {}
    vocab_list = list(vocab)
    V = len(vocab_list)

    for word, counter in counts.items():
        total = sum(counter.values()) + smoothing * V
        model[word] = {
            w: (counter.get(w, 0.0) + smoothing) / total
            for w in vocab_list
        }

    return model


def generate_words(model: dict, max_words: int = 30, seed: int = None) -> str:
    """Génère une phrase mot-à-mot à partir du modèle."""
    if seed is not None:
        np.random.seed(seed)

    current = SPECIAL_START
    result  = []

    for _ in range(max_words):
        if current not in model:
            break
        dist      = model[current]
        words     = list(dist.keys())
        probs     = list(dist.values())
        total     = sum(probs)
        probs     = [p / total for p in probs]
        next_word = np.random.choice(words, p=probs)

        if next_word == SPECIAL_END:
            break
        if next_word != UNKNOWN:
            result.append(next_word)
        current = next_word

    return " ".join(result)


def evaluate_word_model(tokens: list, model: dict, label: str = "") -> None:
    """Calcule la log-vraisemblance du modèle sur une séquence de tokens."""
    ll = 0.0
    n  = 0
    for k in range(len(tokens) - 1):
        src = tokens[k]
        dst = tokens[k + 1]
        if src in model and dst in model[src]:
            p = model[src][dst]
            if p > 0:
                ll += math.log(p)
                n  += 1

    pp = math.exp(-ll / n) if n > 0 else float("inf")
    print(f"  [{label}]  LL = {ll:.2f}  |  Perplexité = {pp:.4f}  |  n = {n}")


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # Textes de démonstration
    DEMO_TRAIN = (
        "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
        "at the mouth of the Bou Regreg river. The city has a rich history dating back "
        "to ancient times and is known for its beautiful architecture. "
        "The Hassan Tower is one of the most famous symbols of Rabat. "
        "The city also hosts the Royal Palace and several important museums. "
        "Rabat became the capital of Morocco during the French protectorate. "
        "Today it is a modern city with many universities and institutions."
    ) * 10

    DEMO_TEST = (
        "Casablanca is the largest city of Morocco and its main economic hub. "
        "The city is known for the Hassan II Mosque, one of the largest mosques. "
        "It is a modern city with a strong industrial and financial sector."
    ) * 5

    # Tokenisation
    print("Tokenisation du texte d'entraînement…")
    tokens_train, vocab = tokenize(DEMO_TRAIN, max_vocab=300)
    tokens_test,  _     = tokenize(DEMO_TEST,  max_vocab=300)
    # Remplacer les tokens de test hors du vocabulaire d'entraînement
    tokens_test = [t if t in vocab else UNKNOWN for t in tokens_test]

    print(f"  Taille du vocabulaire : {len(vocab)} mots")
    print(f"  Tokens entraînement   : {len(tokens_train)}")

    # Construction du modèle
    model = build_word_model(tokens_train, vocab, smoothing=0.1)

    # Évaluation
    print("\nÉvaluation :")
    evaluate_word_model(tokens_train, model, "Texte d'entraînement")
    evaluate_word_model(tokens_test,  model, "Texte de test       ")

    # Génération
    print("\nPhrases générées :")
    for i in range(5):
        phrase = generate_words(model, max_words=20, seed=i * 7)
        print(f"  [{i+1}] {phrase}")
