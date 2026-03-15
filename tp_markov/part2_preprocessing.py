"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 2 : Prétraitement du texte
----------------------------------
Cette étape normalise le texte brut :
  - Conversion en minuscules
  - Suppression de toute ponctuation et chiffres
  - Conservation uniquement de a-z et de l'espace
  - Ajout des marqueurs de début (^) et de fin ($)

Vocabulaire final V = {^, $, espace, a, b, …, z}  →  |V| = 29
"""

import re
import string

# -----------------------------------------------------------------------
# Vocabulaire : 27 caractères de base + marqueurs ^ et $
# Index 0 = espace, indices 1-26 = a-z, index 27 = '^', index 28 = '$'
# -----------------------------------------------------------------------
ALPHABET = " " + string.ascii_lowercase          # 27 chars : espace + a-z
VOCAB    = ["^"] + list(ALPHABET) + ["$"]        # 29 éléments
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}   # dict char → indice
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}   # dict indice → char
VOCAB_SIZE = len(VOCAB)                           # 29


def preprocess(raw_text: str) -> str:
    """
    Nettoie et normalise un texte brut.

    Étapes
    ------
    1. Minuscules
    2. Remplacement de tout caractère hors [a-z ] par un espace
    3. Réduction des espaces multiples en un seul
    4. Ajout des marqueurs ^ (début) et $ (fin)

    Paramètres
    ----------
    raw_text : texte brut (str)

    Retourne
    --------
    Texte normalisé encadré de ^ et $ (str)

    Exemple
    -------
    >>> preprocess("Hello, World!")
    '^hello world$'
    """
    if not raw_text:
        raise ValueError("Le texte en entrée est vide.")

    # 1. Minuscules
    text = raw_text.lower()

    # 2. Garder uniquement les lettres a-z et les espaces
    text = re.sub(r"[^a-z ]", " ", text)

    # 3. Supprimer les espaces multiples
    text = re.sub(r" +", " ", text).strip()

    # 4. Ajouter les marqueurs de début et de fin
    text = "^" + text + "$"

    return text


def load_and_preprocess(filepath: str) -> str:
    """Charge un fichier texte brut et le prétraite."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return preprocess(raw)


def text_stats(text: str) -> None:
    """Affiche quelques statistiques sur le texte prétraité."""
    chars = [c for c in text if c in CHAR2IDX]
    freq  = {c: chars.count(c) for c in ALPHABET}
    freq_sorted = sorted(freq.items(), key=lambda x: -x[1])

    print(f"  Longueur totale         : {len(text)} caractères")
    print(f"  Caractères valides      : {len(chars)}")
    print(f"  Top-10 caractères       : {freq_sorted[:10]}")


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    # Charger les fichiers bruts générés par la partie 1
    try:
        train_clean = load_and_preprocess("train_raw.txt")
        test_clean  = load_and_preprocess("test_raw.txt")
    except FileNotFoundError:
        # Textes de démonstration si les fichiers ne sont pas présents
        print("[AVERTISSEMENT] Fichiers bruts non trouvés – utilisation de textes de démonstration.\n")
        DEMO_TRAIN = (
            "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
            "at the mouth of the Bou Regreg river. The city has a rich history dating back "
            "to ancient times and is known for its beautiful architecture."
        )
        DEMO_TEST = (
            "Casablanca is the largest city of Morocco and its main economic hub. "
            "The city is known for the Hassan II Mosque, one of the largest mosques "
            "in the world. Its name means White House in Spanish."
        )
        train_clean = preprocess(DEMO_TRAIN)
        test_clean  = preprocess(DEMO_TEST)

    # Sauvegarder les textes prétraités
    with open("train_clean.txt", "w", encoding="utf-8") as f:
        f.write(train_clean)
    with open("test_clean.txt", "w", encoding="utf-8") as f:
        f.write(test_clean)

    print("=== Statistiques – Texte d'entraînement ===")
    text_stats(train_clean)
    print("\n=== Statistiques – Texte de test ===")
    text_stats(test_clean)

    print(f"\nAperçu entraînement : {train_clean[:80]}…")
    print(f"Aperçu test         : {test_clean[:80]}…")

    print("\n[OK] Fichiers 'train_clean.txt' et 'test_clean.txt' sauvegardés.")
