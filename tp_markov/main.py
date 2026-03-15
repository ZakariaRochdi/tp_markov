"""
TP Chaîne de Markov – Analyse de Texte
========================================
main.py  –  Script principal d'exécution complète du TP
---------------------------------------------------------
Lance les 7 parties du TP dans l'ordre et produit tous les résultats.

Usage
-----
    python main.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # mode sans interface graphique
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Ajouter le répertoire courant au chemin Python
# -----------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from part2_preprocessing import preprocess, VOCAB, CHAR2IDX, VOCAB_SIZE
from part3_order1_model  import build_transition_matrix, verify_stochastic, top_transitions
from part4_scoring       import score_model, log_likelihood
from part5_generation    import generate_text
from part6_orderN_model  import build_ngram_model, generate_ngram, perplexity_ngram
from part7_wordlevel     import tokenize, build_word_model, generate_words, evaluate_word_model


# =======================================================================
# Textes de démonstration (utilisés si les fichiers .txt sont absents)
# =======================================================================
DEMO_TRAIN = (
    "Rabat is the capital city of Morocco. It is situated on the Atlantic Ocean "
    "at the mouth of the Bou Regreg river. The city has a rich history dating back "
    "to ancient times and is known for its beautiful architecture. "
    "The Hassan Tower is one of the most famous symbols of Rabat. "
    "The city also hosts the Royal Palace and several important museums. "
    "Rabat became the capital of Morocco during the French protectorate. "
    "Today it is a modern city with many universities and research institutions. "
    "The medina of Rabat is a UNESCO World Heritage Site. "
    "The Kasbah of the Udayas overlooks the Atlantic Ocean and the Bou Regreg river."
) * 20

DEMO_TEST = (
    "Casablanca is the largest city of Morocco and its main economic hub. "
    "The city is known for the Hassan II Mosque, one of the largest mosques in the world. "
    "Casablanca is an important port city and a center for trade and commerce. "
    "The name Casablanca means White House in Spanish. "
    "It was founded in the 10th century and has grown into a major metropolis."
) * 10


# =======================================================================
def load_texts():
    """Charge ou génère les textes d'entraînement et de test."""
    try:
        with open("train_clean.txt", "r", encoding="utf-8") as f:
            train = f.read()
        with open("test_clean.txt", "r", encoding="utf-8") as f:
            test = f.read()
        print("[OK] Textes chargés depuis les fichiers.\n")
    except FileNotFoundError:
        print("[INFO] Fichiers non trouvés – génération de textes de démo.\n")
        train = preprocess(DEMO_TRAIN)
        test  = preprocess(DEMO_TEST)
        with open("train_clean.txt", "w") as f:
            f.write(train)
        with open("test_clean.txt", "w") as f:
            f.write(test)
    return train, test


# =======================================================================
def visualize_transition_matrix(P: np.ndarray, filename: str = "heatmap_transition.png"):
    """Génère et sauvegarde une heatmap de la matrice de transition."""
    labels = VOCAB
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(P, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Probabilité de transition")

    ax.set_xticks(range(VOCAB_SIZE))
    ax.set_yticks(range(VOCAB_SIZE))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Caractère suivant")
    ax.set_ylabel("Caractère courant")
    ax.set_title("Matrice de transition – Modèle de Markov d'ordre 1\n(Probabilités P[i][j])")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[OK] Heatmap sauvegardée dans '{filename}'")


def visualize_perplexity_comparison(results: dict, filename: str = "perplexity_comparison.png"):
    """Génère un graphique comparant les perplexités selon l'ordre."""
    orders = list(results.keys())
    pp_train = [results[o]["pp_train"] for o in orders]
    pp_test  = [results[o]["pp_test"]  for o in orders]

    x = np.arange(len(orders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, pp_train, width, label="Entraînement", color="#2196F3")
    bars2 = ax.bar(x + width/2, pp_test,  width, label="Test",          color="#FF5722")

    ax.set_xlabel("Ordre du modèle")
    ax.set_ylabel("Perplexité")
    ax.set_title("Perplexité selon l'ordre du modèle de Markov\n(plus bas = meilleur)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ordre {o}" for o in orders])
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[OK] Graphique de perplexité sauvegardé dans '{filename}'")


# =======================================================================
# EXÉCUTION PRINCIPALE
# =======================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  TP CHAÎNE DE MARKOV – ANALYSE DE TEXTE")
    print("  IDF – 2SCL – 2IA | Pr. M. Naoum | 2025-2026")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Chargement des textes
    # ------------------------------------------------------------------
    print("\n[PARTIE 1 & 2] Acquisition et prétraitement des textes")
    print("-" * 50)
    train_text, test_text = load_texts()
    print(f"  Texte d'entraînement : {len(train_text)} caractères")
    print(f"  Texte de test        : {len(test_text)} caractères")

    # ------------------------------------------------------------------
    # 2. Modèle d'ordre 1
    # ------------------------------------------------------------------
    print("\n[PARTIE 3] Construction de la matrice de transition (ordre 1)")
    print("-" * 50)
    P = build_transition_matrix(train_text, smoothing=1.0)
    verify_stochastic(P)
    top_transitions(P, n=10)
    np.save("transition_matrix_order1.npy", P)
    visualize_transition_matrix(P)

    # ------------------------------------------------------------------
    # 3. Évaluation
    # ------------------------------------------------------------------
    print("\n[PARTIE 4] Évaluation du modèle d'ordre 1")
    print("-" * 50)
    import random, string
    gibberish = "^" + "".join(random.choices(string.ascii_lowercase + " ", k=500)) + "$"

    s_train = score_model(train_text, P, "Entraînement (Rabat)")
    s_test  = score_model(test_text,  P, "Test (Casablanca)")
    s_gib   = score_model(gibberish,  P, "Texte aléatoire")

    # ------------------------------------------------------------------
    # 4. Génération avec l'ordre 1
    # ------------------------------------------------------------------
    print("\n[PARTIE 5] Génération de texte (ordre 1)")
    print("-" * 50)
    for strat, k in [("full", 5), ("top_k", 5), ("greedy", 1)]:
        gen = generate_text(P, max_length=150, strategy=strat, k=k, seed=42)
        print(f"  [{strat:10s}] {gen}")

    # ------------------------------------------------------------------
    # 5. Modèles d'ordre supérieur
    # ------------------------------------------------------------------
    print("\n[PARTIE 6] Modèles d'ordre supérieur (N = 1, 2, 3)")
    print("-" * 50)
    ngram_results = {}
    for order in [1, 2, 3]:
        model    = build_ngram_model(train_text, order=order)
        pp_train = perplexity_ngram(train_text, model, order)
        pp_test  = perplexity_ngram(test_text,  model, order)
        ngram_results[order] = {"model": model, "pp_train": pp_train, "pp_test": pp_test}
        print(f"  Ordre {order}: PP_train = {pp_train:8.3f} | PP_test = {pp_test:8.3f}")
        # Exemple de texte généré
        gen = generate_ngram(model, order=order, max_length=200, seed=42)
        print(f"         Texte : {gen[:100]}…\n")

    visualize_perplexity_comparison(ngram_results)

    # ------------------------------------------------------------------
    # 6. Modèle au niveau des mots
    # ------------------------------------------------------------------
    print("\n[PARTIE 7] Modèle au niveau des mots")
    print("-" * 50)
    tokens_train, vocab = tokenize(DEMO_TRAIN, max_vocab=300)
    tokens_test,  _     = tokenize(DEMO_TEST,  max_vocab=300)
    from part7_wordlevel import UNKNOWN
    tokens_test = [t if t in vocab else UNKNOWN for t in tokens_test]
    word_model  = build_word_model(tokens_train, vocab, smoothing=0.1)
    evaluate_word_model(tokens_train, word_model, "Entraînement")
    evaluate_word_model(tokens_test,  word_model, "Test")
    print("\n  Phrases générées (modèle mots) :")
    for i in range(3):
        print(f"  [{i+1}] {generate_words(word_model, max_words=20, seed=i*7)}")

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL")
    print("=" * 65)
    print(f"  Ordre 1 – Perplexité train : {s_train['perplexity']:.3f}  test : {s_test['perplexity']:.3f}")
    for o in [1, 2, 3]:
        r = ngram_results[o]
        print(f"  Ordre {o} (n-gram) – PP_train : {r['pp_train']:.3f}  PP_test : {r['pp_test']:.3f}")
    print("\n  Fichiers générés :")
    print("    - transition_matrix_order1.npy")
    print("    - heatmap_transition.png")
    print("    - perplexity_comparison.png")
    print("\n[TERMINÉ] TP exécuté avec succès.")
