"""
TP Chaîne de Markov – Analyse de Texte
========================================
Partie 1 : Acquisition des données (Web Scraping)
--------------------------------------------------
Ce module récupère un texte depuis une page web (Wikipedia ou Project Gutenberg).
On utilise BeautifulSoup pour extraire le texte brut depuis le HTML.
"""

import urllib.request
from bs4 import BeautifulSoup


def fetch_text_from_url(url: str, max_chars: int = 50000) -> str:
    """
    Récupère et extrait le texte brut depuis une URL.

    Paramètres
    ----------
    url      : adresse de la page web à scraper
    max_chars: nombre maximum de caractères à conserver (pour limiter la mémoire)

    Retourne
    --------
    Le texte brut extrait (str), ou une chaîne vide en cas d'erreur.
    """
    try:
        # ---- Téléchargement du contenu HTML ----
        print(f"[INFO] Téléchargement de : {url}")
        html_bytes = urllib.request.urlopen(url).read()

        # ---- Parsing avec BeautifulSoup ----
        soup = BeautifulSoup(html_bytes, "html.parser")

        # Supprimer les balises inutiles (scripts, styles, nav…)
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Extraire le texte en joingnant les strings nettoyées
        clean_text = " ".join(soup.stripped_strings)

        # Limiter la taille pour éviter des matrices trop grandes
        clean_text = clean_text[:max_chars]

        print(f"[INFO] Texte extrait : {len(clean_text)} caractères")
        return clean_text

    except Exception as e:
        print(f"[ERREUR] Impossible de récupérer la page : {e}")
        return ""


def save_text(text: str, filepath: str) -> None:
    """Sauvegarde le texte dans un fichier .txt."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Texte sauvegardé dans '{filepath}'")


# ======================================================================
# PROGRAMME PRINCIPAL – à exécuter directement pour tester
# ======================================================================
if __name__ == "__main__":
    # Source 1 : article Wikipedia (texte d'entraînement)
    URL_TRAIN = "https://en.wikipedia.org/wiki/Rabat"
    # Source 2 : article Wikipedia différent (texte de test)
    URL_TEST  = "https://en.wikipedia.org/wiki/Casablanca"

    # Récupération
    text_train = fetch_text_from_url(URL_TRAIN)
    text_test  = fetch_text_from_url(URL_TEST)

    # Sauvegarde locale pour les étapes suivantes
    if text_train:
        save_text(text_train, "train_raw.txt")
    if text_test:
        save_text(text_test, "test_raw.txt")

    # Aperçu
    print("\n--- Aperçu du texte d'entraînement (500 premiers caractères) ---")
    print(text_train[:500])
