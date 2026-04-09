"""
scripts/setup_nlp_models.py
-----------------------------
Run this ONCE after `pip install -r requirements.txt` to download all
NLP model assets required by the project.

Usage:
    python scripts/setup_nlp_models.py
"""

import subprocess
import sys

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def download_nltk_resources() -> None:
    banner("Downloading NLTK resources …")
    import nltk
    resources = [
        ("tokenizers/punkt",         "punkt"),
        ("corpora/stopwords",        "stopwords"),
        ("corpora/wordnet",          "wordnet"),
        ("corpora/omw-1.4",          "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"  ✓ {name} already present")
        except LookupError:
            print(f"  ↓ Downloading {name} …")
            nltk.download(name, quiet=False)
    print("  NLTK resources ready.")


def download_spacy_model() -> None:
    banner("Downloading spaCy model: en_core_web_sm …")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("  ✓ spaCy model ready.")
    else:
        print("  ✗ spaCy model download failed — check your internet connection.")


def prefetch_sentence_transformer() -> None:
    banner("Pre-fetching SentenceTransformer: all-MiniLM-L6-v2 …")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Quick sanity check
        vec = model.encode(["Hello world"], convert_to_numpy=True)
        assert vec.shape == (1, 384), "Unexpected embedding shape"
        print("  ✓ SentenceTransformer model ready (dim=384).")
    except ImportError:
        print("  ✗ sentence-transformers not installed. Run: pip install sentence-transformers")
    except Exception as exc:
        print(f"  ✗ Error loading SentenceTransformer: {exc}")


if __name__ == "__main__":
    print("\n🚀 AI Resume Screening System — NLP Model Setup")
    download_nltk_resources()
    download_spacy_model()
    prefetch_sentence_transformer()
    print("\n✅ All setup complete. You can now run the project.\n")
