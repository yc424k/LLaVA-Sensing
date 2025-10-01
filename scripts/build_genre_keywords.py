#!/usr/bin/env python3
"""Build per-genre keyword lists from the Novel corpus using TF-IDF."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TF-IDF based keyword lists for each genre directory under Novel/",
    )
    parser.add_argument(
        "--novel-dir",
        type=Path,
        default=Path("Novel"),
        help="Path to the directory containing per-genre subdirectories of novels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Novel/genre_keywords.json"),
        help="Where to save the resulting JSON dictionary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=150,
        help="Number of top keywords to keep per genre.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Minimum number of documents a term must appear in to be kept.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.5,
        help="Ignore terms that appear in more than this fraction of documents (0-1).",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Use unigrams only (1), include bigrams (2), or include trigrams (3).",
    )
    parser.add_argument(
        "--scoring",
        choices=["tfidf", "log_odds"],
        default="tfidf",
        help="Keyword scoring method: TF-IDF mean or log-odds ratio against the rest.",
    )
    parser.add_argument(
        "--additional-stopwords",
        type=Path,
        default=None,
        help="Optional path to a text file listing extra stop words (one per line).",
    )
    parser.add_argument(
        "--strip-person-entities",
        action="store_true",
        help="Use spaCy named entity recognition to remove PERSON tokens before analysis.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name to load when stripping PERSON entities.",
    )
    parser.add_argument(
        "--spacy-n-process",
        type=int,
        default=2,
        help="Number of processes spaCy should use for entity stripping (>=1).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display progress bars (requires tqdm; falls back to plain iteration if unavailable).",
    )
    return parser.parse_args()


def load_additional_stopwords(stopwords_path: Path | None) -> list[str]:
    if not stopwords_path:
        return []
    if not stopwords_path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")
    return [line.strip() for line in stopwords_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def maybe_progress(
    iterable: Iterable[T],
    enabled: bool,
    description: str,
    total: int | None = None,
) -> Iterable[T]:
    if not enabled:
        return iterable

    try:
        from tqdm import tqdm
    except ImportError:
        print("[info] tqdm not found; progress output disabled.")
        return iterable

    return tqdm(iterable, desc=description, total=total)


def gather_documents(novel_dir: Path, show_progress: bool = False) -> tuple[list[str], list[str]]:
    if not novel_dir.exists():
        raise FileNotFoundError(f"Novel directory not found: {novel_dir}")

    doc_entries: list[tuple[Path, str]] = []
    for genre_dir in sorted(p for p in novel_dir.iterdir() if p.is_dir()):
        text_paths = sorted(genre_dir.glob("*.txt"))
        doc_entries.extend((text_path, genre_dir.name) for text_path in text_paths)

    if not doc_entries:
        raise ValueError(f"No .txt files found under {novel_dir}")

    corpus: list[str] = []
    labels: list[str] = []

    iterator = maybe_progress(
        doc_entries,
        enabled=show_progress,
        description="Reading texts",
        total=len(doc_entries),
    )

    for text_path, genre_name in iterator:
        corpus.append(text_path.read_text(encoding="utf-8", errors="ignore"))
        labels.append(genre_name)

    return corpus, labels


def build_stopword_list(extra_stopwords: list[str]) -> list[str]:
    combined: set[str] = set()

    combined.update(sklearn_text.ENGLISH_STOP_WORDS)

    try:
        from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS  # type: ignore

        combined.update(SPACY_STOPWORDS)
    except ImportError:
        pass

    try:
        from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS  # type: ignore

        combined.update(GENSIM_STOPWORDS)
    except ImportError:
        pass

    try:
        from nltk.corpus import stopwords as nltk_stopwords  # type: ignore

        try:
            combined.update(nltk_stopwords.words("english"))
        except LookupError:
            pass
    except ImportError:
        pass

    genre_specific = {
        "mr",
        "mrs",
        "miss",
        "ms",
        "sir",
        "madam",
        "madame",
        "lady",
        "lord",
        "colonel",
        "captain",
        "general",
        "reverend",
        "dr",
        "professor",
        "uncle",
        "aunt",
        "mister",
        "master",
        "mistress",
        "monsieur",
        "father",
        "mother",
        "papa",
        "mamma",
        "grandmother",
        "grandfather",
        "brother",
        "sister",
        "daughter",
        "son",
        "dear",
        "thy",
        "thou",
        "thee",
        "ye",
        "aint",
        "dont",
        "didnt",
        "doesnt",
        "wont",
        "cant",
    }
    combined.update(genre_specific)

    combined.update(word.lower() for word in extra_stopwords)

    return sorted({word for word in combined if word})


def filter_person_entities(
    texts: list[str],
    model_name: str,
    show_progress: bool = False,
    n_process: int = 1,
) -> list[str]:
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for --strip-person-entities; install it with `pip install spacy`."
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. Run `python -m spacy download {model_name}` and retry."
        ) from exc

    cleaned: list[str] = []
    iterator = nlp.pipe(texts, batch_size=32, n_process=max(1, n_process))
    iterator = maybe_progress(
        iterator,
        enabled=show_progress,
        description="Removing PERSON entities",
        total=len(texts),
    )

    for doc in iterator:
        tokens = [token.text for token in doc if token.ent_type_ != "PERSON"]
        cleaned.append(" ".join(tokens))
    return cleaned


def _build_vectorizer(
    scoring: str,
    min_df: int,
    max_df: float,
    max_ngram: int,
    stop_words: list[str],
):
    common_kwargs = dict(
        lowercase=True,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, max_ngram),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    if scoring == "tfidf":
        return TfidfVectorizer(**common_kwargs)
    return CountVectorizer(**common_kwargs)


def _score_tfidf(matrix, feature_names: np.ndarray, top_k: int):
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    order = mean_scores.argsort()[::-1]
    limit = min(top_k, len(order))
    return feature_names[order][:limit].tolist()


def _score_log_odds(
    matrix,
    feature_names: np.ndarray,
    top_k: int,
    genre_indices: dict[str, list[int]],
):
    # Use informative Dirichlet prior for stability
    alpha = 0.01
    total_counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocab_size = len(feature_names)
    genre_keywords: dict[str, list[str]] = {}

    for genre, indices in genre_indices.items():
        genre_matrix = matrix[indices]
        genre_counts = np.asarray(genre_matrix.sum(axis=0)).ravel()
        background_counts = total_counts - genre_counts

        genre_total = genre_counts.sum()
        background_total = background_counts.sum()

        if genre_total == 0 or background_total == 0:
            genre_keywords[genre] = []
            continue

        numerator = (genre_counts + alpha) / (genre_total + alpha * vocab_size)
        denominator = (background_counts + alpha) / (background_total + alpha * vocab_size)
        log_odds = np.log(numerator / denominator)
        variance = 1.0 / (genre_counts + alpha) + 1.0 / (background_counts + alpha)
        with np.errstate(divide="ignore"):
            z_scores = log_odds / np.sqrt(variance)

        order = np.argsort(z_scores)[::-1]
        limit = min(top_k, len(order))
        genre_keywords[genre] = feature_names[order][:limit].tolist()

    return genre_keywords


def compute_keywords(
    corpus: list[str],
    labels: list[str],
    top_k: int,
    min_df: int,
    max_df: float,
    max_ngram: int,
    scoring: str,
    stop_words: list[str],
) -> dict[str, list[str]]:
    vectorizer = _build_vectorizer(
        scoring=scoring,
        min_df=min_df,
        max_df=max_df,
        max_ngram=max_ngram,
        stop_words=stop_words,
    )
    matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())

    genre_indices: dict[str, list[int]] = defaultdict(list)
    for idx, genre in enumerate(labels):
        genre_indices[genre].append(idx)

    if scoring == "tfidf":
        return {
            genre: _score_tfidf(matrix[indices], feature_names, top_k)
            for genre, indices in genre_indices.items()
        }

    # scoring == "log_odds"
    return _score_log_odds(matrix, feature_names, top_k, genre_indices)


def main() -> None:
    args = parse_args()
    corpus, labels = gather_documents(args.novel_dir, show_progress=args.show_progress)
    extra_stopwords = load_additional_stopwords(args.additional_stopwords)
    stop_words = build_stopword_list(extra_stopwords)

    if args.strip_person_entities:
        corpus = filter_person_entities(
            corpus,
            model_name=args.spacy_model,
            show_progress=args.show_progress,
            n_process=args.spacy_n_process,
        )

    keywords = compute_keywords(
        corpus=corpus,
        labels=labels,
        top_k=args.top_k,
        min_df=args.min_df,
        max_df=args.max_df,
        max_ngram=args.max_ngram,
        scoring=args.scoring,
        stop_words=stop_words,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(keywords, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved keyword dictionary to {args.output} (genres: {', '.join(sorted(keywords))})")


if __name__ == "__main__":
    main()
