"""
Utility helpers for the content-aware router.

All functions are pure (no I/O) and dependency-free so they can be unit-tested
without standing up any LLM infrastructure.

Matching pipeline
-----------------
tokenize (lowercase + punctuation removal + stop-word filter + light stemming)
  → build_bm25_index (precomputes per-term IDF and average document length)
  → bm25_score       (Okapi BM25 — handles term saturation and length normalisation)

Stemming ensures morphological variants ("reason" / "reasoning",
"function" / "functions", "debug" / "debugging") map to the same stem so they
match across descriptions and prompts even when written in different forms.
"""
import math
import re
import string
from typing import Dict, List, Optional, Tuple

# Okapi BM25 hyper-parameters (standard defaults)
_BM25_K1 = 1.5  # term-frequency saturation
_BM25_B = 0.75  # length normalisation


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "i", "you", "he",
        "she", "it", "we", "they", "this", "that", "these", "those", "as",
        "if", "then", "than", "so", "not", "no", "nor", "yet", "both",
        "either", "neither", "about", "into", "through", "during", "including",
        "until", "while", "of", "about", "against", "between", "into",
        "through", "such", "any", "more", "also", "use", "using", "used",
        "help", "based", "related", "request", "task", "tasks",
    }
)


# ---------------------------------------------------------------------------
# Light stemmer
# ---------------------------------------------------------------------------

def stem(word: str) -> str:
    """
    Light suffix-stripping stemmer for English.

    Reduces the most common inflected forms to a shared base so that
    morphological variants produce the same index token:

        reasoning   → reason    (-ing, 6+ chars remaining)
        debugging   → debug     (-ing + de-double trailing consonant)
        implementing→ implement  (-ing)
        functions   → function   (-s)
        algorithms  → algorithm  (-s)
        processes   → process    (-es)
        stories     → story      (-ies → y)

    Deliberately conservative: only strips -ing, -ed, -ies, -es, and -s so
    that common technical terms ("function", "computation", "section") are
    not truncated to unrecognisable stems by over-aggressive -tion/-ation
    rules.

    Rules are applied in priority order; the first matching rule wins.
    Minimum word-length guards prevent over-truncation on short words.
    """
    n = len(word)

    # -ing → strip, then de-double trailing consonant
    # reasoning→reason, debugging→debugg→debug, implementing→implement
    if n > 5 and word.endswith("ing"):
        candidate = word[:-3]
        if (
            len(candidate) >= 2
            and candidate[-1] == candidate[-2]
            and candidate[-1] not in "aeiou"
        ):
            candidate = candidate[:-1]
        return candidate

    # -ed → strip, then de-double (only for longer words)
    if n > 5 and word.endswith("ed"):
        candidate = word[:-2]
        if (
            len(candidate) >= 2
            and candidate[-1] == candidate[-2]
            and candidate[-1] not in "aeiou"
        ):
            candidate = candidate[:-1]
        return candidate

    # -ies → y (entries→entry, stories→story)
    if n > 4 and word.endswith("ies"):
        return word[:-3] + "y"

    # -es → strip (processes→process, classes→class, accesses→access)
    if n > 4 and word.endswith("es"):
        return word[:-2]

    # -s → strip (functions→function, algorithms→algorithm)
    # Skip words ending in -ss (class, process, etc.) to avoid truncation
    if n > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Normalise *text* to a list of stemmed tokens.

    Steps:
      1. Lowercase
      2. Replace punctuation with spaces
      3. Split on whitespace
      4. Drop stop words and single-character tokens
      5. Apply light stemming so inflected forms share a common base
    """
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    raw = text.split()
    return [
        stem(t)
        for t in raw
        if t and t not in _STOP_WORDS and len(t) > 1
    ]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(
    corpus: List[str],
) -> Tuple[List[List[str]], Dict[str, float], float]:
    """
    Build an Okapi BM25 index for a list of documents.

    Args:
        corpus: list of raw text strings (preference descriptions).

    Returns:
        tokenized_corpus : stemmed token list per document
        idf_weights      : per-term IDF values (Robertson–Sparck Jones, always ≥ 0)
        avgdl            : average document length in tokens
    """
    tokenized: List[List[str]] = [tokenize(doc) for doc in corpus]
    n = len(tokenized)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(n, 1)

    # Document frequency
    df: Dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — always positive
    idf: Dict[str, float] = {
        t: math.log((n - count + 0.5) / (count + 0.5) + 1)
        for t, count in df.items()
    }

    return tokenized, idf, avgdl


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf: Dict[str, float],
    avgdl: float,
) -> float:
    """
    Okapi BM25 relevance score for a query against a single document.

    Args:
        query_tokens : stemmed prompt tokens.
        doc_tokens   : stemmed description tokens (from the BM25 index).
        idf          : shared IDF weights built by build_bm25_index().
        avgdl        : average document length from build_bm25_index().

    Returns:
        Non-negative float; higher means more relevant.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    dl = len(doc_tokens)
    doc_freq: Dict[str, int] = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1

    score = 0.0
    for term in set(query_tokens):
        f = doc_freq.get(term, 0)
        if f == 0:
            continue
        term_idf = idf.get(term, 0.0)
        if term_idf <= 0.0:
            continue
        # BM25 term-frequency component with length normalisation
        length_norm = 1 - _BM25_B + _BM25_B * dl / max(avgdl, 1)
        tf_component = f * (_BM25_K1 + 1) / (f + _BM25_K1 * length_norm)
        score += term_idf * tf_component

    return score


# ---------------------------------------------------------------------------
# Dense-vector cosine similarity (used by embedding_similarity classifier)
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------

def extract_prompt_text(
    messages: Optional[List[Dict]],
) -> Tuple[str, Optional[str]]:
    """
    Extract the last user message and the last system prompt from a messages list.

    Returns:
        (user_text, system_text) — system_text may be None if no system message found.
    """
    if not messages:
        return "", None

    user_text: Optional[str] = None
    system_text: Optional[str] = None

    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Content-part format: [{"type": "text", "text": "..."}]
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = " ".join(parts).strip()
        if isinstance(content, str) and content:
            if role == "user" and user_text is None:
                user_text = content
            elif role == "system" and system_text is None:
                system_text = content
        if user_text is not None and system_text is not None:
            break

    return user_text or "", system_text
