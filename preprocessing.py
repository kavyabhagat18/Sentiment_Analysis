"""
preprocessing.py
-----------------
Text cleaning utilities for the Twitter US Airline Sentiment project.

Pipeline:
    1. Lowercase the text
    2. Remove URLs, @mentions, the leading '#' of hashtags, HTML entities, and
       non-alphabetic characters
    3. Tokenize with NLTK's word_tokenize
    4. Remove stopwords (NLTK English stopword list), keeping a few negation
       words ('no', 'not', 'nor') since they flip sentiment polarity
    5. Lemmatize each remaining token with WordNetLemmatizer (verb + noun pass)

The same `clean_text` function is used both for training the model and for
scoring new text typed into the Streamlit app, so behaviour is guaranteed to
be identical between training and inference.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources():
    """Download required NLTK corpora/models if they aren't already present.

    Safe to call every time the module is imported - NLTK checks locally
    first and skips the download when the resource already exists.
    """
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg_id, quiet=True)


ensure_nltk_resources()

_lemmatizer = WordNetLemmatizer()

# Standard English stopword list, minus negation terms which carry
# sentiment-relevant meaning (e.g. "not delayed" vs "delayed").
_STOPWORDS = set(stopwords.words("english")) - {"no", "not", "nor"}

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#")
_HTML_ENTITY_RE = re.compile(r"&\w+;")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean and normalise a single raw text string (e.g. a tweet).

    Returns a space-joined string of cleaned, lemmatized tokens, ready to be
    fed into a TF-IDF vectorizer.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)          # drop @airline mentions
    text = _HASHTAG_SYMBOL_RE.sub(" ", text)   # keep hashtag word, drop '#'
    text = _HTML_ENTITY_RE.sub(" ", text)      # &amp; etc.
    text = _NON_ALPHA_RE.sub(" ", text)        # strip digits/punctuation
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        if tok in _STOPWORDS or len(tok) <= 1:
            continue
        # Lemmatize as a verb first (handles "flying" -> "fly"), then as a
        # noun (handles "flights" -> "flight"); WordNetLemmatizer only
        # changes the word if the requested POS produces a real lemma.
        lemma = _lemmatizer.lemmatize(tok, pos="v")
        lemma = _lemmatizer.lemmatize(lemma, pos="n")
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    sample = "@VirginAmerica it's really aggressive to blast obnoxious entertainment in your guests' faces & they have little recourse #flying"
    print("RAW   :", sample)
    print("CLEAN :", clean_text(sample))