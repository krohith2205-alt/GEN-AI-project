"""Semantic retrieval, filtering and ranking."""
from dataclasses import dataclass, field

from .data_loader import load_catalog
from .utils import DATA_DIR, INDEX_DIR, format_rating, tokenize

# Semantic-similarity bands (cosine similarity of MiniLM embeddings, 0..1).
# MIN_SIMILARITY is the cut-off for showing a result at all. All three numbers are starting
# values: run `python evaluation/threshold_report.py` (real model) and adjust them to the results.
MIN_SIMILARITY = 0.15   # below this a result is considered "not a real match"
MODERATE_SIMILARITY = 0.30
STRONG_SIMILARITY = 0.45
KINDS = ("movies", "books")
STOP_WORDS = {"a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "me", "i", "want",
              "recommend", "suggest", "movies", "movie", "books", "book", "that", "are", "is", "some",
              "good", "about", "suitable", "like", "novels", "my", "please"}


@dataclass
class Filters:
    """User-selected filters for one catalog. Empty list / 0 means 'no filter'."""
    genres: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    min_rating: float = 0.0


@dataclass
class Recommendation:
    kind: str                 # 'movies' or 'books'
    record: dict              # full catalog record (from CSV metadata, never from the LLM)
    similarity: float         # 0..1, higher = closer to the query
    reason: str = ""          # explanation shown to the user
    reason_source: str = "catalog"   # 'llm' or 'catalog'

    @property
    def id(self) -> str:
        return str(self.record["id"])


def passes_filters(record: dict, filters: Filters) -> bool:
    """Apply genre / language / minimum-rating filters to one record."""
    if filters.genres and record.get("genre") not in filters.genres:
        return False
    if filters.languages and record.get("language") not in filters.languages:
        return False
    if filters.min_rating:
        rating = record.get("rating")
        if rating is None or rating < filters.min_rating:
            return False
    return True


def distance_to_similarity(distance: float) -> float:
    """FAISS returns squared L2 distance. For unit-length vectors cosine = 1 - d/2."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


@dataclass(frozen=True)
class Preferences:
    """Everything the user told us. The SAME object is used for retrieval and for the
    explanation writer, so the LLM sees exactly the preferences that produced the results."""
    query: str
    mood: str = "Any"
    favorites: str = ""

    @property
    def has_mood(self) -> bool:
        return bool(self.mood) and self.mood != "Any"

    @property
    def favorite_list(self) -> list:
        return [f.strip() for f in (self.favorites or "").split(",") if f.strip()]

    def search_text(self) -> str:
        """One text used to search the vector index."""
        parts = [self.query]
        if self.has_mood:
            parts.append(f"Mood: {self.mood}")
        if self.favorite_list:
            parts.append(f"Similar to: {', '.join(self.favorite_list)}")
        return ". ".join(parts)

    def prompt_block(self) -> str:
        """The same preferences, labelled, for the explanation prompt."""
        lines = [f"<user_request>{self.query}</user_request>"]
        lines.append(f"<mood>{self.mood if self.has_mood else 'not specified'}</mood>")
        lines.append(f"<favourite_titles>{', '.join(self.favorite_list) or 'not specified'}</favourite_titles>")
        return "\n".join(lines)


def as_preferences(value) -> Preferences:
    """Accept a Preferences object or a plain query string."""
    return value if isinstance(value, Preferences) else Preferences(query=str(value))


def build_search_query(user_query: str, mood: str = "Any", favorites: str = "") -> str:
    """Combine the typed request with optional mood and favourite titles into one search text."""
    return Preferences(user_query, mood, favorites).search_text()


def match_strength(similarity: float) -> str:
    """'strong', 'moderate' or 'weak' - used for honest wording of explanations."""
    if similarity >= STRONG_SIMILARITY:
        return "strong"
    if similarity >= MODERATE_SIMILARITY:
        return "moderate"
    return "weak"


def similarity_label(similarity: float) -> str:
    """Text for the card. It is semantic similarity, NOT a chance that the user will enjoy it."""
    return f"Semantic similarity: {similarity:.0%}"


def fallback_reason(rec: Recommendation, preferences) -> str:
    """Explanation built only from catalog metadata (used when the LLM is unavailable).

    The wording depends on how strong the match really is.
    """
    prefs = as_preferences(preferences)
    r = rec.record
    searchable = " ".join(str(r.get(k, "")) for k in ("title", "genre", "keywords", "description"))
    wanted = tokenize(prefs.search_text())
    matched = sorted((wanted & tokenize(searchable)) - STOP_WORDS - {"mood", "similar"})
    shared = ", ".join(matched[:5])
    kind_word = "movie" if rec.kind == "movies" else "book"
    genre = str(r.get("genre", "")).lower()
    strength = match_strength(rec.similarity)

    if strength == "strong":
        text = f"This {genre} {kind_word} is a strong semantic match to your request"
        text += f" (shared themes: {shared})." if shared else "."
    elif strength == "moderate":
        text = f"This {genre} {kind_word} is a moderate match to your request"
        text += f", sharing the themes: {shared}." if shared else ", based on its overall description."
    else:
        text = f"This {genre} {kind_word} is only a weak match, so treat it as a tentative suggestion"
        text += (f"; the only shared themes are: {shared}." if shared
                 else "; no specific shared theme was found in its catalog entry.")
    if r.get("keywords"):
        text += f" Catalog keywords: {r['keywords']}."
    return text


class Recommender:
    """Holds one FAISS store + dataframe per catalog and answers search requests."""

    def __init__(self, stores: dict, catalogs: dict):
        self.stores = stores
        self.catalogs = catalogs

    @classmethod
    def from_embeddings(cls, embeddings, data_dir=DATA_DIR, index_dir=INDEX_DIR, force_rebuild=False):
        from .embeddings import build_or_load_index
        stores, catalogs = {}, {}
        for kind in KINDS:
            stores[kind], catalogs[kind] = build_or_load_index(kind, embeddings, data_dir, index_dir, force_rebuild)
        return cls(stores, catalogs)

    # ---- options for the sidebar -------------------------------------------------
    def genres(self, kind: str) -> list:
        return sorted(g for g in self.catalogs[kind]["genre"].unique() if g)

    def unknown_titles(self, favorites) -> list:
        """Favourite titles that are not in the catalog (so we can tell the user honestly)."""
        known = {str(t).strip().lower() for df in self.catalogs.values() for t in df["title"]}
        names = favorites.favorite_list if isinstance(favorites, Preferences) else \
            [f.strip() for f in str(favorites).split(",") if f.strip()]
        return [n for n in names if n.lower() not in known]

    def languages(self) -> list:
        langs = set()
        for df in self.catalogs.values():
            langs.update(l for l in df["language"].unique() if l)
        return sorted(langs)

    # ---- main search -------------------------------------------------------------
    def search_kind(self, kind: str, query: str, top_k: int, filters: Filters) -> list:
        store, df = self.stores[kind], self.catalogs[kind]
        fetch = max(top_k * 10, len(df))          # small catalog: look at every record, then filter
        hits = store.similarity_search_with_score(query, k=fetch)
        best = {}                                  # keep the best chunk of every record
        for doc, distance in hits:
            rec = dict(doc.metadata)
            sim = distance_to_similarity(distance)
            if rec["id"] not in best or sim > best[rec["id"]].similarity:
                best[rec["id"]] = Recommendation(kind, rec, sim)
        results = [r for r in best.values()
                   if r.similarity >= MIN_SIMILARITY and passes_filters(r.record, filters)]
        results.sort(key=lambda r: (-r.similarity, -(r.record.get("rating") or 0)))
        return results[:top_k]

    def search(self, query: str, category: str = "Both", top_k: int = 5, filters: dict | None = None) -> dict:
        """Return {'movies': [...], 'books': [...]} for the chosen category."""
        if isinstance(query, Preferences):
            query = query.search_text()
        if not query or not query.strip():
            raise ValueError("Empty search query.")
        top_k = max(1, min(int(top_k), 10))
        filters = filters or {}
        kinds = {"Movies": ["movies"], "Books": ["books"]}.get(category, list(KINDS))
        return {k: self.search_kind(k, query, top_k, filters.get(k, Filters())) for k in kinds}
