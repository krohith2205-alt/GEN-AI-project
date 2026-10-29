"""Semantic retrieval, filtering and ranking."""
from dataclasses import dataclass, field

from .data_loader import load_catalog
from .utils import DATA_DIR, INDEX_DIR, format_rating, tokenize

MIN_SIMILARITY = 0.15  # below this a result is considered "not a real match"
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


def build_search_query(user_query: str, mood: str = "Any", favorites: str = "") -> str:
    """Combine the typed request with optional mood and favourite titles into one search text."""
    parts = [user_query]
    if mood and mood != "Any":
        parts.append(f"Mood: {mood}")
    if favorites.strip():
        parts.append(f"Similar to: {favorites.strip()}")
    return ". ".join(parts)


def fallback_reason(rec: Recommendation, query: str) -> str:
    """Explanation built only from catalog metadata (used when the LLM is unavailable)."""
    r = rec.record
    searchable = " ".join(str(r.get(k, "")) for k in ("title", "genre", "keywords", "description"))
    matched = sorted((tokenize(query) & tokenize(searchable)) - STOP_WORDS)
    kind_word = "movie" if rec.kind == "movies" else "book"
    text = f"This {str(r.get('genre', '')).lower()} {kind_word} is a close semantic match to your request"
    if matched:
        text += f" (shared themes: {', '.join(matched[:5])})"
    text += "."
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
        if not query or not query.strip():
            raise ValueError("Empty search query.")
        top_k = max(1, min(int(top_k), 10))
        filters = filters or {}
        kinds = {"Movies": ["movies"], "Books": ["books"]}.get(category, list(KINDS))
        return {k: self.search_kind(k, query, top_k, filters.get(k, Filters())) for k in kinds}
