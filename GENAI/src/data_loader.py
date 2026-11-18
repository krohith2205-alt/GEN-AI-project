"""Load and validate the CSV catalogs, and convert rows into LangChain Documents."""
import math
from pathlib import Path

import pandas as pd

try:  # normal case: use LangChain's Document class
    from langchain_core.documents import Document
except ImportError:  # tiny stand-in so the data code can be tested without LangChain
    from dataclasses import dataclass, field

    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict = field(default_factory=dict)

from .utils import DATA_DIR, MAX_RATING, MAX_YEAR, MIN_RATING, MIN_YEAR

MOVIE_COLUMNS = ["id", "title", "genre", "description", "language", "release_year", "rating", "keywords"]
BOOK_COLUMNS = ["id", "title", "author", "genre", "description", "language", "publication_year", "rating", "keywords"]
REQUIRED_COLUMNS = {"movies": MOVIE_COLUMNS, "books": BOOK_COLUMNS}
CSV_FILES = {"movies": "movies.csv", "books": "books.csv"}
TEXT_COLUMNS = ["title", "genre", "description", "language", "keywords", "author"]


class DatasetError(Exception):
    """Raised when a dataset is missing, empty or has wrong columns."""


def csv_path(kind: str, data_dir: Path = DATA_DIR) -> Path:
    if kind not in CSV_FILES:
        raise DatasetError(f"Unknown catalog type: {kind!r}. Use 'movies' or 'books'.")
    return Path(data_dir) / CSV_FILES[kind]


def load_catalog(kind: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one CSV, check its columns, and clean missing values."""
    path = csv_path(kind, data_dir)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        # Read ids as text so "7" stays "7" and an empty id stays missing (never "nan").
        df = pd.read_csv(path, encoding="utf-8", dtype={"id": str})
    except Exception as exc:  # corrupt file, wrong encoding, etc.
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS[kind] if c not in df.columns]
    if missing:
        raise DatasetError(f"{path.name} is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise DatasetError(f"{path.name} has no records.")

    notes = []   # data-quality messages (shown in the app sidebar)

    # Record ids: reject missing / blank ids BEFORE converting to strings.
    before = len(df)
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(str).str.strip()
    df = df[df["id"] != ""]
    if len(df) < before:
        notes.append(f"{path.name}: {before - len(df)} row(s) without a record id were skipped.")

    # Text columns -> stripped strings
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Numbers: invalid or out-of-range values become missing (NaN), never a crash later.
    year_col = "release_year" if kind == "movies" else "publication_year"
    df[year_col], bad_years = _validated_numbers(df[year_col], MIN_YEAR[kind], MAX_YEAR, whole=True)
    df["rating"], bad_ratings = _validated_numbers(df["rating"], MIN_RATING, MAX_RATING)
    if bad_years:
        notes.append(f"{path.name}: {bad_years} invalid {year_col} value(s) ignored.")
    if bad_ratings:
        notes.append(f"{path.name}: {bad_ratings} invalid rating value(s) ignored (allowed {MIN_RATING:g}-{MAX_RATING:g}).")

    before = len(df)
    df = df[df["title"] != ""].drop_duplicates(subset="id").reset_index(drop=True)
    if len(df) < before:
        notes.append(f"{path.name}: {before - len(df)} row(s) with an empty title or duplicate id were skipped.")
    if df.empty:
        raise DatasetError(f"{path.name} has no valid records after cleaning.")
    df.attrs["data_warnings"] = notes
    return df


def _validated_numbers(series: pd.Series, low: float, high: float, whole: bool = False):
    """Convert to numbers; values that are non-numeric, not finite, out of range
    (or fractional when whole=True) become NaN. Returns (clean_series, number_invalid)."""
    raw = series.copy()
    nums = pd.to_numeric(raw, errors="coerce").astype(float)
    nums = nums.where(nums.map(math.isfinite))            # inf / -inf -> NaN
    ok = nums.between(low, high)
    if whole:
        ok &= nums == nums.round()
    clean = nums.where(ok)
    # "invalid" = there was some text in the CSV cell but it could not be used
    had_value = raw.notna() & (raw.astype(str).str.strip() != "")
    return clean, int((had_value & clean.isna()).sum())


def row_to_text(row: pd.Series, kind: str) -> str:
    """Turn one catalog row into the text that will be embedded (searched)."""
    parts = [f"Title: {row['title']}"]
    if kind == "books" and row.get("author"):
        parts.append(f"Author: {row['author']}")
    parts.append(f"Genre: {row['genre']}")
    if row.get("language"):
        parts.append(f"Language: {row['language']}")
    parts.append(f"Description: {row['description']}")
    if row.get("keywords"):
        parts.append(f"Keywords: {row['keywords']}")
    return ". ".join(parts)


def _clean_value(value):
    """Make a value safe to store in metadata (NaN -> None, numpy -> python)."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_documents(df: pd.DataFrame, kind: str, chunk_size: int = 800) -> list:
    """Create one Document per record (and split any very long ones).

    The full record is stored in `metadata`, so after retrieval we always know the
    real title, author, rating, etc. - the LLM never has to guess them.
    """
    docs = []
    for _, row in df.iterrows():
        metadata = {k: _clean_value(v) for k, v in row.items()}
        metadata["kind"] = kind
        docs.append(Document(page_content=row_to_text(row, kind), metadata=metadata))
    return split_documents(docs, chunk_size)


def split_documents(docs: list, chunk_size: int = 800) -> list:
    """Split long documents into chunks. Our records are short, so this rarely changes anything."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return docs  # without the splitter, keep documents whole (they are short)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return splitter.split_documents(docs)
