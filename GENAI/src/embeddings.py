"""Embedding model + FAISS index creation / loading.

Embeddings turn text into vectors of numbers; texts with similar meaning get vectors
that are close together. FAISS is a library that finds the closest vectors quickly.

Every saved index has a small `manifest.json` next to it that records WHAT it was built from
(the CSV fingerprint, the embedding model and its settings). The saved index is reused only if
the manifest matches the current situation - so editing a CSV or changing the embedding model
can never silently reuse old, incompatible vectors.
"""
import json
from pathlib import Path

from .data_loader import CSV_FILES, build_documents, csv_path, load_catalog
from .utils import DATA_DIR, EMBEDDING_MODEL_NAME, INDEX_DIR, file_fingerprint

MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1          # bump when the way documents/indexes are built changes
CHUNK_SIZE = 800
NORMALIZE_EMBEDDINGS = True   # needed for the squared-L2 -> cosine conversion


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Create the sentence-transformers embedding model (downloaded on first use, ~90 MB).

    Vectors are normalised so FAISS L2 distance can be converted to cosine similarity.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older LangChain installs
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )


def embedding_model_name(embeddings) -> str:
    """Name of the embedding model behind an embeddings object (falls back to the default)."""
    return str(getattr(embeddings, "model_name", None) or EMBEDDING_MODEL_NAME)


def dataset_fingerprint(data_dir: Path = DATA_DIR) -> str:
    """One fingerprint for both CSV files. Used as a Streamlit cache key, so editing a CSV
    while the app is running makes the app rebuild its indexes on the next interaction."""
    parts = []
    for kind in CSV_FILES:
        path = csv_path(kind, data_dir)
        parts.append(file_fingerprint(path) if path.exists() else "missing")
    return "-".join(parts)


def expected_manifest(kind: str, embeddings, data_dir: Path = DATA_DIR) -> dict:
    """What a saved index MUST have been built from to be reused now."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": kind,
        "csv_fingerprint": file_fingerprint(csv_path(kind, data_dir)),
        "embedding_model": embedding_model_name(embeddings),
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "chunk_size": CHUNK_SIZE,
    }


def read_manifest(folder: Path):
    """Return the saved manifest dict, or None if it is missing or unreadable."""
    try:
        data = json.loads((Path(folder) / MANIFEST_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def index_is_current(folder: Path, expected: dict) -> bool:
    """True only if the index files exist AND the manifest equals what we expect now."""
    folder = Path(folder)
    if not (folder / "index.faiss").exists() or not (folder / "index.pkl").exists():
        return False
    return read_manifest(folder) == expected


def build_or_load_index(kind: str, embeddings, data_dir: Path = DATA_DIR,
                        index_dir: Path = INDEX_DIR, force_rebuild: bool = False):
    """Return (faiss_store, dataframe) for 'movies' or 'books'.

    The saved index is reused if its manifest matches (same CSV content, same embedding model
    and settings), so embeddings are not recomputed on every run. Use force_rebuild=True to rebuild.
    """
    from langchain_community.vectorstores import FAISS

    df = load_catalog(kind, data_dir)
    folder = Path(index_dir) / kind
    manifest = expected_manifest(kind, embeddings, data_dir)

    if not force_rebuild and index_is_current(folder, manifest):
        try:
            # allow_dangerous_deserialization is OK here: we load our own local file.
            store = FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)
            return store, df
        except Exception:
            pass  # corrupt index -> rebuild below

    docs = build_documents(df, kind, CHUNK_SIZE)
    store = FAISS.from_documents(docs, embeddings)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST_FILE).unlink(missing_ok=True)      # never leave a manifest for a half-written index
    store.save_local(str(folder))
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store, df
