"""Embedding model + FAISS index creation / loading.

Embeddings turn text into vectors of numbers; texts with similar meaning get vectors
that are close together. FAISS is a library that finds the closest vectors quickly.
"""
from pathlib import Path

from .data_loader import build_documents, csv_path, load_catalog
from .utils import DATA_DIR, EMBEDDING_MODEL_NAME, INDEX_DIR, file_fingerprint

FINGERPRINT_FILE = "source.md5"


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
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_index(kind: str, embeddings, data_dir: Path = DATA_DIR,
                        index_dir: Path = INDEX_DIR, force_rebuild: bool = False):
    """Return (faiss_store, dataframe) for 'movies' or 'books'.

    The saved index is reused if the CSV has not changed (checked with an MD5 hash),
    so embeddings are not recomputed on every run. Use force_rebuild=True to rebuild.
    """
    from langchain_community.vectorstores import FAISS

    df = load_catalog(kind, data_dir)
    folder = Path(index_dir) / kind
    fingerprint = file_fingerprint(csv_path(kind, data_dir))
    stamp = folder / FINGERPRINT_FILE

    if not force_rebuild and stamp.exists() and (folder / "index.faiss").exists():
        if stamp.read_text().strip() == fingerprint:
            try:
                # allow_dangerous_deserialization is OK here: we load our own local file.
                store = FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)
                return store, df
            except Exception:
                pass  # corrupt index -> rebuild below

    docs = build_documents(df, kind)
    store = FAISS.from_documents(docs, embeddings)
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))
    stamp.write_text(fingerprint)
    return store, df
