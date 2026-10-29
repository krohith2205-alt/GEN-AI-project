"""Test helpers: a tiny offline stand-in for the embedding model + FAISS store.

Real embeddings need a ~90 MB model download, so unit tests use hashed bag-of-words
vectors instead. This tests OUR logic (filters, ranking, fallbacks), not the quality of MiniLM.
"""
import hashlib
import re

import numpy as np

from src.data_loader import build_documents, load_catalog
from src.recommender import Recommender

DIM = 512


def embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeStore:
    """Mimics FAISS.similarity_search_with_score (squared L2 distance, smaller = closer)."""

    def __init__(self, docs):
        self.docs = docs
        self.vectors = np.array([embed(d.page_content) for d in docs])

    def similarity_search_with_score(self, query, k=4):
        q = embed(query)
        dists = ((self.vectors - q) ** 2).sum(axis=1)
        order = np.argsort(dists)[:k]
        return [(self.docs[i], float(dists[i])) for i in order]


def make_recommender() -> Recommender:
    stores, catalogs = {}, {}
    for kind in ("movies", "books"):
        catalogs[kind] = load_catalog(kind)
        stores[kind] = FakeStore(build_documents(catalogs[kind], kind))
    return Recommender(stores, catalogs)
