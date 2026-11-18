"""Regression tests for index persistence and invalidation (manifest, CSV fingerprint, model config)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.data_loader import CSV_FILES
from src.embeddings import (MANIFEST_FILE, build_or_load_index, dataset_fingerprint, expected_manifest,
                            index_is_current, read_manifest)
from src.utils import DATA_DIR
from tests.helpers import embed

try:
    import faiss  # noqa: F401
    from langchain_community.vectorstores import FAISS  # noqa: F401
    from langchain_core.embeddings import Embeddings
    HAVE_FAISS = True
except ImportError:      # pragma: no cover
    HAVE_FAISS, Embeddings = False, object


class FakeEmbeddings(Embeddings):
    """Offline embeddings that count how many texts were embedded (to prove an index was reused)."""
    def __init__(self, model_name="fake/model-a"):
        self.model_name = model_name
        self.documents_embedded = 0

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return [embed(t).tolist() for t in texts]

    def embed_query(self, text):
        return embed(text).tolist()


def copy_data(tmp: Path) -> Path:
    data = tmp / "data"
    data.mkdir()
    for name in CSV_FILES.values():
        shutil.copy(DATA_DIR / name, data / name)
    return data


class TestManifestLogic(unittest.TestCase):
    """Pure logic - no FAISS needed."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = copy_data(self.tmp)
        self.folder = self.tmp / "index" / "movies"
        self.folder.mkdir(parents=True)
        (self.folder / "index.faiss").write_bytes(b"x")
        (self.folder / "index.pkl").write_bytes(b"x")
        self.emb = FakeEmbeddings()
        self.manifest = expected_manifest("movies", self.emb, self.data)
        (self.folder / MANIFEST_FILE).write_text(json.dumps(self.manifest))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matching_manifest_is_current(self):
        self.assertTrue(index_is_current(self.folder, self.manifest))

    def test_editing_the_csv_invalidates(self):
        with open(self.data / "movies.csv", "a", encoding="utf-8") as f:
            f.write("M999,New,Drama,d,English,2020,4.0,k\n")
        new = expected_manifest("movies", self.emb, self.data)
        self.assertNotEqual(new["csv_fingerprint"], self.manifest["csv_fingerprint"])
        self.assertFalse(index_is_current(self.folder, new))

    def test_changing_the_embedding_model_invalidates(self):
        other = expected_manifest("movies", FakeEmbeddings("fake/model-b"), self.data)
        self.assertEqual(other["embedding_model"], "fake/model-b")
        self.assertFalse(index_is_current(self.folder, other))

    def test_changing_normalisation_or_chunking_invalidates(self):
        for key, value in (("normalize_embeddings", False), ("chunk_size", 100), ("manifest_version", 99)):
            self.assertFalse(index_is_current(self.folder, {**self.manifest, key: value}), key)

    def test_missing_or_corrupt_manifest_invalidates(self):
        (self.folder / MANIFEST_FILE).write_text("{not json")
        self.assertIsNone(read_manifest(self.folder))
        self.assertFalse(index_is_current(self.folder, self.manifest))
        (self.folder / MANIFEST_FILE).unlink()
        self.assertFalse(index_is_current(self.folder, self.manifest))

    def test_missing_index_files_invalidate(self):
        (self.folder / "index.pkl").unlink()
        self.assertFalse(index_is_current(self.folder, self.manifest))

    def test_dataset_fingerprint_changes_when_either_csv_changes(self):
        """This value is part of the Streamlit cache key in app.py."""
        before = dataset_fingerprint(self.data)
        self.assertEqual(before, dataset_fingerprint(self.data))
        with open(self.data / "books.csv", "a", encoding="utf-8") as f:
            f.write("B999,New,Au,Fiction,d,English,2020,4.0,k\n")
        self.assertNotEqual(before, dataset_fingerprint(self.data))

    def test_dataset_fingerprint_survives_missing_csv(self):
        (self.data / "books.csv").unlink()
        self.assertIn("missing", dataset_fingerprint(self.data))


@unittest.skipUnless(HAVE_FAISS, "faiss / langchain-community not installed")
class TestIndexWithRealFaiss(unittest.TestCase):
    """End-to-end with a real FAISS index and offline embeddings."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = copy_data(self.tmp)
        self.index = self.tmp / "index"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, emb, **kw):
        return build_or_load_index("movies", emb, self.data, self.index, **kw)

    def test_second_start_reuses_saved_index(self):
        first = FakeEmbeddings()
        self.build(first)
        self.assertGreater(first.documents_embedded, 0)
        second = FakeEmbeddings()
        store, df = self.build(second)
        self.assertEqual(second.documents_embedded, 0)            # loaded from disk, nothing re-embedded
        self.assertEqual(len(df), 53)
        self.assertTrue(store.similarity_search("dreams heist", k=1))

    def test_csv_edit_triggers_rebuild(self):
        self.build(FakeEmbeddings())
        with open(self.data / "movies.csv", "a", encoding="utf-8") as f:
            f.write("M999,Brand New Film,Drama,A new story about rebuilding.,English,2020,4.0,rebuild\n")
        emb = FakeEmbeddings()
        store, df = self.build(emb)
        self.assertGreater(emb.documents_embedded, 0)
        self.assertEqual(len(df), 54)
        titles = {d.metadata["title"] for d in store.similarity_search("rebuilding", k=60)}
        self.assertIn("Brand New Film", titles)

    def test_model_change_triggers_rebuild(self):
        self.build(FakeEmbeddings("fake/model-a"))
        emb = FakeEmbeddings("fake/model-b")
        self.build(emb)
        self.assertGreater(emb.documents_embedded, 0)             # NOT silently reusing model-a vectors
        self.assertEqual(read_manifest(self.index / "movies")["embedding_model"], "fake/model-b")

    def test_force_rebuild(self):
        self.build(FakeEmbeddings())
        emb = FakeEmbeddings()
        self.build(emb, force_rebuild=True)
        self.assertGreater(emb.documents_embedded, 0)

    def test_old_style_index_without_manifest_is_rebuilt(self):
        self.build(FakeEmbeddings())
        (self.index / "movies" / MANIFEST_FILE).unlink()
        (self.index / "movies" / "source.md5").write_text("old-format-stamp")
        emb = FakeEmbeddings()
        self.build(emb)
        self.assertGreater(emb.documents_embedded, 0)

    def test_corrupt_index_file_is_rebuilt(self):
        self.build(FakeEmbeddings())
        (self.index / "movies" / "index.faiss").write_bytes(b"garbage")
        emb = FakeEmbeddings()
        store, _ = self.build(emb)
        self.assertGreater(emb.documents_embedded, 0)
        self.assertTrue(store.similarity_search("dreams", k=1))


if __name__ == "__main__":
    unittest.main()
