import tempfile
import unittest
from pathlib import Path

from src.data_loader import (BOOK_COLUMNS, MOVIE_COLUMNS, DatasetError, build_documents, load_catalog)


class TestDataLoader(unittest.TestCase):
    def test_movies_loaded_with_required_columns(self):
        df = load_catalog("movies")
        self.assertGreaterEqual(len(df), 50)
        for col in MOVIE_COLUMNS:
            self.assertIn(col, df.columns)

    def test_books_loaded_with_required_columns(self):
        df = load_catalog("books")
        self.assertGreaterEqual(len(df), 50)
        for col in BOOK_COLUMNS:
            self.assertIn(col, df.columns)

    def test_ids_are_unique(self):
        for kind in ("movies", "books"):
            df = load_catalog(kind)
            self.assertEqual(df["id"].nunique(), len(df))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetError):
                load_catalog("movies", Path(tmp))

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "movies.csv").write_text("id,title\nM1,Test\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_catalog("movies", Path(tmp))

    def test_missing_values_are_handled(self):
        header = ",".join(MOVIE_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "movies.csv").write_text(f"{header}\nM1,Test Movie,Drama,,English,notayear,,\n", encoding="utf-8")
            df = load_catalog("movies", Path(tmp))
            self.assertEqual(df.loc[0, "description"], "")
            self.assertTrue(df["rating"].isna().iloc[0])

    def test_unknown_kind_raises(self):
        with self.assertRaises(DatasetError):
            load_catalog("music")

    def test_documents_keep_metadata(self):
        df = load_catalog("books")
        docs = build_documents(df, "books")
        self.assertGreaterEqual(len(docs), len(df))
        self.assertIn("title", docs[0].metadata)
        self.assertEqual(docs[0].metadata["kind"], "books")
        self.assertIn("Author:", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
