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

    def _load(self, kind, rows):
        header = ",".join(MOVIE_COLUMNS if kind == "movies" else BOOK_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"{kind}.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            return load_catalog(kind, Path(tmp))

    def test_invalid_years_become_missing(self):
        rows = ["M1,A,Drama,d,English,2010,4.0,k",
                "M2,B,Drama,d,English,inf,4.0,k",       # not finite
                "M3,C,Drama,d,English,-5,4.0,k",        # out of range
                "M4,D,Drama,d,English,1700,4.0,k",      # before cinema existed
                "M5,E,Drama,d,English,2010.5,4.0,k",    # not a whole year
                "M6,F,Drama,d,English,abc,4.0,k",
                "M7,G,Drama,d,English,9999,4.0,k"]      # far future
        df = self._load("movies", rows)
        self.assertEqual(df.loc[0, "release_year"], 2010)
        self.assertTrue(df["release_year"].iloc[1:].isna().all())
        self.assertEqual(len(df), 7)                      # records are kept, only the year is dropped
        self.assertTrue(any("invalid" in n for n in df.attrs["data_warnings"]))

    def test_invalid_ratings_become_missing(self):
        rows = ["M1,A,Drama,d,English,2010,4.5,k", "M2,B,Drama,d,English,2010,7,k",
                "M3,C,Drama,d,English,2010,-1,k", "M4,D,Drama,d,English,2010,nan,k",
                "M5,E,Drama,d,English,2010,inf,k", "M6,F,Drama,d,English,2010,great,k"]
        df = self._load("movies", rows)
        self.assertEqual(df.loc[0, "rating"], 4.5)
        self.assertTrue(df["rating"].iloc[1:].isna().all())

    def test_book_years_validated_too(self):
        df = self._load("books", ["B1,T,Au,Classic,d,English,1813,4.5,k", "B2,T2,Au,Classic,d,English,2500,4.5,k"])
        self.assertEqual(df.loc[0, "publication_year"], 1813)
        self.assertTrue(df["publication_year"].isna().iloc[1])

    def test_missing_ids_are_rejected_not_turned_into_nan_strings(self):
        rows = [",No id,Drama,d,English,2010,4.0,k", "  ,Blank id,Drama,d,English,2010,4.0,k",
                "M2,Has id,Drama,d,English,2010,4.0,k"]
        df = self._load("movies", rows)
        self.assertEqual(list(df["id"]), ["M2"])
        self.assertNotIn("nan", set(df["id"]))
        self.assertTrue(any("without a record id" in n for n in df.attrs["data_warnings"]))

    def test_all_ids_missing_raises(self):
        with self.assertRaises(DatasetError):
            self._load("movies", [",A,Drama,d,English,2010,4.0,k"])

    def test_numeric_ids_stay_clean_text(self):
        df = self._load("movies", ["7,A,Drama,d,English,2010,4.0,k", "8,B,Drama,d,English,2010,4.0,k"])
        self.assertEqual(list(df["id"]), ["7", "8"])

    def test_bundled_catalogs_have_no_data_warnings(self):
        for kind in ("movies", "books"):
            self.assertEqual(load_catalog(kind).attrs["data_warnings"], [])

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
