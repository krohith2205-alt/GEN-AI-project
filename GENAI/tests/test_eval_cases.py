"""The evaluation cases must be consistent with the catalog (typos here would silently ruin the evaluation)."""
import json
import unittest
from pathlib import Path

from src.data_loader import load_catalog

CASES = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "eval_cases.json").read_text(encoding="utf-8"))
MODES = {"real", "none", "invalid_token", "bad_model", "simulated_503"}


class TestEvalCases(unittest.TestCase):
    def test_at_least_ten_cases_with_required_kinds(self):
        self.assertGreaterEqual(len(CASES), 10)
        self.assertEqual(len({c["id"] for c in CASES}), len(CASES))
        names = " ".join(c["name"].lower() for c in CASES)
        for needed in ("restrictive filters", "unknown favourite", "unrelated", "api failure"):
            self.assertIn(needed, names)

    def test_expected_titles_exist_in_the_catalog(self):
        titles = set(load_catalog("movies")["title"]) | set(load_catalog("books")["title"])
        for c in CASES:
            for t in c.get("expected_titles", []):
                self.assertIn(t, titles, f"{c['id']}: '{t}' is not in the catalog")

    def test_filters_use_values_that_exist(self):
        movies, books = load_catalog("movies"), load_catalog("books")
        languages = set(movies["language"]) | set(books["language"])
        for c in CASES:
            for lang in c.get("languages", []):
                self.assertIn(lang, languages, c["id"])
            for g in c.get("movie_genres", []):
                self.assertIn(g, set(movies["genre"]), c["id"])
            self.assertIn(c.get("llm_mode", "real"), MODES, c["id"])

    def test_unknown_favourites_really_are_unknown(self):
        titles = {t.lower() for t in set(load_catalog("movies")["title"]) | set(load_catalog("books")["title"])}
        for c in CASES:
            for t in c.get("expected_unknown_titles", []):
                self.assertNotIn(t.lower(), titles)


if __name__ == "__main__":
    unittest.main()
