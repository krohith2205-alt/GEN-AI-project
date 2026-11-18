import unittest

from src.recommender import (MIN_SIMILARITY, MODERATE_SIMILARITY, STRONG_SIMILARITY, Filters, Preferences,
                             Recommendation, build_search_query, distance_to_similarity, fallback_reason,
                             match_strength, passes_filters, similarity_label)
from tests.helpers import make_recommender
from src.utils import clean_query, validate_query


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = make_recommender()

    def test_scifi_query_returns_scifi_movies(self):
        res = self.rec.search("mind-bending science fiction movies", "Movies", 5)
        self.assertEqual(list(res), ["movies"])
        self.assertTrue(res["movies"])
        genres = [r.record["genre"] for r in res["movies"]]
        self.assertIn("Science Fiction", genres[:3])

    def test_category_both_returns_two_groups(self):
        res = self.rec.search("fantasy adventure", "Both", 4)
        self.assertEqual(set(res), {"movies", "books"})
        self.assertLessEqual(len(res["movies"]), 4)

    def test_books_only(self):
        res = self.rec.search("beginner python programming", "Books", 3)
        self.assertEqual(list(res), ["books"])
        self.assertTrue(any("Python" in r.record["title"] for r in res["books"]))

    def test_genre_filter(self):
        res = self.rec.search("great story", "Movies", 10, {"movies": Filters(genres=["Romance"])})
        self.assertTrue(all(r.record["genre"] == "Romance" for r in res["movies"]))

    def test_language_and_rating_filter(self):
        res = self.rec.search("inspiring story", "Movies", 10, {"movies": Filters(languages=["Hindi"], min_rating=4.5)})
        for r in res["movies"]:
            self.assertEqual(r.record["language"], "Hindi")
            self.assertGreaterEqual(r.record["rating"], 4.5)

    def test_top_k_is_respected(self):
        res = self.rec.search("drama", "Movies", 2)
        self.assertLessEqual(len(res["movies"]), 2)

    def test_no_match_returns_empty(self):
        res = self.rec.search("zzzz qqqq xxxx", "Both", 5)
        self.assertEqual(res["movies"], [])
        self.assertEqual(res["books"], [])

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.rec.search("   ")

    def test_results_come_from_catalog_only(self):
        res = self.rec.search("mystery with unexpected endings", "Both", 5)
        for kind, items in res.items():
            titles = set(self.rec.catalogs[kind]["title"])
            for item in items:
                self.assertIn(item.record["title"], titles)

    def test_helpers(self):
        self.assertEqual(distance_to_similarity(0.0), 1.0)
        self.assertEqual(distance_to_similarity(2.0), 0.0)
        self.assertTrue(passes_filters({"genre": "Drama", "language": "English", "rating": 4.0}, Filters()))
        self.assertFalse(passes_filters({"genre": "Drama", "language": "English", "rating": None}, Filters(min_rating=3)))
        self.assertIn("Mood: Relaxing", build_search_query("movies", "Relaxing", ""))
        self.assertIn("Similar to: Inception", build_search_query("movies", "Any", "Inception"))

    def test_fallback_reason_uses_metadata(self):
        record = {"id": "M1", "title": "Inception", "genre": "Science Fiction", "keywords": "dreams, heist"}
        text = fallback_reason(Recommendation("movies", record, 0.5), "dreams and heist movies")
        self.assertIn("dreams", text)
        self.assertIn("science fiction", text)

    def test_fallback_wording_follows_match_strength(self):
        record = {"id": "M1", "title": "Inception", "genre": "Science Fiction", "keywords": "dreams, heist"}
        strong = fallback_reason(Recommendation("movies", record, STRONG_SIMILARITY + 0.1), "dreams")
        moderate = fallback_reason(Recommendation("movies", record, (STRONG_SIMILARITY + MODERATE_SIMILARITY) / 2), "dreams")
        weak = fallback_reason(Recommendation("movies", record, MIN_SIMILARITY + 0.01), "dreams")
        self.assertIn("strong semantic match", strong)
        self.assertIn("moderate match", moderate)
        self.assertIn("weak match", weak)
        self.assertIn("tentative", weak)
        self.assertNotIn("close semantic match", weak + moderate + strong)

    def test_weak_fallback_with_no_shared_words_is_honest(self):
        record = {"id": "M1", "title": "Inception", "genre": "Science Fiction", "keywords": "dreams, heist"}
        text = fallback_reason(Recommendation("movies", record, 0.16), "gardening tips")
        self.assertIn("weak match", text)
        self.assertIn("no specific shared theme", text)

    def test_fallback_uses_mood_and_favourites(self):
        record = {"id": "M1", "title": "Inception", "genre": "Science Fiction", "keywords": "dreams, heist"}
        prefs = Preferences("something to watch", "Relaxing", "heist films")
        self.assertIn("heist", fallback_reason(Recommendation("movies", record, 0.4), prefs))

    def test_match_strength_bands(self):
        self.assertEqual(match_strength(STRONG_SIMILARITY), "strong")
        self.assertEqual(match_strength(MODERATE_SIMILARITY), "moderate")
        self.assertEqual(match_strength(MIN_SIMILARITY), "weak")

    def test_similarity_is_labelled_as_semantic_similarity_not_probability(self):
        label = similarity_label(0.42)
        self.assertIn("Semantic similarity", label)
        self.assertIn("42%", label)
        self.assertNotIn("probab", label.lower())

    def test_preferences_search_text_and_favourites(self):
        prefs = Preferences("relaxing weekend", "Relaxing", " Inception ,  The Hobbit,, ")
        self.assertEqual(prefs.favorite_list, ["Inception", "The Hobbit"])
        self.assertEqual(prefs.search_text(), build_search_query("relaxing weekend", "Relaxing", " Inception ,  The Hobbit,, "))
        self.assertEqual(Preferences("x").search_text(), "x")

    def test_search_accepts_preferences_object(self):
        prefs = Preferences("mind-bending science fiction", "Any", "")
        res = self.rec.search(prefs, "Movies", 3)
        self.assertTrue(res["movies"])

    def test_unknown_titles_are_reported(self):
        prefs = Preferences("x", "Any", "Inception, The Hobbit, Totally Made Up Film")
        self.assertEqual(self.rec.unknown_titles(prefs), ["Totally Made Up Film"])
        self.assertEqual(self.rec.unknown_titles("inception"), [])      # case-insensitive

    def test_query_validation(self):
        self.assertFalse(validate_query("   ")[0])
        self.assertFalse(validate_query("!!!")[0])
        self.assertTrue(validate_query("sci-fi movies")[0])
        self.assertEqual(len(clean_query("a" * 1000)), 300)


if __name__ == "__main__":
    unittest.main()
