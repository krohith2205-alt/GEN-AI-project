import unittest

from src.utils import clean_query, format_rating, format_year, is_finite_number, validate_query


class TestUtils(unittest.TestCase):
    def test_format_year_is_safe_for_every_kind_of_value(self):
        """Regression: int(year) crashed for NaN / inf, and 'truthy' checks let NaN through."""
        self.assertEqual(format_year(2010), "2010")
        self.assertEqual(format_year(2010.0), "2010")
        self.assertEqual(format_year("1999"), "1999")
        for bad in (None, float("nan"), float("inf"), float("-inf"), "abc", "", 0, -3, 2010.5, 99999, [], {}):
            self.assertEqual(format_year(bad), "Year n/a", repr(bad))

    def test_format_rating(self):
        self.assertEqual(format_rating(4.5), "4.5 / 5")
        for bad in (None, float("nan"), float("inf"), "x"):
            self.assertEqual(format_rating(bad), "Not available")

    def test_is_finite_number(self):
        self.assertTrue(is_finite_number(0))
        self.assertFalse(is_finite_number(float("nan")))
        self.assertFalse(is_finite_number(None))

    def test_query_validation(self):
        self.assertFalse(validate_query("   ")[0])
        self.assertFalse(validate_query("!!!")[0])
        self.assertTrue(validate_query("sci-fi movies")[0])
        self.assertEqual(len(clean_query("a" * 1000)), 300)


if __name__ == "__main__":
    unittest.main()
