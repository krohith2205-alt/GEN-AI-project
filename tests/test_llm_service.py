import json
import unittest
from types import SimpleNamespace

from src.llm_service import LLMService, add_explanations, build_user_prompt, parse_reasons
from src.recommender import Recommendation


def make_recs():
    return {"movies": [Recommendation("movies", {"id": "M001", "title": "Inception", "genre": "Science Fiction",
                                                 "description": "Dreams.", "keywords": "dreams"}, 0.5)]}


class FakeClient:
    def __init__(self, content=None, error=None):
        self.content, self.error = content, error

    def chat_completion(self, **kwargs):
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class TestLLMService(unittest.TestCase):
    def test_no_token_uses_catalog_fallback(self):
        results = make_recs()
        warnings = add_explanations("dream movies", results, LLMService(token=""))
        rec = results["movies"][0]
        self.assertEqual(rec.reason_source, "catalog")
        self.assertTrue(rec.reason)
        self.assertTrue(warnings)

    def test_valid_llm_answer_is_used(self):
        client = FakeClient(json.dumps({"reasons": {"M001": "It explores dreams."}}))
        results = make_recs()
        warnings = add_explanations("dream movies", results, LLMService(client=client))
        self.assertEqual(results["movies"][0].reason_source, "llm")
        self.assertEqual(warnings, [])

    def test_invented_ids_are_ignored(self):
        self.assertEqual(parse_reasons('{"reasons": {"M999": "fake"}}', {"M001"}), {})

    def test_api_error_gives_friendly_message_and_fallback(self):
        client = FakeClient(error=RuntimeError("404 model_not_supported"))
        results = make_recs()
        warnings = add_explanations("dream movies", results, LLMService(model_name="x/y", client=client))
        self.assertIn("x/y", warnings[0])
        self.assertEqual(results["movies"][0].reason_source, "catalog")

    def test_unreadable_output_falls_back(self):
        results = make_recs()
        warnings = add_explanations("q", results, LLMService(client=FakeClient("sorry, no json")))
        self.assertTrue(warnings)
        self.assertEqual(results["movies"][0].reason_source, "catalog")

    def test_prompt_contains_only_retrieved_records(self):
        prompt = build_user_prompt("dream movies", make_recs()["movies"])
        self.assertIn("Inception", prompt)
        self.assertEqual(prompt.count("<record "), 1)


if __name__ == "__main__":
    unittest.main()
