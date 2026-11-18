import json
import unittest
from types import SimpleNamespace

from src.llm_service import (LLMService, add_explanations, build_user_prompt, check_evidence, classify_error,
                             find_unsupported_claims, friendly_error, ground_reasons, parse_reasons)
from src.recommender import Preferences, Recommendation

RECORD = {"id": "M001", "title": "Inception", "genre": "Science Fiction",
          "description": "A skilled thief enters people's dreams to steal secrets.",
          "keywords": "mind-bending, dreams, heist, reality, twist"}


def make_recs():
    return {"movies": [Recommendation("movies", dict(RECORD), 0.5)]}


def good_answer(explanation="This film may suit you because it is about dreams and a heist.", evidence=None):
    evidence = evidence or [{"field": "keywords", "text": "dreams"}, {"field": "genre", "text": "Science Fiction"}]
    return json.dumps({"reasons": {"M001": {"explanation": explanation, "evidence": evidence}}})


class HTTPError(Exception):
    """Looks like huggingface_hub's HTTP errors: has .response.status_code."""
    def __init__(self, status, message=""):
        super().__init__(message or f"{status} error")
        self.response = SimpleNamespace(status_code=status)


class FakeClient:
    """Returns a streamed answer. `script` is a list of items; each is a string (success) or an exception."""
    def __init__(self, *script):
        self.script, self.calls, self.kwargs = list(script), 0, []

    def chat_completion(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        chunks = [item[i:i + 12] for i in range(0, len(item), 12)]          # stream in small pieces
        return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))]) for c in chunks)


def service(*script, retries=3):
    sleeps = []
    svc = LLMService(client=FakeClient(*script), max_retries=retries, sleep=sleeps.append)
    svc.sleeps = sleeps
    return svc


class TestParseReasons(unittest.TestCase):
    """Regression tests: malformed model output must give {} (-> catalog fallback), never an exception."""
    IDS = {"M001"}

    def test_valid_answer(self):
        out = parse_reasons(good_answer(), self.IDS)
        self.assertEqual(out["M001"]["explanation"], "This film may suit you because it is about dreams and a heist.")
        self.assertEqual(out["M001"]["evidence"][0], ("keywords", "dreams"))

    def test_reasons_null(self):
        self.assertEqual(parse_reasons('{"reasons": null}', self.IDS), {})

    def test_reasons_list(self):
        self.assertEqual(parse_reasons('{"reasons": []}', self.IDS), {})

    def test_reasons_string_and_number(self):
        self.assertEqual(parse_reasons('{"reasons": "text"}', self.IDS), {})
        self.assertEqual(parse_reasons('{"reasons": 5}', self.IDS), {})

    def test_outer_object_is_a_list(self):
        self.assertEqual(parse_reasons('[{"M001": "x"}]', self.IDS), {})
        self.assertEqual(parse_reasons('"just text"', self.IDS), {})

    def test_missing_fields(self):
        self.assertEqual(parse_reasons("{}", self.IDS), {})
        self.assertEqual(parse_reasons('{"reasons": {"M001": {}}}', self.IDS), {})
        self.assertEqual(parse_reasons('{"reasons": {"M001": {"evidence": []}}}', self.IDS), {})

    def test_empty_or_wrong_type_explanations(self):
        for value in ('""', '"   "', "null", "5", "[]", '{"explanation": ""}', '{"explanation": null}',
                      '{"explanation": 7}', '{"explanation": ["a"]}'):
            self.assertEqual(parse_reasons('{"reasons": {"M001": %s}}' % value, self.IDS), {}, value)

    def test_evidence_with_wrong_shape_is_dropped(self):
        out = parse_reasons('{"reasons": {"M001": {"explanation": "ok", "evidence": "dreams"}}}', self.IDS)
        self.assertEqual(out["M001"]["evidence"], [])
        out = parse_reasons('{"reasons": {"M001": {"explanation": "ok", "evidence": [1, null, {"field": 3}]}}}', self.IDS)
        self.assertEqual(out["M001"]["evidence"], [])

    def test_invented_ids_are_ignored(self):
        self.assertEqual(parse_reasons('{"reasons": {"M999": {"explanation": "fake"}}}', self.IDS), {})

    def test_not_json_or_not_text(self):
        for text in ("sorry, no json", "", None, 42, "{broken json"):
            self.assertEqual(parse_reasons(text, self.IDS), {})

    def test_json_inside_markdown_fence(self):
        self.assertIn("M001", parse_reasons("```json\n" + good_answer() + "\n```", self.IDS))


class TestGrounding(unittest.TestCase):
    def test_invented_actor_is_flagged(self):
        flagged = find_unsupported_claims("It stars Leonardo DiCaprio as a thief in dreams.", RECORD)
        self.assertTrue(flagged)
        self.assertIn("Leonardo", flagged)

    def test_invented_award_and_number_are_flagged(self):
        self.assertTrue(find_unsupported_claims("This award winning film grossed 800 million.", RECORD))
        self.assertTrue(find_unsupported_claims("It won an Oscar for visual effects.", RECORD))
        self.assertIn("2010", find_unsupported_claims("Released in 2010, it explores dreams.", RECORD))

    def test_invented_plot_name_is_flagged(self):
        self.assertTrue(find_unsupported_claims("The thief travels to Paris to steal secrets.", RECORD))

    def test_supported_text_passes(self):
        text = "This science fiction film may suit you: a thief enters dreams to steal secrets in a heist."
        self.assertEqual(find_unsupported_claims(text, RECORD), [])

    def test_names_from_the_users_own_preferences_are_allowed(self):
        text = "Like Inception, it blends dreams with a heist."
        self.assertEqual(find_unsupported_claims(text, RECORD, "Similar to: Inception"), [])

    def test_evidence_must_be_quoted_from_the_record(self):
        self.assertEqual(check_evidence([("keywords", "dreams")], RECORD), [])
        self.assertTrue(check_evidence([("keywords", "aliens")], RECORD))     # not in the record
        self.assertTrue(check_evidence([("cast", "someone")], RECORD))        # unknown field
        self.assertTrue(check_evidence([], RECORD))                            # no evidence

    def test_ground_reasons_splits_accepted_and_rejected(self):
        parsed = {"M001": {"explanation": "Stars Leonardo DiCaprio in dreams.", "evidence": [("keywords", "dreams")]},
                  "M002": {"explanation": "About dreams.", "evidence": [("keywords", "dreams")]}}
        accepted, rejected = ground_reasons(parsed, {"M001": RECORD, "M002": RECORD})
        self.assertEqual(list(accepted), ["M002"])
        self.assertIn("unsupported", rejected["M001"])

    def test_parser_accepts_but_grounding_rejects_invented_actor_for_valid_id(self):
        """The exact problem from the feedback: valid ID, invented actor."""
        answer = good_answer("This film stars Tom Hardy and suits fans of dreams.")
        results = make_recs()
        warnings = add_explanations("dream movies", results, LLMService(client=FakeClient(answer)))
        rec = results["movies"][0]
        self.assertEqual(rec.reason_source, "catalog")
        self.assertNotIn("Tom Hardy", rec.reason)
        self.assertTrue(warnings)


class TestLLMService(unittest.TestCase):
    def test_no_token_uses_catalog_fallback(self):
        results = make_recs()
        warnings = add_explanations("dream movies", results, LLMService(token=""))
        rec = results["movies"][0]
        self.assertEqual(rec.reason_source, "catalog")
        self.assertTrue(rec.reason)
        self.assertTrue(warnings)

    def test_valid_grounded_answer_is_used(self):
        results = make_recs()
        warnings = add_explanations("dream movies", results, service(good_answer()))
        self.assertEqual(results["movies"][0].reason_source, "llm")
        self.assertEqual(warnings, [])

    def test_answer_without_evidence_is_rejected(self):
        answer = json.dumps({"reasons": {"M001": "It explores dreams."}})
        results = make_recs()
        warnings = add_explanations("dream movies", results, service(answer))
        self.assertEqual(results["movies"][0].reason_source, "catalog")
        self.assertTrue(warnings)

    def test_fabricated_evidence_is_rejected(self):
        answer = good_answer(evidence=[{"field": "description", "text": "a famous director's masterpiece"}])
        results = make_recs()
        add_explanations("dream movies", results, service(answer))
        self.assertEqual(results["movies"][0].reason_source, "catalog")

    def test_malformed_json_structure_triggers_fallback_not_crash(self):
        """Regression: {"reasons": null} / [] used to raise AttributeError outside the error handler."""
        for bad in ('{"reasons": null}', '{"reasons": []}', "[]", '{"reasons": {"M001": null}}'):
            results = make_recs()
            warnings = add_explanations("q", results, service(bad))
            self.assertEqual(results["movies"][0].reason_source, "catalog", bad)
            self.assertTrue(warnings, bad)

    def test_api_error_gives_friendly_message_and_fallback(self):
        svc = LLMService(model_name="x/y", client=FakeClient(RuntimeError("404 model_not_supported")), sleep=lambda s: None)
        results = make_recs()
        warnings = add_explanations("dream movies", results, svc)
        self.assertIn("x/y", warnings[0])
        self.assertEqual(results["movies"][0].reason_source, "catalog")

    def test_unreadable_output_falls_back(self):
        results = make_recs()
        warnings = add_explanations("q", results, service("sorry, no json"))
        self.assertTrue(warnings)
        self.assertEqual(results["movies"][0].reason_source, "catalog")

    def test_prompt_contains_only_retrieved_records(self):
        prompt = build_user_prompt("dream movies", make_recs()["movies"])
        self.assertIn("Inception", prompt)
        self.assertEqual(prompt.count("<record "), 1)


class TestPreferenceContext(unittest.TestCase):
    def test_explanation_prompt_contains_mood_and_favourites(self):
        prefs = Preferences("something to watch", "Relaxing", "Inception, The Hobbit")
        prompt = build_user_prompt(prefs, make_recs()["movies"])
        self.assertIn("something to watch", prompt)
        self.assertIn("Relaxing", prompt)
        self.assertIn("Inception, The Hobbit", prompt)

    def test_llm_receives_same_context_that_retrieval_used(self):
        prefs = Preferences("something to watch", "Relaxing", "Inception")
        client = FakeClient(good_answer())
        add_explanations(prefs, make_recs(), LLMService(client=client))
        sent = client.kwargs[0]["messages"][-1]["content"]
        for part in ("something to watch", "Relaxing", "Inception"):
            self.assertIn(part, sent)
            self.assertIn(part, prefs.search_text())

    def test_unspecified_preferences_are_marked(self):
        prompt = build_user_prompt(Preferences("q"), make_recs()["movies"])
        self.assertIn("<mood>not specified</mood>", prompt)


class TestFewShotAndStreaming(unittest.TestCase):
    def test_request_includes_few_shot_example_and_streams(self):
        client = FakeClient(good_answer())
        LLMService(client=client).explain("dream movies", make_recs()["movies"])
        roles = [m["role"] for m in client.kwargs[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])         # system, example in/out, real request
        self.assertTrue(client.kwargs[0]["stream"])
        example = json.loads(client.kwargs[0]["messages"][2]["content"])        # the example answer is valid JSON
        self.assertIn("evidence", example["reasons"]["X1"])

    def test_pieces_are_streamed_to_callback(self):
        pieces = []
        svc = LLMService(client=FakeClient(good_answer()))
        result = svc.explain("dream movies", make_recs()["movies"], on_token=pieces.append)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces), good_answer())
        self.assertEqual(list(result.reasons), ["M001"])

    def test_non_streaming_response_object_still_works(self):
        class OneShot:
            def chat_completion(self, **kw):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=good_answer()))])
        result = LLMService(client=OneShot()).explain("q", make_recs()["movies"])
        self.assertEqual(list(result.reasons), ["M001"])


class TestRetries(unittest.TestCase):
    def test_temporary_errors_are_retried_then_succeed(self):
        svc = service(HTTPError(503), HTTPError(429), good_answer())
        result = svc.explain("q", make_recs()["movies"])
        self.assertEqual(result.attempts, 3)
        self.assertEqual(list(result.reasons), ["M001"])
        self.assertEqual(len(svc.sleeps), 2)
        self.assertGreater(svc.sleeps[1], svc.sleeps[0])          # backoff grows

    def test_retries_are_bounded(self):
        svc = service(HTTPError(503))
        result = svc.explain("q", make_recs()["movies"])
        self.assertEqual(svc._client.calls, 3)                    # never more than max_retries attempts
        self.assertEqual(result.attempts, 3)
        self.assertIn("tried 3 times", result.error)
        self.assertEqual(result.reasons, {})

    def test_timeout_and_connection_errors_are_retried(self):
        for exc in (TimeoutError("timed out"), ConnectionError("Connection reset")):
            svc = service(exc, good_answer())
            self.assertEqual(svc.explain("q", make_recs()["movies"]).attempts, 2)

    def test_authentication_errors_are_not_retried(self):
        for status in (401, 403):
            svc = service(HTTPError(status))
            result = svc.explain("q", make_recs()["movies"])
            self.assertEqual(svc._client.calls, 1)
            self.assertEqual(svc.sleeps, [])
            self.assertIn("token was rejected", result.error)

    def test_model_and_quota_errors_are_not_retried(self):
        for status in (404, 402):
            svc = service(HTTPError(status))
            svc.explain("q", make_recs()["movies"])
            self.assertEqual(svc._client.calls, 1)

    def test_error_midway_through_stream_is_retried_and_ui_reset(self):
        class FlakyStream:
            calls = 0
            def chat_completion(self, **kw):
                FlakyStream.calls += 1
                if FlakyStream.calls == 1:
                    def gen():
                        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"reas'))])
                        raise ConnectionError("connection dropped")
                    return gen()
                return FakeClient(good_answer()).chat_completion(**kw)
        events = []
        svc = LLMService(client=FlakyStream(), sleep=lambda s: None)
        result = svc.explain("q", make_recs()["movies"], on_token=events.append)
        self.assertIn(None, events)                               # UI was told to discard partial text
        self.assertEqual(list(result.reasons), ["M001"])

    def test_classification(self):
        self.assertEqual(classify_error(HTTPError(401)), "auth")
        self.assertEqual(classify_error(HTTPError(402)), "quota")
        self.assertEqual(classify_error(HTTPError(404)), "model")
        self.assertEqual(classify_error(HTTPError(429)), "rate_limit")
        self.assertEqual(classify_error(HTTPError(502)), "server")
        self.assertEqual(classify_error(TimeoutError()), "network")
        self.assertEqual(classify_error(ValueError("weird")), "unknown")
        self.assertIn("m/x", friendly_error(HTTPError(404), "m/x"))


if __name__ == "__main__":
    unittest.main()
