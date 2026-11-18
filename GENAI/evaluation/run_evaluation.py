"""Run the evaluation cases against the REAL embedding model and the REAL language model.

    python evaluation/run_evaluation.py              # real run (needs internet + HF_TOKEN in .env)
    python evaluation/run_evaluation.py --dry-run    # offline smoke test of this script (NOT real results)

Output (in evaluation/):
    results.csv            one row per recommended item, with automatic checks and EMPTY columns for your
                           manual relevance / unsupported-claim judgments
    EVALUATION_REPORT.md   one section per case: expected vs actual, automatic checks, API behaviour

After the run, open both files and fill in the manual columns, then submit them.
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_service import LLMService, add_explanations, find_unsupported_claims, friendly_error  # noqa: E402
from src.recommender import Filters, Preferences, Recommender, match_strength  # noqa: E402
from src.utils import load_settings  # noqa: E402

HERE = Path(__file__).resolve().parent
WEAK_BELOW = 0.30   # same as MODERATE_SIMILARITY in src/recommender.py


class FailingClient:
    """Simulates an API that always fails with an HTTP status (used for outage cases and for dry runs)."""
    def __init__(self, status: int, message: str = ""):
        self.status, self.message, self.calls = status, message, 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        exc = RuntimeError(self.message or f"{self.status} simulated error")
        exc.response = type("Resp", (), {"status_code": self.status})()
        raise exc


def build_recommender(dry_run: bool):
    if dry_run:
        sys.path.insert(0, str(ROOT))
        from tests.helpers import make_recommender        # hashed bag-of-words stand-in, NOT real embeddings
        return make_recommender(), "DRY RUN stand-in (hashed bag-of-words) - not real embeddings"
    from src.embeddings import get_embedding_model
    from src.utils import EMBEDDING_MODEL_NAME
    return Recommender.from_embeddings(get_embedding_model()), EMBEDDING_MODEL_NAME


def build_llm(mode: str, settings: dict, dry_run: bool):
    """Return (LLMService, description) for one case."""
    token, model = settings["hf_token"], settings["model_name"]
    fast = dict(max_retries=settings["max_retries"], sleep=lambda s: None if dry_run else time.sleep(min(s, 2)))
    if mode == "none":
        return LLMService(token="", model_name=model), "no token"
    if mode == "simulated_503":
        return LLMService(client=FailingClient(503), model_name=model, **{**fast, "max_retries": 3,
                          "sleep": lambda s: None}), "simulated 503 outage (3 attempts)"
    if mode == "invalid_token":
        if dry_run:
            return LLMService(client=FailingClient(401), model_name=model, **fast), "simulated 401 (dry run)"
        return LLMService(token="hf_invalid_token_for_evaluation", model_name=model, **fast), "real call, invalid token"
    if mode == "bad_model":
        if dry_run:
            return LLMService(client=FailingClient(404), model_name="nonexistent/model-xyz", **fast), "simulated 404 (dry run)"
        return LLMService(token=token, model_name="nonexistent/model-xyz", **fast), "real call, nonexistent model"
    if dry_run:
        return LLMService(token="", model_name=model), "dry run: no LLM"
    return LLMService(token=token, model_name=model, **fast), f"real model {model}" if token else "NO TOKEN in .env"


def case_filters(case: dict) -> dict:
    languages, rating = case.get("languages", []), case.get("min_rating", 0.0)
    return {"movies": Filters(case.get("movie_genres", []), languages, rating),
            "books": Filters(case.get("book_genres", []), languages, rating)}


def run_case(case: dict, recommender, settings: dict, dry_run: bool) -> dict:
    prefs = Preferences(case["query"], case.get("mood", "Any"), case.get("favorites", ""))
    llm, llm_desc = build_llm(case.get("llm_mode", "real"), settings, dry_run)
    started = time.time()
    results = recommender.search(prefs.search_text(), case.get("category", "Both"), case.get("top_k", 5), case_filters(case))
    warnings = add_explanations(prefs, results, llm)
    unknown = recommender.unknown_titles(prefs)
    outcome = llm.last_result
    items = []
    for kind, recs in results.items():
        for rank, r in enumerate(recs, 1):
            claims = find_unsupported_claims(r.reason, r.record, prefs.search_text()) if r.reason_source == "llm" else []
            items.append({"kind": kind, "rank": rank, "rec": r, "claims": claims})
    return {"case": case, "prefs": prefs, "items": items, "warnings": warnings, "unknown": unknown,
            "llm_desc": llm_desc, "attempts": outcome.attempts, "rejected": outcome.rejected,
            "seconds": round(time.time() - started, 1)}


def automatic_checks(run: dict) -> dict:
    case, items = run["case"], run["items"]
    titles = [i["rec"].record["title"] for i in items]
    checks = {}
    expected = case.get("expected_titles", [])
    if expected:
        hits = [t for t in expected if t in titles]
        checks[f"expected titles found (>= {case.get('min_expected_hits', 1)})"] = len(hits) >= case.get("min_expected_hits", 1)
    if case.get("expect_empty"):
        checks["no results returned"] = not items
    if case.get("expect_all_weak_or_empty"):
        checks["empty or only weak matches"] = all(i["rec"].similarity < WEAK_BELOW for i in items)
    if case.get("expect_languages"):
        checks["language filter respected"] = all(i["rec"].record["language"] in case["expect_languages"] for i in items)
    if case.get("min_rating"):
        checks["rating filter respected"] = all((i["rec"].record.get("rating") or 0) >= case["min_rating"] for i in items)
    if case.get("movie_genres"):
        checks["genre filter respected"] = all(i["rec"].record["genre"] in case["movie_genres"] for i in items if i["kind"] == "movies")
    if case.get("forbidden_titles"):
        checks["forbidden titles absent"] = not any(t in titles for t in case["forbidden_titles"])
    if "expected_unknown_titles" in case:
        checks["unknown favourite reported"] = run["unknown"] == case["expected_unknown_titles"]
    if case.get("expected_warning_contains"):
        checks["expected warning shown"] = any(case["expected_warning_contains"].lower() in w.lower() for w in run["warnings"])
    if case.get("expected_reason_source"):
        checks[f"all explanations from '{case['expected_reason_source']}'"] = all(
            i["rec"].reason_source == case["expected_reason_source"] for i in items)
    if "expected_attempts" in case:
        checks[f"API attempts == {case['expected_attempts']}"] = run["attempts"] == case["expected_attempts"]
    llm_items = [i for i in items if i["rec"].reason_source == "llm"]
    if llm_items:
        checks["no unsupported claims in AI explanations"] = not any(i["claims"] for i in llm_items)
    if items:
        checks["every item has an explanation"] = all(i["rec"].reason for i in items)
    return checks


def write_outputs(runs: list, out_dir: Path, meta: dict):
    fields = ["case_id", "case_name", "query", "category", "mood", "favourites", "filters", "rank", "kind", "title",
              "genre", "language", "rating", "semantic_similarity", "strength", "explanation_source", "explanation",
              "expected_title", "unsupported_claims_found_by_script", "relevance_0_1_2 (FILL IN)",
              "unsupported_claims_in_explanation Y/N (FILL IN)", "reviewer_notes (FILL IN)"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for run in runs:
            case, prefs = run["case"], run["prefs"]
            filters = json.dumps({k: case[k] for k in ("movie_genres", "book_genres", "languages", "min_rating") if k in case})
            if not run["items"]:
                w.writerow([case["id"], case["name"], case["query"], case.get("category", "Both"), prefs.mood, prefs.favorites,
                            filters, "-", "-", "(no results)", "", "", "", "", "", "", "", "", "", "", "", ""])
            for i in run["items"]:
                r, rec = i["rec"], i["rec"].record
                w.writerow([case["id"], case["name"], case["query"], case.get("category", "Both"), prefs.mood, prefs.favorites,
                            filters, i["rank"], i["kind"], rec["title"], rec["genre"], rec["language"], rec.get("rating"),
                            f"{r.similarity:.3f}", match_strength(r.similarity), r.reason_source, r.reason,
                            "Y" if rec["title"] in case.get("expected_titles", []) else "N", "; ".join(i["claims"]), "", "", ""])

    passed = total = 0
    lines = [f"# Evaluation report", "",
             f"- Run date: {meta['date']}",
             f"- Mode: **{meta['mode']}**",
             f"- Embedding model: {meta['embedding']}",
             f"- Language model: {meta['llm']}",
             f"- Python {platform.python_version()} on {platform.system()}", ""]
    if meta["dry_run"]:
        lines += ["> **DRY RUN - these are NOT real-model results.** Run `python evaluation/run_evaluation.py` "
                  "without `--dry-run` to produce the results you submit.", ""]
    summary = ["| Case | Name | LLM mode | Automatic checks | Items |", "|---|---|---|---|---|"]
    details = []
    for run in runs:
        case, checks = run["case"], automatic_checks(run)
        ok = all(checks.values())
        passed, total = passed + ok, total + 1
        titles = [f"{i['rec'].record['title']} ({i['rec'].similarity:.0%})" for i in run["items"]]
        summary.append(f"| {case['id']} | {case['name']} | {run['llm_desc']} | {'PASS' if ok else '**FAIL**'} | {len(run['items'])} |")
        details += [f"## {case['id']} - {case['name']}", "",
                    f"- **Request:** {case['query']}  (category: {case.get('category', 'Both')}, mood: {run['prefs'].mood}, "
                    f"favourites: {run['prefs'].favorites or '-'})",
                    f"- **Filters:** " + (json.dumps({k: case[k] for k in ('movie_genres', 'book_genres', 'languages', 'min_rating') if k in case}) or "none"),
                    f"- **Expected:** " + ("; ".join(case.get("expected_titles", [])) or
                                          ("no results" if case.get("expect_empty") else "see notes")),
                    f"- **Actual:** " + ("; ".join(titles) or "no results"),
                    f"- **API:** {run['llm_desc']}; attempts made: {run['attempts']}; time: {run['seconds']} s",
                    f"- **Warnings shown to the user:** " + (" | ".join(run["warnings"]) or "none"),
                    f"- **AI explanations rejected by the fact check:** " + (json.dumps(run["rejected"]) if run["rejected"] else "none"),
                    f"- **Notes:** {case.get('notes', '-')}", "", "| Automatic check | Result |", "|---|---|"]
        details += [f"| {name} | {'PASS' if value else '**FAIL**'} |" for name, value in checks.items()]
        details += ["", "**Manual judgment (fill in):** relevance of each result (0 = irrelevant, 1 = partly, 2 = relevant): ______  "
                    "| unsupported claims in explanations? ______  | comments: ______", ""]
    lines += [f"**Automatic checks passed: {passed} / {total} cases**", ""] + summary + [""] + details
    (out_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return passed, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="offline smoke test with stand-in embeddings and no real LLM")
    parser.add_argument("--cases", default=str(HERE / "eval_cases.json"))
    parser.add_argument("--out-dir", default=str(HERE))
    args = parser.parse_args()

    settings = load_settings()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    recommender, embedding = build_recommender(args.dry_run)
    if not args.dry_run and not settings["hf_token"]:
        print("WARNING: HF_TOKEN is not set - cases that need the real LLM will show catalog explanations only.")
    runs = []
    for case in cases:
        print(f"Running {case['id']}: {case['name']} ...")
        runs.append(run_case(case, recommender, settings, args.dry_run))
    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "dry_run": args.dry_run,
            "mode": "DRY RUN (not real)" if args.dry_run else "REAL embedding model + real Hugging Face calls",
            "embedding": embedding, "llm": settings["model_name"] if not args.dry_run else "none (dry run)"}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    passed, total = write_outputs(runs, Path(args.out_dir), meta)
    print(f"\nDone. Automatic checks passed for {passed} of {total} cases.")
    print(f"Open {Path(args.out_dir) / 'EVALUATION_REPORT.md'} and {Path(args.out_dir) / 'results.csv'} and fill in the manual columns.")


if __name__ == "__main__":
    main()
