"""Look at REAL similarity scores to choose the similarity thresholds in src/recommender.py.

    python evaluation/threshold_report.py

It uses the real embedding model (needs internet the first time) and prints
  * the similarity of the items you EXPECT for on-topic requests (from eval_cases.json), and
  * the highest similarity that ANY item reaches for clearly unrelated requests.
A good MIN_SIMILARITY lies above the unrelated scores and below the on-topic scores.
Copy your chosen numbers into MIN_SIMILARITY / MODERATE_SIMILARITY / STRONG_SIMILARITY and re-run the evaluation.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.recommender as R  # noqa: E402
from src.embeddings import get_embedding_model  # noqa: E402

UNRELATED = ["How do I repair a bicycle tyre?", "What is the weather forecast for tomorrow?",
             "How to file income tax returns", "asdf qwer zxcv", "latest cricket match score",
             "best way to cook rice", "how do I reset my router password"]


def all_scores(rec, query):
    """Similarity of EVERY record (no threshold, no filters)."""
    old, R.MIN_SIMILARITY = R.MIN_SIMILARITY, 0.0
    try:
        out = {}
        for kind in R.KINDS:
            for r in rec.search_kind(kind, query, 1000, R.Filters()):
                out[(kind, r.record["title"])] = r.similarity
        return out
    finally:
        R.MIN_SIMILARITY = old


def main():
    rec = R.Recommender.from_embeddings(get_embedding_model())
    cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text(encoding="utf-8"))
    on_topic = []
    print("ON-TOPIC requests: similarity of the titles we expect")
    for c in cases:
        if not c.get("expected_titles") or c.get("languages") or c.get("min_rating") or c.get("llm_mode") != "real":
            continue
        scores = all_scores(rec, R.Preferences(c["query"], c.get("mood", "Any"), c.get("favorites", "")).search_text())
        found = [s for (kind, title), s in scores.items() if title in c["expected_titles"]]
        if found:
            on_topic += found
            print(f"  {c['id']}  min {min(found):.2f}  median {statistics.median(found):.2f}  max {max(found):.2f}   {c['name']}")
    print("\nUNRELATED requests: highest similarity of any record")
    off_topic = []
    for q in UNRELATED:
        top = max(all_scores(rec, q).values())
        off_topic.append(top)
        print(f"  top {top:.2f}   {q}")
    print("\nSummary")
    print(f"  lowest expected on-topic similarity : {min(on_topic):.2f}")
    print(f"  highest unrelated similarity        : {max(off_topic):.2f}")
    print(f"  current MIN_SIMILARITY={R.MIN_SIMILARITY}, MODERATE={R.MODERATE_SIMILARITY}, STRONG={R.STRONG_SIMILARITY}")
    gap = max(off_topic) < min(on_topic)
    print("  -> " + (f"a MIN_SIMILARITY between {max(off_topic):.2f} and {min(on_topic):.2f} separates the two groups."
                     if gap else "the groups overlap: MIN_SIMILARITY alone cannot separate them; consider a higher value "
                                 "and accept that some weak on-topic items are dropped."))


if __name__ == "__main__":
    main()
