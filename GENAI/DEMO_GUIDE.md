# Demo guide (what to say and show)

Your reviewer asked you to explain three things. Show each one live.

## 1. How recommendations are selected
1. Type *"mind-bending science fiction movies"*, mood **Thought-provoking**, favourite **Inception**.
2. Say: "The app joins my request, mood and favourite titles into ONE preference text. That text is turned into an
   embedding and FAISS compares it with the embedding of every movie/book. Genre, language and rating filters are
   then applied, and only the closest records are shown."
3. Point at the card: title, year, rating come from the CSV. The percentage is **semantic similarity**, not the
   chance that I will enjoy it. The LLM only writes the explanation.
4. Say: "The SAME preference object (request + mood + favourites) goes to the explanation writer, so the AI sees
   what retrieval used."

## 2. Why a weak match is rejected
1. Type an unrelated request: *"How do I repair a bicycle tyre?"*
2. Say: "Every result has a semantic similarity. Below `MIN_SIMILARITY` (0.15 by default) it is dropped, so the app
   prefers saying 'no suitable item' over inventing a link. Items between the minimum and 0.30 are shown but
   explained as a *weak* match."
3. Show the numbers you measured with `python evaluation/threshold_report.py` and why you chose the thresholds.
4. Restrictive filters: Romance + Korean + rating 4.8 -> friendly "no suitable movie" message.

## 3. What happens when the LLM is unavailable
Show each case (you can use the sidebar status line and the warning box):
- **No token** (empty `HF_TOKEN`): warning + catalog-based explanations, results unchanged.
- **Invalid token**: "token was rejected" - NOT retried (retrying cannot fix it).
- **Temporary outage / rate limit / timeout**: up to `LLM_MAX_RETRIES` attempts with a growing pause, then fallback.
- **Malformed or ungrounded AI answer**: that answer (or just that one explanation) is replaced by the catalog text.
  Say: "The AI must quote evidence from the record and may not add actors, awards, years or names that are not in the catalog."

## Bonus points to mention
- Streaming: the AI text appears live while it is being written.
- Index safety: editing a CSV or changing the embedding model automatically rebuilds the index.
- Honest limits: tiny catalog, illustrative ratings, no conversation memory, the fact check is a heuristic.
