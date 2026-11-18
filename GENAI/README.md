# Movie & Book Recommendation Assistant (Generative AI + RAG)

A college project. Users describe what they feel like watching or reading, and the app recommends movies and books
from a local catalog, with a fact-checked explanation for each recommendation.

**Live demo / recorded demo:** `ADD YOUR LINK HERE` (see section 14 - "Deployment" - for how to create it)

## 1. Problem statement
Choosing a movie or book is hard: keyword search only finds exact words, and chatbots may invent titles that do not exist.
We need a system that understands the *meaning* of a request ("something relaxing for the weekend") and recommends only
real items from a known catalog.

## 2. Objectives
- Demonstrate Retrieval-Augmented Generation (RAG) with semantic search, embeddings, FAISS and an LLM.
- Recommend only items that exist in the catalog (no invented titles or metadata).
- Keep AI-written explanations grounded in the catalog (evidence + fact check) and fall back safely when they are not.
- Work on an ordinary laptop, without a GPU, paid database or paid API.

## 3. Features
- Movies, Books or Both; genre, language and minimum-rating filters; optional mood and favourite titles.
- One **preference context** (request + mood + favourites) is used for BOTH retrieval and the explanation writer.
- Cards with title, genre, author/year, description, sample rating, **semantic similarity** and a reason.
- LLM explanations via Hugging Face, **streamed** live, with **evidence quotes** and a **fact check**;
  automatic fallback to catalog-based explanations (no token, API error, malformed or ungrounded answer).
- Fallback wording depends on match strength (strong / moderate / weak).
- Bounded **retries** for temporary API failures (rate limit, 5xx, timeout); authentication/model errors are never retried.
- Validated catalog numbers (years, ratings, ids); safe rendering of missing values.
- Index saved with a manifest (CSV fingerprint + embedding model + settings); automatic rebuild when anything changes,
  including edits made while the app is running.
- Included datasets: 53 movies and 52 books.

## 4. Architecture and workflow
```
CSV files -> data_loader (validate + Documents) -> embeddings (MiniLM) -> FAISS index (movies / books) + manifest.json
Preferences(request, mood, favourites) -> search_text -> embed -> FAISS search -> filters + threshold -> top-k records
Preferences + top-k records (+ few-shot example) -> LLM (streamed, retried) -> JSON {explanation, evidence}
   -> parse (type-checked) -> evidence check -> unsupported-claim check -> accepted explanations
   anything rejected / missing / failing -> catalog-based explanation (wording by match strength)
-> Streamlit cards
```
1. Each CSV row becomes a text document with the full record stored as metadata.
2. Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2`; movies and books have separate FAISS indexes.
3. The preferences are embedded and the closest documents are retrieved; filters and the similarity threshold are applied.
4. Only the retrieved records and the same preferences are sent to the LLM.
5. Each AI explanation must quote evidence from the record and pass the fact check; otherwise it is replaced.
6. Titles, years, authors and ratings are always shown **from the CSV**; the LLM only supplies commentary.

## 5. What is RAG?
Retrieval-Augmented Generation = **retrieve** relevant facts first, then let an LLM **generate** an answer using only
those facts. This reduces hallucination, but does not remove it - so this project also checks the generated text.

## 6. Embeddings, FAISS and the match score
- An *embedding* converts text into numbers (384 for MiniLM); similar meaning gives nearby vectors.
- *FAISS* finds the closest vectors. Vectors are normalised, so squared-L2 distance `d` converts to cosine
  similarity as `1 - d/2`.
- The percentage on each card is **semantic similarity between your request and the catalog text. It is not a probability
  that you will enjoy the item.**
- Thresholds (`src/recommender.py`): below `MIN_SIMILARITY` an item is not shown; below `MODERATE_SIMILARITY` it is
  explained as a *weak* match. The defaults (0.15 / 0.30 / 0.45) are starting values: run
  `python evaluation/threshold_report.py` with the real model and adjust them (see section 12).

## 7. The LLM and how explanations are checked
A configurable Hugging Face instruction model (default `Qwen/Qwen2.5-1.5B-Instruct`) is called through
`huggingface_hub.InferenceClient` with `stream=True`. The prompt contains the rules, **one few-shot example** of the
answer format, the user's preferences, and only the retrieved records. The model must answer with JSON:
`{"reasons": {"<id>": {"explanation": "...", "evidence": [{"field": "keywords", "text": "dreams"}]}}}`

Checks, in order (each failure only affects what it must):
1. **Structure** - `parse_reasons()` checks the type of every level; `null`, lists, missing fields, empty text -> `{}` -> fallback.
2. **Valid ids** - ids that were not retrieved are ignored.
3. **Evidence** - every quote must appear word-for-word in the named catalog field.
4. **Unsupported claims** - an explanation is rejected if it names people/places/titles, numbers, awards or cast/crew
   claims that are not in the record text or the user's own words.
   *This is a heuristic* (a name that is the first word of a sentence is not checked); the evaluation therefore also
   asks you to judge explanations manually.
5. Rejected or missing explanations are replaced by the catalog-based explanation, and the card says which kind it is.

Retries: rate limit (429), server errors (5xx), timeouts and connection errors are retried up to `LLM_MAX_RETRIES`
attempts in total (default 3) with 1 s, 2 s, ... pauses. Wrong/expired token (401/403), unavailable model (404) and used-up
credits (402) fail immediately.
Chat history is displayed on screen only; it is **not** sent to the model (no conversational memory).

## 8. Folder structure
```
GENAI/
    app.py                 Streamlit UI + session state
    requirements.txt  requirements-tested.txt  README.md  DEMO_GUIDE.md  .env.example  .gitignore
    .streamlit/config.toml dark theme colours
    data/movies.csv  data/books.csv
    src/
        data_loader.py     load/validate CSV (ids, years, ratings), build Documents
        embeddings.py      embedding model, FAISS build/load, manifest + dataset fingerprint
        recommender.py     Preferences, retrieval, filters, thresholds, fallback reasons
        llm_service.py     prompt, streaming, retries, parsing, fact check
        utils.py           settings, input validation, number/year formatting
    tests/                 97 unit / regression tests (offline)
    evaluation/            eval_cases.json, run_evaluation.py, threshold_report.py (+ results you generate)
```
`index/` (the saved FAISS indexes) is created automatically on first run and is not committed to Git.

## 9. Prerequisites
- Python **3.10 - 3.12** (check with `python --version`). Tested on 3.12.3.
- Internet for the first run (packages and the ~90 MB embedding model).
- Optional: a free Hugging Face account for LLM explanations.

## 10. Installation
The project folder is called **`GENAI`** (the repository name). If you cloned it under another name, use that folder
name instead in the `cd` command.
```
git clone <your-repository-url> GENAI
cd GENAI
python -m venv .venv
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (macOS / Linux)
pip install -r requirements.txt
```
PowerShell may block activation scripts; if so run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
Command Prompt. Installation is large (PyTorch comes with sentence-transformers) and can take several minutes.

### Tested versions
| Package | Version used for the automated tests |
|---|---|
| Python | 3.12.3 (Linux) |
| streamlit | 1.67.0 |
| pandas / numpy | 3.0.6 / 2.5.4 |
| python-dotenv | 1.2.4 |
| langchain-core / -community / -text-splitters | 1.7.6 / 0.4.2 / 1.1.3 |
| faiss-cpu | 1.15.1 |
| huggingface_hub | 2.4.0 |
| pytest | 9.2.0 |
| langchain-huggingface, sentence-transformers | **not verified** in the test environment (no model download there) - `requirements.txt` ranges apply |

`pip install -r requirements-tested.txt` reproduces the verified set. The 97 tests use offline stand-ins for the
embedding model and the LLM (plus a real FAISS index), and the app was also started headlessly with Streamlit's test
runner. The real-model behaviour is covered by the evaluation in section 12, which you must run yourself.
Windows was not part of that test run.

## 11. Hugging Face token (optional)
1. Create an account at https://huggingface.co and open **Settings -> Access Tokens**.
2. Create a token that can call Inference Providers.
3. Copy `.env.example` to `.env` (`copy .env.example .env` on Windows, `cp .env.example .env` elsewhere) and edit it:
```
HF_TOKEN=hf_your_real_token
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
LLM_MAX_RETRIES=3
```
Never share `.env` or commit it. Hosted inference is not guaranteed to be free or always available. Without a token the
app still works with catalog-based explanations.

## 12. Run, test and evaluate
```
streamlit run app.py
python -m pytest                                   # 97 offline tests
python evaluation/threshold_report.py              # real model: see real similarity scores, tune thresholds
python evaluation/run_evaluation.py                # real model + real LLM: 17 evaluation cases
```
`run_evaluation.py` writes `evaluation/results.csv` and `evaluation/EVALUATION_REPORT.md` with the expected
recommendations, the actual results, automatic checks (filters, unknown titles, unsupported claims, API behaviour) and
**empty columns for your own relevance judgments (0/1/2) and unsupported-claim review**. Fill those in and submit the files.
The cases cover on-topic requests, mood + favourites, restrictive filters (including an impossible combination), an unknown
favourite title, an unrelated request, a prompt-injection attempt, and API failures (invalid token, unavailable model,
simulated outage with retries, no token). `--dry-run` only tests the script itself with stand-in embeddings; its output is
marked as not real and must not be submitted as results.

## 13. Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` not recognised | Activate the virtual environment |
| `faiss` install fails | Use Python 3.10-3.12 (64-bit) and upgrade pip: `python -m pip install --upgrade pip` |
| Model download fails | Check internet/proxy; run again |
| "token was rejected" | Re-check `HF_TOKEN` in `.env`, restart the app |
| "model is not available" | Change `MODEL_NAME` in `.env` to another instruct model your provider supports |
| "temporary network or server problem" | The app already retried; wait and try again |
| AI explanations never appear | Expected if the model's answers fail the evidence / fact check - the card then says "Catalog-based explanation" |
| Catalog changed | Indexes rebuild automatically (the CSV fingerprint is part of the cache key). **Rebuild indexes** forces it |
| "Data quality notes" in the sidebar | Some CSV rows had invalid ids, years or ratings; fix the CSV |

## 14. Deployment (Streamlit Community Cloud) / recorded demo
1. Push the project to GitHub (repository `GENAI`; `.env` and `index/` are ignored by `.gitignore`).
2. On https://share.streamlit.io choose **New app**, select the repository, branch and `app.py`.
3. Under **Advanced settings -> Secrets** add `HF_TOKEN = "hf_..."` and optionally `MODEL_NAME`. Top-level secrets are
   available to the app as environment variables, which `load_settings()` reads.
4. Put the app URL in the "Live demo" line at the top of this README. If you cannot deploy, record a screen demo
   following `DEMO_GUIDE.md` and link the video instead.

## 15. Known limitations
- The catalog has only ~50 movies and ~50 books.
- **Ratings are illustrative sample values (1-5), not official ratings.** Descriptions are short original summaries.
  Check years, authors and languages before relying on them.
- The fact check is a heuristic: it catches invented names, numbers, awards and cast/crew claims, but cannot verify
  every possible false statement (e.g. a wrong plot detail written with ordinary words). Review explanations manually.
- A small LLM often fails the evidence format; then you see catalog-based explanations (this is the safe outcome).
- Semantic search can miss very specific requests (for example an exact actor name).
- No personalised profiles or conversational memory.
- The similarity thresholds are starting values until you tune them with `threshold_report.py` on your machine.

## 16. Future enhancements
Larger datasets (TMDB / Open Library APIs), hybrid keyword + semantic search, user feedback, conversation memory,
posters/covers.

## 17. Simple explanation for your professor
"My project is a recommendation assistant. I store movies and books in CSV files and convert each description into an
embedding. A user's request, mood and favourite titles become one preference text that is embedded too, and FAISS finds
the closest items. Items that are too weak are rejected. Only the retrieved items and the same preferences go to a
language model that explains each match. The model must quote evidence from the catalog and cannot add actors, awards or
numbers that are not there; if it fails, is unavailable or is rejected by my checks, the app shows explanations built
from the catalog instead. Titles, years and ratings always come from the CSV."

## 18. What changed after the review
| Review point | Change |
|---|---|
| 1 JSON validation | `parse_reasons()` type-checks every level; malformed output -> `{}` -> fallback; tests for null, list, missing fields, empty text |
| 2 Grounding | evidence-linked JSON, evidence quote check, unsupported-claim check, per-explanation rejection |
| 3 Preference context | `Preferences` object used for retrieval **and** the explanation prompt |
| 4 Fallback wording | strong / moderate / weak wording; label "Semantic similarity"; `threshold_report.py` to tune thresholds |
| 5 Numeric fields | years and ratings validated on load (ranges, finite); missing ids rejected; `format_year()` renders safely |
| 6 Index refresh | dataset fingerprint + model name in the Streamlit cache key; `manifest.json` with CSV hash, model, settings |
| 7 Course deliverables | few-shot example, streaming, bounded retries (auth never retried), 17-case evaluation kit |
| Docs / structure | `.env.example` (with `LLM_MAX_RETRIES`), `GENAI` directory, tested versions, regression tests, deployment/demo guide |
