# Movie & Book Recommendation Assistant (Generative AI + RAG)

A beginner-friendly college project. Users describe what they feel like watching or reading, and the app
recommends movies and books from a local catalog, with an explanation for each recommendation.

## 1. Problem statement
Choosing a movie or book is hard: keyword search only finds exact words, and chatbots may invent titles that do not exist.
We need a system that understands the *meaning* of a request ("something relaxing for the weekend") and recommends only real items from a known catalog.

## 2. Objectives
- Demonstrate Retrieval-Augmented Generation (RAG) with semantic search, embeddings, FAISS and an LLM.
- Recommend only items that exist in the catalog (no invented titles or metadata).
- Work on an ordinary Windows laptop, without a GPU, paid database or paid API.
- Keep the code modular and easy to explain.

## 3. Features
- Movies, Books or Both; genre, language and minimum-rating filters; optional mood and favourite titles.
- Configurable number of recommendations (default 5 per category).
- Cards with title, genre, author/year, description, sample rating, match score and reason.
- LLM explanations via Hugging Face; automatic fallback to catalog-based explanations (no token, model unavailable, API error).
- Cached model and indexes; "Rebuild indexes" button; clear-chat button.
- Included datasets: 53 movies and 52 books.

## 4. Architecture and workflow
```
CSV files -> data_loader (clean + Documents) -> embeddings (MiniLM) -> FAISS index (movies / books)
User query -> clean/validate -> embed -> FAISS search -> filters -> top-k records
top-k records + query -> LLM (Hugging Face) -> reasons   (fallback: metadata-based reasons)
-> Streamlit cards
```
1. Each CSV row becomes a text document ("Title... Genre... Description... Keywords...") with the full record stored as metadata.
2. Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in two FAISS indexes (movies and books are kept separate).
3. The user's request (plus optional mood/favourites) is embedded and the closest documents are retrieved.
4. Genre / language / rating filters are applied; weak matches (below a similarity threshold) are dropped.
5. Only the retrieved records are sent to the LLM, which writes a short explanation for each.
6. The app shows titles and metadata **from the CSV**; the LLM only supplies the explanation text.

## 5. What is RAG?
Retrieval-Augmented Generation = **retrieve** relevant facts first, then let an LLM **generate** an answer using only those facts.
This reduces hallucination because the model is given the real catalog records and told not to invent anything.

## 6. Embeddings and FAISS
- An *embedding* converts text into a list of numbers (384 numbers for MiniLM). Texts with similar meaning get nearby vectors, so "mind-bending sci-fi" can match a film described with "dreams, reality, twist".
- *FAISS* (Facebook AI Similarity Search) quickly finds the vectors closest to the query vector.

## 7. The LLM
A configurable Hugging Face instruction model (default `Qwen/Qwen2.5-1.5B-Instruct`) is called through `huggingface_hub.InferenceClient`.
The system prompt tells it to use only the provided records, avoid "best/perfect" claims, admit weak matches, and treat user text and descriptions as data, not instructions.
Not every model is available through every inference provider; if the call fails the app shows a message and uses catalog-based explanations.
Chat history is displayed on screen only; it is **not** sent to the model, so the assistant has no conversational memory.

## 8. Folder structure
```
movie-book-recommendation-assistant/
    app.py                 Streamlit UI + session state
    requirements.txt  README.md  .env.example  .gitignore
    .streamlit/config.toml dark theme colours
    data/movies.csv  data/books.csv
    src/
        data_loader.py     load/validate CSV, build Documents
        embeddings.py      embedding model, build/load FAISS indexes
        recommender.py     retrieval, filters, ranking, fallback reasons
        llm_service.py     prompt, Hugging Face call, error handling
        utils.py           settings, input validation, helpers
    tests/                 unit tests
```
An `index/` folder is created automatically on first run (it stores the FAISS indexes).

## 9. Prerequisites
- Windows 10/11, Python **3.10, 3.11 or 3.12** (check with `python --version`).
- Internet for the first run (downloads packages and the ~90 MB embedding model).
- Optional: a free Hugging Face account for LLM explanations.

## 10. Installation (Windows, Command Prompt or PowerShell)
```
cd movie-book-recommendation-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
PowerShell may block activation scripts; if so run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.
Installation is large (PyTorch is installed by sentence-transformers) and can take several minutes.

## 11. Hugging Face token (optional)
1. Create an account at https://huggingface.co and open **Settings -> Access Tokens**.
2. Create a token with permission to call inference (a fine-grained token with "Make calls to Inference Providers", or a read token).
3. Copy `.env.example` to `.env` (`copy .env.example .env`) and edit it:
```
HF_TOKEN=hf_your_real_token
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```
Never share `.env` or commit it to Git. Hosted inference is not guaranteed to be free or always available: free usage is limited and a model may be unavailable via your provider. Without a token the app still works with catalog-based explanations.

## 12. Run
```
streamlit run app.py
```
The first start downloads the embedding model and builds the indexes (about a minute); later starts are fast.
Run the tests with `python -m pytest` (or `python -m unittest discover -s tests -t .`).

## 13. Example queries
- Recommend mind-bending science fiction movies.
- Suggest inspiring books for personal growth.
- I want mystery movies with unexpected endings.
- Recommend beginner-friendly books about Python programming.
- Suggest fantasy movies and adventure novels.
- Recommend movies and books suitable for a relaxing weekend.

## 14. Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` not recognised | Activate the virtual environment: `.venv\Scripts\activate` |
| `faiss` install fails | Use Python 3.10-3.12 (64-bit) and upgrade pip: `python -m pip install --upgrade pip` |
| Model download fails | Check internet/proxy; run again |
| "token was rejected" | Re-check `HF_TOKEN` in `.env`, restart the app |
| "model is not available" | Change `MODEL_NAME` in `.env` to another instruct model your provider supports |
| Catalog changed but results look old | Press **Rebuild indexes** (indexes also rebuild automatically when a CSV changes) |
| No results | The catalog is small; remove filters or rephrase |

## 15. Known limitations
- The catalog has only ~50 movies and ~50 books.
- **Ratings are illustrative sample values (1-5), not official ratings.** Descriptions are short original summaries, not copied text. Years, authors and languages should be double-checked before you rely on them; the `language` column for books is the book's original language.
- A small LLM may give generic explanations or occasionally fail to follow the JSON format (the app then falls back).
- Semantic search can miss very specific requests (for example an exact actor name).
- No personalised user profiles or conversational memory.

## 16. Future enhancements
Larger datasets (TMDB / Open Library APIs), hybrid keyword + semantic search, user ratings and feedback, conversation memory, posters/covers, deployment on Streamlit Community Cloud.

## 17. Simple explanation for your professor
"My project is a recommendation assistant. First, I store movies and books in CSV files and convert each description into an embedding, a vector that represents its meaning. When a user types a request, I convert it into an embedding as well and use FAISS to find the closest items. These retrieved items, and only these, are given to a language model which explains why they match. This is Retrieval-Augmented Generation: retrieval keeps the answer grounded in real data and generation makes it readable. If the model or token is not available, the app still shows results with explanations built from the catalog metadata. The interface is built with Streamlit."
