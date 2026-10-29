"""Streamlit user interface for the Movie & Book Recommendation Assistant.

Run with:  streamlit run app.py
"""
import html

import streamlit as st

from src.data_loader import DatasetError
from src.embeddings import get_embedding_model
from src.llm_service import LLMService, add_explanations
from src.recommender import Filters, Recommender, build_search_query
from src.utils import EXAMPLE_QUERIES, MOODS, clean_query, format_rating, load_settings, validate_query

st.set_page_config(page_title="Movie & Book Recommendation Assistant", page_icon="🎬", layout="wide")

# ---------- Styling (dark purple / navy theme) ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0b1026 0%, #1a1240 55%, #2a1458 100%); color: #ece9f8; }
[data-testid="stSidebar"] { background: #0f1433; border-right: 1px solid #3a2d78; }
h1, h2, h3 { color: #f1ecff !important; }
.hero { padding: 1.2rem 1.5rem; border-radius: 16px; background: rgba(124,77,255,0.15);
        border: 1px solid #5b3fd1; margin-bottom: 1rem; }
.section-title { font-size: 1.3rem; font-weight: 700; margin: 1rem 0 0.4rem 0; padding-bottom: 0.3rem;
                 border-bottom: 2px solid #7c4dff; color: #f1ecff; }
.card { border-radius: 14px; padding: 1rem 1.2rem; margin: 0.6rem 0; background: rgba(20,24,60,0.85);
        border: 1px solid #3a2d78; border-left: 6px solid #7c4dff; }
.card.book { border-left-color: #29b6f6; }
.card-title { font-size: 1.2rem; font-weight: 700; color: #ffffff; }
.meta { color: #b9b0e8; font-size: 0.9rem; margin: 0.2rem 0 0.5rem 0; }
.desc { color: #dcd7f5; margin-bottom: 0.5rem; }
.reason { background: rgba(124,77,255,0.18); border-radius: 10px; padding: 0.6rem 0.8rem; color: #efeaff; }
.badge { display: inline-block; font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; background: #3a2d78;
         color: #d9ceff; margin-right: 6px; }
.stButton > button { border-radius: 10px; border: 1px solid #7c4dff; background: #1a1448; color: #ece9f8; }
.stButton > button:hover { background: #7c4dff; color: white; }
</style>
""", unsafe_allow_html=True)


# ---------- Cached resources (built once, not on every rerun) ----------
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def get_embeddings():
    return get_embedding_model()


@st.cache_resource(show_spinner="Preparing the movie and book indexes...")
def get_recommender(rebuild_count: int) -> Recommender:
    # rebuild_count > 0 means the user pressed "Rebuild indexes".
    return Recommender.from_embeddings(get_embeddings(), force_rebuild=rebuild_count > 0)


@st.cache_resource
def get_llm() -> LLMService:
    settings = load_settings()
    return LLMService(settings["hf_token"], settings["model_name"])


# ---------- Session state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("rebuild_count", 0)
st.session_state.setdefault("pending_query", None)

try:
    recommender = get_recommender(st.session_state.rebuild_count)
except DatasetError as exc:
    st.error(f"Dataset problem: {exc}")
    st.stop()
except Exception as exc:
    st.error("Could not start the recommender. Check your internet connection for the first model download "
             f"and that all packages are installed. Details: {type(exc).__name__}: {exc}")
    st.stop()

llm = get_llm()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    category = st.radio("Recommendation category", ["Movies", "Books", "Both"], index=2)
    movie_genres = st.multiselect("Movie genre", recommender.genres("movies"),
                                  disabled=category == "Books", placeholder="All genres")
    book_genres = st.multiselect("Book genre", recommender.genres("books"),
                                 disabled=category == "Movies", placeholder="All genres")
    languages = st.multiselect("Language", recommender.languages(), placeholder="Any language")
    min_rating = st.slider("Minimum rating (sample ratings, 0 = off)", 0.0, 5.0, 0.0, 0.5)
    top_k = st.slider("Recommendations per category", 1, 10, 5)
    mood = st.selectbox("Mood (optional)", MOODS)
    favorites = st.text_input("Favourite titles (optional)", placeholder="e.g. Inception, The Hobbit")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Rebuild indexes"):
        st.session_state.rebuild_count += 1
        st.rerun()

    st.markdown("---")
    st.subheader("How it works")
    st.caption(
        "1. Your request is turned into an embedding (a vector of numbers).\n\n"
        "2. FAISS finds the movies/books whose embeddings are closest.\n\n"
        "3. Your filters are applied to the retrieved records.\n\n"
        "4. An LLM explains why each record fits (if a Hugging Face token is set). "
        "Otherwise explanations come from catalog metadata.\n\n"
        "Chat history is shown on screen only - each search is independent and the model "
        "does not remember earlier messages.")
    st.caption("Ratings in the sample catalog are illustrative values, not official ratings.")
    st.caption("LLM status: " + (f"enabled ({llm.model_name})" if llm.is_configured else "no token - catalog explanations only"))


# ---------- Rendering helpers ----------
def card_html(item: dict, kind: str) -> str:
    rec = item["record"]
    e = lambda v: html.escape(str(v))
    if kind == "movies":
        year = rec.get("release_year")
        meta = f"{e(rec['genre'])} • {int(year) if year else 'Year n/a'} • {e(rec['language'])}"
        css = "card"
    else:
        year = rec.get("publication_year")
        meta = f"{e(rec['genre'])} • by {e(rec.get('author', 'Unknown'))} • {int(year) if year else 'Year n/a'} • {e(rec['language'])}"
        css = "card book"
    source = "AI explanation" if item["reason_source"] == "llm" else "Catalog-based explanation"
    return (f'<div class="{css}"><div class="card-title">{e(rec["title"])}</div>'
            f'<div class="meta">{meta}</div><div class="desc">{e(rec["description"])}</div>'
            f'<div class="meta">⭐ Sample rating: {e(format_rating(rec.get("rating")))} &nbsp;|&nbsp; '
            f'Match score: {item["similarity"]:.0%}</div>'
            f'<div class="reason"><span class="badge">{source}</span><b>Why it fits:</b> {e(item["reason"])}</div></div>')


def render_assistant(msg: dict):
    for warning in msg.get("warnings", []):
        st.warning(warning)
    total = 0
    titles = {"movies": "🎬 Movie recommendations", "books": "📚 Book recommendations"}
    for kind, items in msg["results"].items():
        st.markdown(f'<div class="section-title">{titles[kind]}</div>', unsafe_allow_html=True)
        if not items:
            st.info(f"No suitable {kind[:-1]} was found in the catalog for this request. "
                    "Try different words or relax the filters.")
        for item in items:
            st.markdown(card_html(item, kind), unsafe_allow_html=True)
        total += len(items)
    if total == 0:
        st.caption("The catalog is small (about 50 movies and 50 books), so some requests have no close match.")


# ---------- Main page ----------
st.title("🎬📚 Movie & Book Recommendation Assistant")
st.markdown('<div class="hero">Welcome! Tell me what you feel like watching or reading - a genre, a mood, a topic or a '
            'favourite title - and I will search the catalog and explain why each pick may suit you.</div>',
            unsafe_allow_html=True)

st.caption("Try an example:")
cols = st.columns(3)
for i, example in enumerate(EXAMPLE_QUERIES):
    if cols[i % 3].button(example, key=f"ex{i}", use_container_width=True):
        st.session_state.pending_query = example

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant(msg)

typed = st.chat_input("Describe what you want, e.g. 'relaxing movies and books for the weekend'")
query = st.session_state.pending_query or typed
st.session_state.pending_query = None

if query:
    ok, problem = validate_query(query)
    if not ok:
        st.warning(problem)
    else:
        query = clean_query(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching the catalog and writing explanations..."):
                try:
                    filters = {"movies": Filters(movie_genres, languages, min_rating),
                               "books": Filters(book_genres, languages, min_rating)}
                    search_text = build_search_query(query, mood, favorites)
                    results = recommender.search(search_text, category, top_k, filters)
                    warnings = add_explanations(query, results, llm)
                    msg = {"role": "assistant", "warnings": warnings, "results": {
                        kind: [{"record": r.record, "similarity": r.similarity,
                                "reason": r.reason, "reason_source": r.reason_source} for r in recs]
                        for kind, recs in results.items()}}
                except Exception as exc:
                    msg = {"role": "assistant", "results": {},
                           "warnings": [f"Something went wrong while searching: {type(exc).__name__}: {exc}"]}
            render_assistant(msg)
        st.session_state.messages.append(msg)
