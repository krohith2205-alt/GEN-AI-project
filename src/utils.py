"""Small helper functions and constants shared by the whole project."""
import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"          # FAISS indexes are saved here
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_QUERY_LENGTH = 300

EXAMPLE_QUERIES = [
    "Recommend mind-bending science fiction movies.",
    "Suggest inspiring books for personal growth.",
    "I want mystery movies with unexpected endings.",
    "Recommend beginner-friendly books about Python programming.",
    "Suggest fantasy movies and adventure novels.",
    "Recommend movies and books suitable for a relaxing weekend.",
]

MOODS = ["Any", "Relaxing", "Inspiring", "Thrilling", "Funny", "Emotional", "Thought-provoking"]


def load_settings() -> dict:
    """Read HF_TOKEN and MODEL_NAME from the .env file / environment.

    The token is optional: without it the app still works using
    catalog-only (fallback) explanations.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN", "").strip()
    if token.lower() in {"", "your_huggingface_token_here"}:
        token = ""
    model = os.getenv("MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME
    return {"hf_token": token, "model_name": model}


def clean_query(text: str) -> str:
    """Trim whitespace, collapse spaces and limit the length of user input."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_QUERY_LENGTH]


def validate_query(text: str) -> tuple[bool, str]:
    """Return (is_valid, message). Rejects empty or meaningless input."""
    cleaned = clean_query(text)
    if not cleaned:
        return False, "Please type what you are looking for, for example: 'mind-bending science fiction movies'."
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return False, "Your message does not contain any readable words. Please try again."
    if len(cleaned) < 3:
        return False, "Your request is too short. Please describe what you want in a few words."
    return True, ""


def tokenize(text: str) -> set[str]:
    """Lower-case word set, used for the simple keyword-overlap explanations."""
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def file_fingerprint(path: Path) -> str:
    """MD5 hash of a file. If the CSV changes, the hash changes and the index is rebuilt."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def format_rating(rating) -> str:
    """Show a rating like '4.5 / 5', or 'Not available'."""
    try:
        if rating is None or rating != rating:   # None or NaN
            return "Not available"
        return f"{float(rating):.1f} / 5"
    except (TypeError, ValueError):
        return "Not available"
