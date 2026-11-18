
"""Small helper functions and constants shared by the whole project."""

import hashlib
import math
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


# ---------- Project paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"


# ---------- Model configuration ----------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"

MAX_QUERY_LENGTH = 300
DEFAULT_MAX_RETRIES = 3


# ---------- Rating and year limits ----------

MIN_RATING = 0.0
MAX_RATING = 5.0

MIN_YEAR = {
    "movies": 1888,
    "books": 1000
}

MAX_YEAR = date.today().year + 2


# ---------- Example queries ----------

EXAMPLE_QUERIES = [
    "Recommend mind-bending science fiction movies.",
    "Suggest inspiring books for personal growth.",
    "I want mystery movies with unexpected endings.",
    "Recommend beginner-friendly books about Python programming.",
    "Suggest fantasy movies and adventure novels.",
    "Recommend movies and books suitable for a relaxing weekend.",
]


# ---------- Available moods ----------

MOODS = [
    "Any",
    "Relaxing",
    "Inspiring",
    "Thrilling",
    "Funny",
    "Emotional",
    "Thought-provoking"
]


# ---------- Load application settings ----------

def load_settings() -> dict:
    """
    Load Groq configuration from the project's .env file.

    Supports both uppercase and lowercase configuration keys
    for compatibility with the application.
    """

    # Load .env from the project root
    load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True
)

    # Read Groq API key
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Handle placeholder or missing API keys
    invalid_keys = {
        "",
        "your_groq_api_key",
        "your_groq_api_key_here",
        "your_api_key"
    }

    if api_key.lower() in invalid_keys:
        api_key = ""

    # Read model name
    model_name = (
        os.getenv("MODEL_NAME", "").strip()
        or DEFAULT_MODEL_NAME
    )

    # Read maximum retries safely
    try:
        retries = int(
            os.getenv(
                "LLM_MAX_RETRIES",
                str(DEFAULT_MAX_RETRIES)
            )
        )

        retries = max(1, min(retries, 6))

    except (ValueError, TypeError):
        retries = DEFAULT_MAX_RETRIES

    # Return both uppercase and lowercase keys
    # This prevents KeyError in existing application code.

    return {
        "GROQ_API_KEY": api_key,
        "MODEL_NAME": model_name,
        "max_retries": retries,

        # Backward compatibility
        "groq_api_key": api_key,
        "model_name": model_name,
        "LLM_MAX_RETRIES": retries
    }


# ---------- Clean user queries ----------

def clean_query(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text).strip()

    return text[:MAX_QUERY_LENGTH]


# ---------- Validate user queries ----------

def validate_query(text: str) -> tuple[bool, str]:

    cleaned = clean_query(text)

    if not cleaned:
        return False, "Please enter what you want to watch or read."

    if not re.search(r"[A-Za-z0-9]", cleaned):
        return False, "Please enter readable words."

    if len(cleaned) < 3:
        return False, "Your request is too short."

    return True, ""


# ---------- Tokenize text ----------

def tokenize(text: str) -> set[str]:

    return set(
        re.findall(r"[a-z0-9]+", str(text).lower())
    )


# ---------- Check numeric values ----------

def is_finite_number(value) -> bool:

    try:
        return (
            value is not None
            and math.isfinite(float(value))
        )

    except (TypeError, ValueError, OverflowError):
        return False


# ---------- Format year ----------

def format_year(value) -> str:

    if not is_finite_number(value):
        return "Year n/a"

    year = float(value)

    if year != int(year) or not (1 <= year <= MAX_YEAR):
        return "Year n/a"

    return str(int(year))


# ---------- Generate file fingerprint ----------

def file_fingerprint(path: Path) -> str:

    h = hashlib.md5()

    with open(path, "rb") as f:
        h.update(f.read())

    return h.hexdigest()


# ---------- Format rating ----------

def format_rating(rating) -> str:

    if not is_finite_number(rating):
        return "Not available"

    return f"{float(rating):.1f} / 5"
