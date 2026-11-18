
"""Groq-powered explanation service for movie and book recommendations."""

import time

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
)

from .recommender import fallback_reason


# ---------- Groq configuration ----------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMService:

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3
    ):

        self.model_name = (model_name or DEFAULT_MODEL).strip()

        self.max_retries = max(
            1,
            min(int(max_retries), 6)
        )

        self.is_configured = bool(
            api_key and api_key.strip()
        )

        self.client = None

        if self.is_configured:

            self.client = OpenAI(
                api_key=api_key.strip(),
                base_url=GROQ_BASE_URL,
                max_retries=0,
                timeout=60.0
            )

    # ---------- Generate AI explanation ----------

    def generate_explanation(
        self,
        preferences,
        recommendation,
        on_token=None
    ) -> str:

        if not self.is_configured or self.client is None:
            raise ValueError(
                "Groq API is not configured. Check your GROQ_API_KEY."
            )

        rec = recommendation.record

        kind = (
            "movie"
            if recommendation.kind == "movies"
            else "book"
        )

        details = f"""
Title: {rec.get('title', 'Unknown')}
Type: {kind}
Genre: {rec.get('genre', 'Unknown')}
Description: {rec.get('description', 'Not available')}
Language: {rec.get('language', 'Unknown')}
Rating: {rec.get('rating', 'Not available')}
Author: {rec.get('author', 'Not applicable')}
Similarity: {recommendation.similarity:.2f}
"""

        system_prompt = """
You are an intelligent movie and book recommendation assistant.

Your job is to explain why a particular recommendation
may match the user's preferences.

IMPORTANT RULES:

1. Use only the supplied catalog information.
2. Never invent plot details, authors, awards or facts.
3. Consider the user's mood and favourite titles.
4. Explain the connection to the user's request.
5. Be honest if the match is weak.
6. Write naturally in 2 to 4 sentences.
7. Do not repeat the entire catalog entry.
8. Do not include unnecessary introductions.
9. Give a clear and useful explanation.
"""

        user_prompt = f"""
USER PREFERENCES:

{preferences.prompt_block()}

RECOMMENDATION DETAILS:

{details}

Explain why this recommendation fits the user's preferences.
"""

        # ---------- API request with retry handling ----------

        for attempt in range(self.max_retries):

            try:

                stream = self.client.chat.completions.create(

                    model=self.model_name,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    # GPT-OSS reasoning configuration
                    reasoning_effort="low",

                    max_completion_tokens=600,

                    stream=True
                )

                explanation = ""

                for chunk in stream:

                    if not chunk.choices:
                        continue

                    piece = chunk.choices[0].delta.content

                    if piece:

                        explanation += piece

                        if on_token:
                            on_token(piece)

                explanation = explanation.strip()

                if not explanation:

                    raise ValueError(
                        "Groq returned an empty explanation. "
                        "Try increasing max_completion_tokens."
                    )

                return explanation

            # ---------- API errors ----------

            except APIStatusError as exc:

                status_code = exc.status_code

                # Model not found or unavailable
                if status_code == 404:

                    raise RuntimeError(
                        f"Groq model '{self.model_name}' was not found. "
                        "Check MODEL_NAME in your .env file. "
                        "Recommended model: openai/gpt-oss-120b. "
                        "Check available models at "
                        "https://console.groq.com/docs/models"
                    ) from exc

                # Invalid API key
                if status_code == 401:

                    raise RuntimeError(
                        "Groq authentication failed. "
                        "Check your GROQ_API_KEY in the .env file."
                    ) from exc

                # Permission issue
                if status_code == 403:

                    raise RuntimeError(
                        "Groq access denied. Check your API key "
                        "and model permissions."
                    ) from exc

                # Other client errors should not be retried
                if status_code < 500 and status_code != 429:

                    raise RuntimeError(
                        f"Groq API rejected the request "
                        f"(HTTP {status_code}): {exc}"
                    ) from exc

                # Retry temporary server errors and rate limits
                if attempt == self.max_retries - 1:
                    raise

                if on_token:
                    on_token(None)

                time.sleep(min(2 ** attempt, 8))

            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError
            ) as exc:

                if attempt == self.max_retries - 1:
                    raise

                if on_token:
                    on_token(None)

                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            "Groq request failed after all retry attempts."
        )


# ---------- Add explanations to recommendations ----------

def add_explanations(
    preferences,
    results: dict,
    llm: LLMService,
    on_token=None
) -> list:

    warnings = []

    for kind, recommendations in results.items():

        for recommendation in recommendations:

            # Use catalog fallback if API is not configured
            if not llm.is_configured:

                recommendation.reason = fallback_reason(
                    recommendation,
                    preferences
                )

                recommendation.reason_source = "catalog"

                continue

            try:

                explanation = llm.generate_explanation(
                    preferences,
                    recommendation,
                    on_token
                )

                recommendation.reason = explanation

                recommendation.reason_source = "llm"

            except Exception as exc:

                # Preserve recommendation even if Groq fails
                recommendation.reason = fallback_reason(
                    recommendation,
                    preferences
                )

                recommendation.reason_source = "catalog"

                warning = (
                    "Groq API could not generate all explanations. "
                    "Catalog-based explanations were used instead. "
                    f"Details: {type(exc).__name__}: {exc}"
                )

                if warning not in warnings:
                    warnings.append(warning)

    return warnings

