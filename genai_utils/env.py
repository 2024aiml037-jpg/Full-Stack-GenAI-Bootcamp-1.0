"""Environment and API key handling shared by the class notebooks and apps."""

from __future__ import annotations

import os
from collections.abc import Iterable

PROVIDER_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "pinecone": "PINECONE_API_KEY",
    "qdrant": "QDRANT_API_KEY",
    "llamacloud": "LLAMA_CLOUD_API_KEY",
}


def load_env(dotenv_path: str | None = None) -> None:
    """Load a .env file into ``os.environ`` if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path)


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"{name} is missing. Add it to your .env file (see .env.example).")
    return value


def get_api_key(provider: str, required: bool = True) -> str | None:
    """Return the API key for ``provider`` ("openai", "google", "pinecone", ...)."""
    key_name = PROVIDER_KEY_NAMES.get(provider.lower())
    if key_name is None:
        raise ValueError(
            f"Unknown provider {provider!r}. Known providers: {sorted(PROVIDER_KEY_NAMES)}"
        )
    return get_env(key_name, required=required)


def require_env(names: Iterable[str]) -> dict[str, str]:
    """Validate that every variable in ``names`` is set and return them."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}
