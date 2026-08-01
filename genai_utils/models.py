"""Factories for chat and embedding models used across the classes.

Third party imports are done lazily so that importing :mod:`genai_utils` only
requires the integrations actually used by a given notebook.
"""

from __future__ import annotations

from typing import Any

DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3",
}

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-large",
    "google": "gemini-embedding-001",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}


def _missing(provider: str, package: str) -> RuntimeError:
    return RuntimeError(f"Install {package} to use the {provider} provider (pip install {package}).")


def get_chat_model(provider: str = "google", model: str | None = None, temperature: float = 0.0, **kwargs: Any):
    """Return a LangChain chat model for ``provider``."""
    provider = provider.lower()
    if provider not in DEFAULT_CHAT_MODELS:
        raise ValueError(f"Unknown chat provider {provider!r}. Known: {sorted(DEFAULT_CHAT_MODELS)}")
    model = model or DEFAULT_CHAT_MODELS[provider]

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise _missing(provider, "langchain-openai") from exc
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise _missing(provider, "langchain-anthropic") from exc
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise _missing(provider, "langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise _missing(provider, "langchain-groq") from exc
        return ChatGroq(model=model, temperature=temperature, **kwargs)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise _missing(provider, "langchain-ollama") from exc
        return ChatOllama(model=model, temperature=temperature, **kwargs)
    raise AssertionError(f"Unhandled chat provider {provider!r}")


def get_embeddings(provider: str = "google", model: str | None = None, **kwargs: Any):
    """Return a LangChain embedding model for ``provider``."""
    provider = provider.lower()
    if provider not in DEFAULT_EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding provider {provider!r}. Known: {sorted(DEFAULT_EMBEDDING_MODELS)}"
        )
    model = model or DEFAULT_EMBEDDING_MODELS[provider]

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise _missing(provider, "langchain-openai") from exc
        return OpenAIEmbeddings(model=model, **kwargs)
    if provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise _missing(provider, "langchain-google-genai") from exc
        return GoogleGenerativeAIEmbeddings(model=model, **kwargs)
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise _missing(provider, "langchain-huggingface") from exc
        return HuggingFaceEmbeddings(model_name=model, **kwargs)
    raise AssertionError(f"Unhandled embedding provider {provider!r}")


def embedding_dimension(embeddings, probe: str = "dimension check") -> int:
    """Number of dimensions produced by ``embeddings`` (needed to size an index)."""
    return len(embeddings.embed_query(probe))
