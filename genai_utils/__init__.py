"""Shared utilities for the Full-Stack GenAI Bootcamp notebooks and apps.

Usage from a notebook that lives in a class folder::

    import sys, pathlib
    sys.path.append(str(next(p for p in pathlib.Path.cwd().parents if (p / "genai_utils").exists())))

    from genai_utils import load_env, get_embeddings, load_and_split_pdf

Third-party integrations are imported lazily, so only the libraries a given
notebook actually uses need to be installed.
"""

from .env import get_api_key, get_env, load_env, require_env
from .hf import (
    benchmark_generation,
    four_bit_config,
    generate_text,
    load_causal_lm,
    load_model_and_tokenizer,
    load_tokenizer,
)
from .models import embedding_dimension, get_chat_model, get_embeddings
from .multimodal import (
    build_human_message,
    build_message_content,
    encode_bytes_to_base64,
    response_text,
)
from .rag import (
    RAG_PROMPT_TEMPLATE,
    add_chunk_metadata,
    build_rag_chain,
    format_docs,
    load_and_split_pdf,
    load_pdf,
    make_retriever,
    print_documents,
    split_documents,
)
from .vectorstores import (
    add_documents,
    build_chroma_store,
    build_faiss_store,
    build_qdrant_store,
)

__all__ = [
    "RAG_PROMPT_TEMPLATE",
    "add_chunk_metadata",
    "add_documents",
    "benchmark_generation",
    "build_chroma_store",
    "build_faiss_store",
    "build_human_message",
    "build_message_content",
    "build_qdrant_store",
    "build_rag_chain",
    "embedding_dimension",
    "encode_bytes_to_base64",
    "format_docs",
    "four_bit_config",
    "generate_text",
    "get_api_key",
    "get_chat_model",
    "get_embeddings",
    "get_env",
    "load_and_split_pdf",
    "load_causal_lm",
    "load_env",
    "load_model_and_tokenizer",
    "load_pdf",
    "load_tokenizer",
    "make_retriever",
    "print_documents",
    "require_env",
    "response_text",
    "split_documents",
]
