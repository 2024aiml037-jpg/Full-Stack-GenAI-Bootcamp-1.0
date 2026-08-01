"""Vector store constructors shared by the FAISS / Chroma / Qdrant classes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import embedding_dimension


def build_faiss_store(embeddings, index_type: str = "flat", m: int = 32, ef_construction: int = 200, ef_search: int = 64):
    """Create an empty FAISS store sized from the embedding model.

    ``index_type`` is either ``"flat"`` (exact search) or ``"hnsw"`` (approximate).
    """
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    dimension = embedding_dimension(embeddings)
    if index_type == "flat":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unknown index_type {index_type!r}; use 'flat' or 'hnsw'.")

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False,
    )


def build_chroma_store(embeddings, collection_name: str, persist_directory: str | None = None, space: str = "cosine"):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": space},
    )


def build_qdrant_store(embeddings, collection_name: str, client=None, url: str | None = None, api_key: str | None = None):
    """Create (if needed) a Qdrant collection and return its LangChain store."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models

    if client is None:
        client = QdrantClient(url=url, api_key=api_key)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dimension(embeddings),
                distance=models.Distance.COSINE,
            ),
        )

    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)


def add_documents(vector_store, chunks: Sequence, **kwargs: Any) -> list:
    """Insert chunks and report how many vectors were written."""
    ids = vector_store.add_documents(documents=list(chunks), **kwargs)
    print("Documents added:", len(ids))
    return ids
