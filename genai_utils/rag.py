"""Document loading, chunking and retrieval helpers shared by the RAG classes."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

RAG_PROMPT_TEMPLATE = """You are a question-answering assistant.

Answer the question only from the supplied context.
If the answer is not available in the context, say:
"I do not have enough information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


def load_pdf(file_path: str | Path) -> list:
    """Load a PDF into one LangChain ``Document`` per page."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def split_documents(
    documents: Sequence,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] | None = None,
) -> list:
    """Split documents with the recursive character splitter used in class."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators is not None else list(DEFAULT_SEPARATORS),
    )
    return splitter.split_documents(list(documents))


def load_and_split_pdf(file_path: str | Path, **split_kwargs: Any) -> list:
    """``load_pdf`` followed by ``split_documents``."""
    return split_documents(load_pdf(file_path), **split_kwargs)


def add_chunk_metadata(chunks: Iterable, **extra: Any) -> list:
    """Tag each chunk with its index plus any additional metadata."""
    tagged = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata.update(extra)
        tagged.append(chunk)
    return tagged


def make_retriever(vector_store, k: int = 5, search_type: str = "similarity", **search_kwargs: Any):
    """Build a retriever from any LangChain vector store."""
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, **search_kwargs},
    )


def format_docs(docs: Iterable, with_metadata: bool = False) -> str:
    """Join retrieved documents into a single context string."""
    if with_metadata:
        return "\n\n".join(f"Source: {doc.metadata}\n{doc.page_content}" for doc in docs)
    return "\n\n".join(doc.page_content for doc in docs)


def print_documents(docs: Iterable, max_chars: int = 500) -> None:
    """Pretty-print retrieval results, as done in the vector database classes."""
    for rank, doc in enumerate(docs, start=1):
        print(f"\n--- Result {rank} ---")
        print(doc.page_content[:max_chars])
        print("Metadata:", doc.metadata)


def build_rag_chain(retriever, model, template: str = RAG_PROMPT_TEMPLATE, with_metadata: bool = False):
    """Wire retriever -> prompt -> chat model -> string output."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    prompt = ChatPromptTemplate.from_template(template)

    def _format(docs):
        return format_docs(docs, with_metadata=with_metadata)

    return (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
