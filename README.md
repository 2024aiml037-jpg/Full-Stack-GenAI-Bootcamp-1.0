# Full-Stack-GenAI-Bootcamp-1.0

### 👨‍🏫 Meet Your Mentor

🔗 [Visit Mentor Profile](https://sunnysavita10.github.io/)

## Shared utilities (`genai_utils`)

Boilerplate that repeats across the class notebooks and apps — loading API keys,
building chat/embedding models, PDF loading + chunking, retrievers and RAG chains,
vector store creation, and 4-bit Hugging Face loading/generation — lives in the
`genai_utils` package at the repository root.

```python
from genai_utils import load_env, get_chat_model, get_embeddings, load_and_split_pdf
from genai_utils import build_faiss_store, add_documents, make_retriever, build_rag_chain

load_env()
embeddings = get_embeddings("google")
chunks = load_and_split_pdf("data/llama2-research-paper.pdf", chunk_size=2000, chunk_overlap=200)

store = build_faiss_store(embeddings, index_type="hnsw")
add_documents(store, chunks)

rag_chain = build_rag_chain(make_retriever(store, k=5), get_chat_model("google"))
print(rag_chain.invoke("What is the architecture of Llama 2?"))
```

Third-party integrations are imported lazily, so only the libraries a given
notebook actually uses need to be installed.

From a notebook inside a class folder, add the repository root to `sys.path` first:

```python
import sys, pathlib
sys.path.append(str(next(p for p in pathlib.Path.cwd().parents if (p / "genai_utils").exists())))
```

Run the unit tests with:

```bash
pip install pytest
python -m pytest tests
```