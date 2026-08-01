import base64
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genai_utils import env, models, multimodal, rag  # noqa: E402


class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def test_get_env_required_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        env.get_env("OPENAI_API_KEY", required=True)


def test_get_api_key_reads_provider_variable(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pc-123")
    assert env.get_api_key("pinecone") == "pc-123"


def test_get_api_key_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        env.get_api_key("not-a-provider")


def test_require_env_lists_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-1")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        env.require_env(["GOOGLE_API_KEY", "GROQ_API_KEY"])


def test_format_docs_with_and_without_metadata():
    docs = [FakeDoc("alpha", {"page": 1}), FakeDoc("beta", {"page": 2})]
    assert rag.format_docs(docs) == "alpha\n\nbeta"
    assert "Source: {'page': 1}" in rag.format_docs(docs, with_metadata=True)


def test_add_chunk_metadata():
    chunks = [FakeDoc("a"), FakeDoc("b")]
    tagged = rag.add_chunk_metadata(chunks, filename="llama2.pdf")
    assert [c.metadata["chunk_id"] for c in tagged] == [0, 1]
    assert all(c.metadata["filename"] == "llama2.pdf" for c in tagged)


def test_load_pdf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag.load_pdf(tmp_path / "nope.pdf")


def test_rag_prompt_template_has_placeholders():
    assert "{context}" in rag.RAG_PROMPT_TEMPLATE
    assert "{question}" in rag.RAG_PROMPT_TEMPLATE


def test_get_chat_model_unknown_provider():
    with pytest.raises(ValueError, match="Unknown chat provider"):
        models.get_chat_model("nope")


def test_get_embeddings_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        models.get_embeddings("nope")


def test_embedding_dimension():
    class FakeEmbeddings:
        def embed_query(self, text):
            return [0.0] * 768

    assert models.embedding_dimension(FakeEmbeddings()) == 768


def test_build_message_content_text():
    assert multimodal.build_message_content("Text", "hello") == "hello"


def test_build_message_content_image():
    content = multimodal.build_message_content("image", None, data=b"png-bytes", mime_type="image/jpeg")
    assert content[0] == {"type": "text", "text": "Describe this image."}
    expected = base64.b64encode(b"png-bytes").decode()
    assert content[1]["image_url"]["url"] == f"data:image/jpeg;base64,{expected}"


def test_build_message_content_audio_defaults_mime():
    content = multimodal.build_message_content("audio", "transcribe", data=b"wav")
    assert content[1]["mime_type"] == multimodal.DEFAULT_AUDIO_MIME
    assert content[1]["data"] == base64.b64encode(b"wav").decode()


def test_build_message_content_unknown_modality():
    with pytest.raises(ValueError, match="Unknown modality"):
        multimodal.build_message_content("video", "hi", data=b"x")


@pytest.mark.parametrize(
    "response,expected",
    [
        ("plain", "plain"),
        (type("R", (), {"content": "from content"})(), "from content"),
        (type("R", (), {"content": [{"type": "text", "text": "block"}]})(), "block"),
    ],
)
def test_response_text(response, expected):
    assert multimodal.response_text(response) == expected
