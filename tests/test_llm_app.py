import base64

import pytest


class FakeUpload:
    def __init__(self, data: bytes, file_type=None):
        self._data = data
        self.type = file_type

    def read(self) -> bytes:
        return self._data


def test_encode_bytes_to_base64_roundtrip(llm_app):
    encoded = llm_app.encode_bytes_to_base64(b"hello bootcamp")
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded) == b"hello bootcamp"


def test_encode_bytes_to_base64_empty(llm_app):
    assert llm_app.encode_bytes_to_base64(b"") == ""


def test_text_message_uses_prompt_directly(llm_app):
    message = llm_app.build_multimodal_message("Text", "summarize langchain", None)
    assert message.content == "summarize langchain"


def test_image_message_embeds_data_url(llm_app):
    upload = FakeUpload(b"\x89PNG-bytes", "image/jpeg")
    message = llm_app.build_multimodal_message("Image", "what is this?", upload)
    text_block, image_block = message.content
    assert text_block == {"type": "text", "text": "what is this?"}
    expected = llm_app.encode_bytes_to_base64(b"\x89PNG-bytes")
    assert image_block["image_url"]["url"] == f"data:image/jpeg;base64,{expected}"


def test_image_message_defaults(llm_app):
    message = llm_app.build_multimodal_message("Image", None, FakeUpload(b"x", None))
    text_block, image_block = message.content
    assert text_block["text"] == "Describe this image."
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_audio_message_uses_media_block(llm_app):
    message = llm_app.build_multimodal_message("Audio", "", FakeUpload(b"wav", "audio/wav"))
    text_block, media_block = message.content
    assert text_block["text"] == "Transcribe this audio."
    assert media_block == {
        "type": "media",
        "mime_type": "audio/wav",
        "data": llm_app.encode_bytes_to_base64(b"wav"),
    }


def test_audio_message_defaults_mime_type(llm_app):
    message = llm_app.build_multimodal_message("Audio", "transcribe", FakeUpload(b"mp3", None))
    assert message.content[1]["mime_type"] == "audio/mpeg"


@pytest.mark.parametrize(
    ("modality", "prompt", "upload", "expected"),
    [
        ("Image", "prompt only", None, "prompt only"),
        ("Audio", "prompt only", None, "prompt only"),
        ("Video", "prompt only", None, "prompt only"),
        ("Image", None, None, ""),
    ],
)
def test_message_falls_back_to_plain_text(llm_app, modality, prompt, upload, expected):
    assert llm_app.build_multimodal_message(modality, prompt, upload).content == expected


def test_get_llm_openai(llm_app, monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return "openai-llm"

    monkeypatch.setattr(llm_app, "ChatOpenAI", fake_chat_openai)
    assert llm_app.get_llm("OpenAI", "gpt-4o-mini", temperature=0.7) == "openai-llm"
    assert captured == {"model": "gpt-4o-mini", "temperature": 0.7}


def test_get_llm_anthropic_default_temperature(llm_app, monkeypatch):
    captured = {}

    def fake_chat_anthropic(**kwargs):
        captured.update(kwargs)
        return "anthropic-llm"

    monkeypatch.setattr(llm_app, "ChatAnthropic", fake_chat_anthropic)
    assert llm_app.get_llm("Anthropic", "claude-3.5-mini") == "anthropic-llm"
    assert captured == {"model": "claude-3.5-mini", "temperature": 0.2}


@pytest.mark.parametrize(
    ("provider", "attribute"), [("OpenAI", "ChatOpenAI"), ("Anthropic", "ChatAnthropic")]
)
def test_get_llm_raises_when_dependency_missing(llm_app, monkeypatch, provider, attribute):
    monkeypatch.setattr(llm_app, attribute, None)
    with pytest.raises(RuntimeError, match="not available"):
        llm_app.get_llm(provider, "some-model")


def test_get_llm_rejects_unknown_provider(llm_app):
    with pytest.raises(ValueError, match="Unknown provider"):
        llm_app.get_llm("Cohere", "command-r")
