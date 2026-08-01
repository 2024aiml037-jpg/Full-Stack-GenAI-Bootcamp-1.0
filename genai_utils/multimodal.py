"""Helpers for building multimodal LangChain messages (text / image / audio)."""

from __future__ import annotations

import base64

DEFAULT_IMAGE_MIME = "image/png"
DEFAULT_AUDIO_MIME = "audio/mpeg"


def encode_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def image_content_block(data: bytes, mime_type: str = DEFAULT_IMAGE_MIME) -> dict:
    b64 = encode_bytes_to_base64(data)
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}


def audio_content_block(data: bytes, mime_type: str = DEFAULT_AUDIO_MIME) -> dict:
    return {"type": "media", "mime_type": mime_type, "data": encode_bytes_to_base64(data)}


def build_message_content(
    modality: str,
    prompt_text: str | None = None,
    data: bytes | None = None,
    mime_type: str | None = None,
) -> str | list:
    """Build the ``content`` payload of a ``HumanMessage`` for a given modality."""
    modality = modality.lower()
    if modality == "text" or data is None:
        return prompt_text or ""
    if modality == "image":
        return [
            {"type": "text", "text": prompt_text or "Describe this image."},
            image_content_block(data, mime_type or DEFAULT_IMAGE_MIME),
        ]
    if modality == "audio":
        return [
            {"type": "text", "text": prompt_text or "Transcribe this audio."},
            audio_content_block(data, mime_type or DEFAULT_AUDIO_MIME),
        ]
    raise ValueError(f"Unknown modality {modality!r}; use 'text', 'image' or 'audio'.")


def build_human_message(
    modality: str,
    prompt_text: str | None = None,
    data: bytes | None = None,
    mime_type: str | None = None,
):
    """Same as :func:`build_message_content`, wrapped in a LangChain ``HumanMessage``."""
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=build_message_content(modality, prompt_text, data, mime_type))


def response_text(response) -> str:
    """Extract plain text from a chat model response of any shape."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ).strip()
    return content
