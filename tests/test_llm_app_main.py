"""Tests for the Streamlit entrypoint of the LLM access app.

Streamlit is replaced by a recording stub so the widget flow of ``main`` can be
exercised without a running Streamlit server.
"""

import base64
import sys
import types

import pytest


class FakeWidgets:
    """Records widget calls and replays scripted return values."""

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def _record(self, widget, *args, **kwargs):
        self.calls.append((widget, args, kwargs))
        return self.values.get(widget)

    def __getattr__(self, name):
        def widget(*args, **kwargs):
            return self._record(name, *args, **kwargs)

        return widget

    def called(self, widget):
        return [call for call in self.calls if call[0] == widget]

    def args_of(self, widget):
        return [call[1] for call in self.calls if call[0] == widget]


class FakeStreamlit(FakeWidgets):
    def __init__(self, values=None, sidebar_values=None):
        super().__init__(values)
        self.sidebar = FakeWidgets(sidebar_values)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response=None, error=None, tool_response=None):
        self._response = response
        self._error = error
        self._tool_response = tool_response
        self.invoked_with = []

    def invoke(self, message):
        self.invoked_with.append(message)
        if self._error is not None:
            raise self._error
        return self._response

    def bind_tools(self, tools):
        self.bound_tools = tools
        return FakeLLM(response=self._tool_response)


@pytest.fixture()
def run_main(llm_app, monkeypatch):
    def _run(*, st, llm=None, llm_error=None):
        monkeypatch.setattr(llm_app, "st", st)

        def fake_get_llm(provider, model_name, temperature=0.2):
            if llm_error is not None:
                raise llm_error
            return llm

        monkeypatch.setattr(llm_app, "get_llm", fake_get_llm)
        llm_app.main()
        return st

    return _run


def base_streamlit(output="Text", button=True, prompt="hello"):
    return FakeStreamlit(
        values={
            "radio": "Text",
            "text_area": prompt,
            "text_input": prompt,
            "selectbox": output,
            "button": button,
        },
        sidebar_values={"selectbox": "OpenAI", "text_input": "gpt-4o-mini", "slider": 0.5},
    )


def test_main_renders_without_generating(run_main):
    st = run_main(st=base_streamlit(button=False))
    assert st.called("title")
    assert st.called("caption")
    assert not st.called("write")


def test_main_writes_text_response(run_main):
    llm = FakeLLM(response=FakeResponse("generated text"))
    st = run_main(st=base_streamlit(), llm=llm)
    assert st.args_of("write") == [("generated text",)]
    assert st.args_of("subheader") == [("Text output",)]
    assert llm.invoked_with[0][0].content == "hello"


def test_main_stringifies_non_message_response(run_main):
    st = run_main(st=base_streamlit(), llm=FakeLLM(response=42))
    assert st.args_of("write") == [("42",)]


def test_main_reports_model_init_failure(run_main):
    st = run_main(st=base_streamlit(), llm_error=RuntimeError("no key"))
    assert "Failed to initialize model: no key" in st.args_of("error")[0][0]
    assert not st.called("write")


def test_main_reports_invocation_failure(run_main):
    st = run_main(st=base_streamlit(), llm=FakeLLM(error=RuntimeError("boom")))
    assert "Model invocation failed: boom" in st.args_of("error")[0][0]


def test_main_renders_generated_image(run_main):
    class ImageMessage:
        content_blocks = [{"type": "image", "base64": base64.b64encode(b"png").decode()}]
        content = "unused"

    llm = FakeLLM(tool_response=ImageMessage())
    st = run_main(st=base_streamlit(output="Image"), llm=llm)
    assert st.args_of("image") == [(b"png",)]
    assert st.called("success")


def test_main_reports_image_generation_failure(run_main):
    llm = FakeLLM(tool_response=None)
    llm.bind_tools = lambda tools: (_ for _ in ()).throw(RuntimeError("tool error"))
    st = run_main(st=base_streamlit(output="Image"), llm=llm)
    assert "Image generation failed: tool error" in st.args_of("error")[0][0]


def test_main_audio_output_falls_back_when_tts_missing(run_main, monkeypatch):
    monkeypatch.setitem(sys.modules, "gtts", None)
    st = run_main(st=base_streamlit(output="Audio"), llm=FakeLLM(response=FakeResponse("spoken")))
    assert st.args_of("write") == [("spoken",)]
    assert st.called("info")
    assert not st.called("audio")


def test_main_image_output_without_tool_support(run_main):
    class NoToolsLLM:
        def invoke(self, message):
            return FakeResponse("no image support")

    st = run_main(st=base_streamlit(output="Image"), llm=NoToolsLLM())
    assert st.args_of("write") == [("no image support",)]
    assert not st.called("image")


def test_main_image_output_without_image_block(run_main):
    class TextBlockMessage:
        content_blocks = [{"type": "text", "text": "sorry"}]
        content = "sorry"

    st = run_main(st=base_streamlit(output="Image"), llm=FakeLLM(tool_response=TextBlockMessage()))
    assert st.args_of("write") == [("sorry",)]
    assert not st.called("image")


def test_main_accepts_plain_string_response(run_main):
    st = run_main(st=base_streamlit(), llm=FakeLLM(response="plain string"))
    assert st.args_of("write") == [("plain string",)]


def test_main_audio_output_plays_tts_audio(run_main, monkeypatch):
    class FakeTTS:
        def __init__(self, text, lang):
            self.text = text
            self.lang = lang

        def write_to_fp(self, fp):
            fp.write(b"mp3-bytes")

    monkeypatch.setitem(sys.modules, "gtts", types.SimpleNamespace(gTTS=FakeTTS))
    st = run_main(st=base_streamlit(output="Audio"), llm=FakeLLM(response=FakeResponse("spoken")))
    assert st.args_of("audio") == [(b"mp3-bytes",)]
    assert st.called("audio")[0][2] == {"format": "audio/mp3"}
    assert not st.called("info")


def test_main_image_input_uploads_file(run_main):
    st = base_streamlit()
    st.values["radio"] = "Image"
    llm = FakeLLM(response=FakeResponse("caption"))
    run_main(st=st, llm=llm)
    assert st.called("file_uploader")
    assert llm.invoked_with[0][0].content == "hello"


def test_main_audio_input_uses_file_uploader(run_main):
    st = base_streamlit()
    st.values["radio"] = "Audio"
    run_main(st=st, llm=FakeLLM(response=FakeResponse("transcript")))
    assert st.args_of("file_uploader")[0] == ("Upload audio",)
