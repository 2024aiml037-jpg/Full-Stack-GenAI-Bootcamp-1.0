import base64
import pathlib
import sys
from io import BytesIO

import streamlit as st

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from genai_utils import (  # noqa: E402
    build_human_message,
    get_chat_model,
    load_env,
    response_text,
)

load_env()


def main():
    st.title("Multimodal LangChain demo — Text / Image / Audio")

    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic", "Google", "Groq"])
    model_name = st.sidebar.text_input("Model", "")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

    st.header("Input")
    input_modality = st.radio("Input modality", ["Text", "Image", "Audio"])
    prompt_text = None
    uploaded_file = None

    if input_modality == "Text":
        prompt_text = st.text_area("Prompt", value="Write a short summary of LangChain.")
    elif input_modality == "Image":
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        prompt_text = st.text_input("Prompt for image (optional)")
    elif input_modality == "Audio":
        uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
        prompt_text = st.text_input("Prompt for audio (optional)")

    st.header("Desired output")
    output_modality = st.selectbox("Output", ["Text", "Image", "Audio"])

    if st.button("Generate"):
        try:
            llm = get_chat_model(provider, model_name or None, temperature=temperature)
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return

        message = build_human_message(
            input_modality,
            prompt_text,
            data=uploaded_file.read() if uploaded_file is not None else None,
            mime_type=uploaded_file.type if uploaded_file is not None else None,
        )

        # Image-generation path (uses bind_tools style shown in the notebook)
        if output_modality == "Image":
            try:
                kwargs = {"type": "image_generation", "quality": "low"}
                if hasattr(llm, "bind_tools"):
                    llm_with_tools = llm.bind_tools([kwargs])
                    ai_message = llm_with_tools.invoke(prompt_text or "Generate an image")
                else:
                    ai_message = llm.invoke([message])

                # Try to extract image
                if hasattr(ai_message, "content_blocks"):
                    img = next((b for b in ai_message.content_blocks if b.get("type") == "image"), None)
                    if img and img.get("base64"):
                        st.image(base64.b64decode(img["base64"]))
                        st.success("Image displayed from model response")
                        return
                st.write(response_text(ai_message))
            except Exception as e:
                st.error(f"Image generation failed: {e}")
            return

        # Default: invoke model with the prepared message
        try:
            response = llm.invoke([message])
        except Exception as e:
            st.error(f"Model invocation failed: {e}")
            return

        text_output = response_text(response)

        if output_modality == "Text":
            st.subheader("Text output")
            st.write(text_output)

        elif output_modality == "Audio":
            # Try TTS fallback using gTTS if installed
            st.subheader("Audio output (TTS)")
            st.write(text_output)
            try:
                from gtts import gTTS
                tts = gTTS(text=text_output, lang="en")
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf.read(), format="audio/mp3")
            except Exception:
                st.info("Install gTTS (`pip install gTTS`) to enable text-to-speech audio playback.")

    st.markdown("---")
    st.caption("Set your API keys in a `.env` file (see `.env.example`) before running.")


if __name__ == "__main__":
    main()
