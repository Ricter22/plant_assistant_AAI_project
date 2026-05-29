"""Streamlit chat interface for the plant disease assistant.

The UI stores chat state, accepts optional image uploads, calls the async agent,
and renders both text answers and generated explanation images.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image

from plant_assistant import settings
from plant_assistant.agent import answer_question


settings.configure_logging()
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
WELCOME_MESSAGE = (
    "Ask me about plant health, disease symptoms, or care. Attach a leaf image to a message "
    "when you want image-based diagnosis."
)
CHAT_AVATARS = {
    "assistant": ":material/eco:",
    "user": ":material/local_florist:",
}
THINKING_MESSAGES = [
    "The model is thinking while the roots take hold...",
    "The model is thinking through the leaves...",
    "The model is thinking and checking the soil...",
    "The model is thinking while new ideas sprout...",
    "The model is thinking under the canopy...",
    "The model is thinking with a little sunlight...",
]


def save_upload(uploaded_file) -> Path:
    """Persist an uploaded image and return the local path used by MCP tools."""

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    path = settings.UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    logger.info("Saved upload name=%s path=%s bytes=%s", uploaded_file.name, path, path.stat().st_size)
    return path


def run_async(coro):
    """Run an async agent call from Streamlit's synchronous script body."""

    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def initialize_session_state() -> None:
    """Create all Streamlit session-state keys used by the app."""

    st.session_state.setdefault("messages", [{"role": "assistant", "content": WELCOME_MESSAGE}])
    st.session_state.setdefault("upload_key", 0)
    reset_context_metrics(only_missing=True)


def reset_context_metrics(only_missing: bool = False) -> None:
    """Reset context-window metrics or initialize missing metric keys."""

    defaults = {
        "context_remaining_percent": None,
        "context_prompt_tokens": None,
        "context_window_tokens": settings.OLLAMA_NUM_CTX,
    }
    for key, value in defaults.items():
        if not only_missing or key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """Clear chat history and reset upload/context state for a new chat."""

    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.upload_key += 1
    reset_context_metrics()


def render_context_status(container=st) -> None:
    """Render model name and remaining context-window status."""

    remaining_percent = st.session_state.context_remaining_percent
    prompt_tokens = st.session_state.context_prompt_tokens
    context_window = st.session_state.context_window_tokens
    model_info = f"Model: `{settings.OLLAMA_MODEL}`"
    if remaining_percent is None or prompt_tokens is None:
        container.caption(f"{model_info} | Context left: not measured yet")
        return

    container.caption(
        f"{model_info} | Context left: {remaining_percent:.2f}% "
        f"({max(context_window - prompt_tokens, 0):,}/{context_window:,} tokens)"
    )


def render_chat_message(message: dict) -> None:
    """Render one stored chat message, including uploaded or generated images."""

    role = message["role"]
    with st.chat_message(role, avatar=CHAT_AVATARS.get(role)):
        if message.get("image_path"):
            try:
                # Uploaded user images are referenced by local path and shown as
                # a small preview above the text message.
                st.image(
                    Image.open(message["image_path"]),
                    caption=message.get("image_name") or Path(message["image_path"]).name,
                    width=280,
                )
            except Exception:
                logger.warning("Could not render uploaded image path=%s", message.get("image_path"))
        st.markdown(message.get("content", ""))
        render_assistant_images(message.get("images", []) or [])


def render_assistant_images(images: list[dict[str, str]]) -> None:
    """Decode and render base64 images returned by MCP tools."""

    for index, image in enumerate(images, start=1):
        try:
            st.image(
                base64.b64decode(image["data"]),
                caption=image.get("caption") or f"Visual explanation {index}",
                width=520,
            )
        except Exception:
            logger.warning("Could not render assistant image index=%s", index)


def stream_characters(text: str, delay_seconds: float = 0.01):
    """Yield answer text one character at a time for Streamlit streaming."""

    for char in text:
        yield char
        time.sleep(delay_seconds)


def update_context_metrics(response) -> None:
    """Copy context metrics from the agent response into session state."""

    st.session_state.context_remaining_percent = getattr(response, "context_remaining_percent", None)
    st.session_state.context_prompt_tokens = getattr(response, "prompt_tokens", None)
    st.session_state.context_window_tokens = getattr(response, "context_window_tokens", settings.OLLAMA_NUM_CTX)


def captioned_images(images: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add stable captions to generated explanation images."""

    return [{**image, "caption": f"Visual explanation {index}"} for index, image in enumerate(images, start=1)]


# Page setup and persistent session-state initialization happen before any
# widgets are rendered because Streamlit reruns this script on every interaction.
st.set_page_config(page_title="Plant Disease Assistant", layout="wide", initial_sidebar_state="collapsed")
initialize_session_state()
logger.info(
    "Loaded Streamlit app ollama=%s model=%s mcp=%s",
    settings.OLLAMA_BASE_URL,
    settings.OLLAMA_MODEL,
    settings.MCP_SERVER_URL,
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 3.5rem;
        padding-bottom: 9rem;
    }

    /* Reserve a gap below the chat input so the status chip has room under it */
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 4rem !important;
    }

    /* ── New Chat button ─ fixed top-right ── */
    .st-key-new_chat_corner,
    [class*="st-key-new_chat_corner"] {
        position: fixed !important;
        top: 0.6rem !important;
        right: 1rem !important;
        z-index: 999999 !important;
        width: auto !important;
    }

    .st-key-new_chat_corner button,
    [class*="st-key-new_chat_corner"] button {
        height: 1.9rem !important;
        padding: 0 0.7rem !important;
        border: 1px solid #b5c9b3 !important;
        border-radius: 8px !important;
        background: rgba(250, 255, 249, 0.97) !important;
        color: #2d5534 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        box-shadow: 0 1px 4px rgba(45, 85, 52, 0.12) !important;
        transition: background 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    }

    .st-key-new_chat_corner button:hover,
    [class*="st-key-new_chat_corner"] button:hover {
        background: rgba(220, 242, 217, 0.98) !important;
        border-color: #7ca27a !important;
        box-shadow: 0 2px 7px rgba(45, 85, 52, 0.2) !important;
        color: #1f4629 !important;
    }

    /* ── Input status chip ─ left-aligned, in the gap below the chat input ── */
    .st-key-input_status_bar,
    [class*="st-key-input_status_bar"] {
        position: fixed !important;
        left: 5rem !important;
        bottom: 1.9rem !important;
        z-index: 999999 !important;
        width: auto !important;
        pointer-events: none !important;
    }

    .st-key-input_status_bar [data-testid="stCaptionContainer"],
    [class*="st-key-input_status_bar"] [data-testid="stCaptionContainer"] {
        display: inline-flex !important;
        align-items: center !important;
        padding: 0.22rem 0.7rem !important;
        border-radius: 100px !important;
        border: 1px solid rgba(100, 160, 100, 0.35) !important;
        background: rgba(244, 252, 243, 0.97) !important;
        box-shadow: 0 2px 8px rgba(20, 60, 30, 0.1), 0 1px 3px rgba(20, 60, 30, 0.07) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        white-space: nowrap !important;
        max-width: 50vw !important;
        overflow: hidden !important;
    }

    .st-key-input_status_bar [data-testid="stCaptionContainer"] p,
    [class*="st-key-input_status_bar"] [data-testid="stCaptionContainer"] p {
        margin: 0 !important;
        padding: 0 !important;
        color: #2d5534 !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.015em !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .st-key-input_status_bar [data-testid="stCaptionContainer"] p code,
    [class*="st-key-input_status_bar"] [data-testid="stCaptionContainer"] p code {
        font-size: inherit !important;
        background: transparent !important;
        padding: 0 !important;
        color: #1f4629 !important;
        font-weight: 600 !important;
    }

    [data-testid="stChatMessageAvatar"] {
        background: #e3f3e1;
        border: 1px solid #8ab58a;
        color: #1f6b3a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="new_chat_corner"):
    if st.button("New chat", key="new_chat_button", help="New chat", icon=":material/add:", type="tertiary"):
        reset_chat()
        st.rerun()

st.title("Plant Disease Assistant")

for message in st.session_state.messages:
    render_chat_message(message)

with st.container(key="input_status_bar"):
    context_status_slot = st.empty()
    render_context_status(context_status_slot)

chat_value = st.chat_input(
    "Message the plant assistant",
    accept_file=True,
    file_type=SUPPORTED_IMAGE_TYPES,
    key=f"chat_input_{st.session_state.upload_key}",
)

if chat_value is not None:
    # Streamlit returns an object when file upload is enabled, but can still
    # return a plain string in older/simple modes. Support both shapes.
    prompt = (getattr(chat_value, "text", "") or "").strip()
    if not prompt and isinstance(chat_value, str):
        prompt = chat_value.strip()
    uploaded_files = list(getattr(chat_value, "files", []) or [])

    image_path: Path | None = None
    image_name: str | None = None
    if uploaded_files:
        # The assistant supports one current image per message; extra files are
        # ignored by this UI even though the widget may return a list.
        uploaded_file = uploaded_files[0]
        image_path = save_upload(uploaded_file)
        image_name = uploaded_file.name

    if not prompt and image_path:
        prompt = "Please analyze this plant image."

    if not prompt:
        # Empty submissions without an image do not create chat turns.
        st.stop()

    user_message = {
        "role": "user",
        "content": prompt.strip(),
        "image_path": str(image_path) if image_path else None,
        "image_name": image_name,
    }
    history = list(st.session_state.messages)
    st.session_state.messages.append(user_message)

    render_chat_message(user_message)

    with st.chat_message("assistant", avatar=CHAT_AVATARS["assistant"]):
        thinking_message = random.choice(THINKING_MESSAGES)
        try:
            logger.info(
                "Chat submitted image=%s",
                image_path,
            )
            with st.spinner(thinking_message):
                # The agent receives the previous history separately from the
                # current message so it can build compact LLM memory.
                response = run_async(
                    answer_question(
                        question=prompt.strip(),
                        image_path=image_path,
                        history=history,
                    )
                )
        except Exception as exc:
            logger.exception("Agent response failed")
            st.error(f"Response failed: {exc}")
        else:
            answer = response.answer
            images = captioned_images(response.images)
            update_context_metrics(response)
            logger.info(
                "Context metric updated prompt_tokens=%s context_window_tokens=%s remaining_percent=%s",
                st.session_state.context_prompt_tokens,
                st.session_state.context_window_tokens,
                f"{st.session_state.context_remaining_percent:.2f}"
                if st.session_state.context_remaining_percent is not None
                else None,
            )
            render_context_status(context_status_slot)
            st.write_stream(stream_characters(answer))
            render_assistant_images(images)
            if getattr(response, "store_in_history", True):
                # Store a compact memory summary when the agent provides one so
                # future turns stay inside the configured context budget.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "memory_summary": getattr(response, "memory_summary", None),
                        "images": images,
                    }
                )
            else:
                logger.warning("Assistant response displayed but excluded from future memory")
            logger.info(
                "Chat response completed answer_chars=%s image_count=%s store_in_history=%s",
                len(answer),
                len(response.images),
                getattr(response, "store_in_history", True),
            )
