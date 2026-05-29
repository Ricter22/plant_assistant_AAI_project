"""Message content helpers for the plant disease agent."""

from __future__ import annotations

from typing import Any

from plant_assistant import settings


def text_block_content(block: Any) -> str:
    """Extract plain text from one LangChain-style content block."""

    if isinstance(block, dict) and block.get("type") == "text":
        return str(block.get("text", ""))
    return block if isinstance(block, str) else ""


def render_content(content: Any) -> str:
    """Render message content, including list-based blocks, as plain Markdown."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(part for block in content if (part := text_block_content(block)))
    return str(content)


def extract_image_blocks(content: Any) -> list[dict[str, str]]:
    """Collect base64 images returned by model/tool messages."""

    if isinstance(content, dict):
        return extract_image_blocks([content])

    images: list[dict[str, str]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            data = block.get("data") or block.get("base64")
            if block_type == "image" and data:
                images.append(
                    {
                        "data": str(data),
                        "mime_type": str(block.get("mimeType") or block.get("mime_type") or "image/png"),
                    }
                )
            elif block_type == "image_url":
                # Some providers encode returned images as data URLs rather than
                # MCP image blocks, so decode that shape into the UI's format.
                image_url = block.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    url = image_url["url"]
                    if url.startswith("data:") and ";base64," in url:
                        mime_type, data = url[5:].split(";base64,", 1)
                        images.append({"data": data, "mime_type": mime_type})
    return images


def message_type(message: Any) -> str:
    """Return a provider-independent message role/type string."""

    return str(getattr(message, "type", "") or getattr(message, "role", "") or "")


def message_content(message: Any) -> Any:
    """Return content from either a dict message or a LangChain message object."""

    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def extract_agent_images(messages: list[Any]) -> list[dict[str, str]]:
    """Extract generated images while ignoring user-provided message content."""

    images: list[dict[str, str]] = []
    for message in messages:
        if message_type(message) in {"human", "user"}:
            continue
        images.extend(extract_image_blocks(message_content(message)))
    return images


def extract_prompt_tokens(messages: list[Any]) -> int | None:
    """Find the largest reported prompt-token count in generated messages."""

    prompt_token_counts: list[int] = []
    for message in messages:
        for metadata_name, token_key in (
            ("usage_metadata", "input_tokens"),
            ("response_metadata", "prompt_eval_count"),
        ):
            metadata = getattr(message, metadata_name, None)
            token_count = metadata.get(token_key) if isinstance(metadata, dict) else None
            if isinstance(token_count, int):
                prompt_token_counts.append(token_count)
                break

    return max(prompt_token_counts) if prompt_token_counts else None


def context_remaining_percent(prompt_tokens: int | None) -> float | None:
    """Convert a prompt-token count into remaining context-window percentage."""

    if prompt_tokens is None or settings.OLLAMA_NUM_CTX <= 0:
        return None
    remaining_tokens = max(settings.OLLAMA_NUM_CTX - prompt_tokens, 0)
    return remaining_tokens / settings.OLLAMA_NUM_CTX * 100


def describe_generated_messages(messages: list[Any]) -> list[str]:
    """Create compact diagnostics for generated LangChain messages."""

    descriptions: list[str] = []
    for index, message in enumerate(messages):
        content = render_content(message_content(message))
        tool_calls = getattr(message, "tool_calls", None) or []
        response_metadata = getattr(message, "response_metadata", None) or {}
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        reasoning = getattr(message, "additional_kwargs", {}).get("reasoning_content", "")
        descriptions.append(
            (
                f"{index}:{message_type(message) or type(message).__name__}"
                f":chars={len(content)}"
                f":tool_calls={len(tool_calls)}"
                f":done={response_metadata.get('done_reason')}"
                f":in={usage_metadata.get('input_tokens')}"
                f":out={usage_metadata.get('output_tokens')}"
                f":reasoning_chars={len(str(reasoning))}"
            )
        )
    return descriptions
