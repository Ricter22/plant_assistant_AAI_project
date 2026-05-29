"""Conversation memory shaping for the plant disease agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plant_assistant import settings
from plant_assistant.agent.content import message_content, message_type, render_content


def clipped_text(text: str, max_chars: int = settings.MAX_MESSAGE_MEMORY_CHARS) -> str:
    """Normalize whitespace and truncate long text before storing it in memory."""

    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def latest_tool_memory_summary(messages: list[Any]) -> str | None:
    """Build a compact memory summary from tool results for later turns."""

    classifier_summary: str | None = None
    comparison_summary: str | None = None
    retrieval_summary: str | None = None
    for message in messages:
        if message_type(message) != "tool":
            continue
        text = render_content(message_content(message)).strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            retrieval_summary = clipped_text(text, 400)
            continue

        # Keep only the latest summary for each tool type. This avoids storing
        # long raw JSON while preserving the facts needed for follow-up turns.
        if isinstance(data, dict) and {"top_class", "top_confidence", "predictions"}.issubset(data):
            confidence = float(data["top_confidence"]) * 100
            predictions = data.get("predictions") or []
            prediction_text = "; ".join(
                f"{item.get('class', 'Unknown')} {float(item.get('confidence', 0.0)) * 100:.1f}%"
                for item in predictions[:3]
            )
            classifier_summary = (
                f"Classifier result: {data['top_class']} ({confidence:.1f}% confidence). "
                f"Top predictions: {prediction_text}."
            )
            continue

        if isinstance(data, dict) and {"agreement", "models", "confidence_gap", "summary"}.issubset(data):
            model_text = "; ".join(
                (
                    f"{model_result.get('model', 'Unknown model')}: "
                    f"{model_result.get('top_class', 'Unknown')} "
                    f"{float(model_result.get('top_confidence', 0.0)) * 100:.1f}%"
                )
                for model_result in (data.get("models") or [])[:2]
            )
            comparison_summary = (
                "Model comparison: "
                f"{'agreed' if data.get('agreement') else 'disagreed'}. "
                f"{model_text}."
            )
            continue

        if isinstance(data, dict) and "results" in data:
            titles = [
                str(item.get("title") or item.get("source_url") or "Local resource")
                for item in (data.get("results") or [])[:3]
            ]
            retrieval_summary = "Retrieved local disease resources: " + "; ".join(titles)

    summaries = [summary for summary in (classifier_summary, comparison_summary, retrieval_summary) if summary]
    return " ".join(summaries) if summaries else None


def history_to_messages(
    history: list[dict[str, Any]] | None,
    max_turns: int = settings.MAX_HISTORY_TURNS,
) -> list[dict[str, Any]]:
    """Convert Streamlit chat state into compact LLM memory messages."""

    messages: list[dict[str, Any]] = []
    for turn in history or []:
        role = turn.get("role")
        if role not in {"user", "assistant"}:
            continue
        if turn.get("memory_exclude"):
            continue

        content = str(turn.get("memory_summary") or turn.get("content", "")).strip()
        if role == "user" and turn.get("image_path"):
            # The LLM cannot see image pixels in memory. Store the local path
            # and a warning so old images are used only when the user asks.
            image_name = turn.get("image_name") or Path(str(turn["image_path"])).name
            content = (
                f"{clipped_text(content)}\n\n"
                f"[Earlier attached image: {image_name}; local path: {Path(str(turn['image_path'])).resolve()}. "
                "Use this earlier image only if the user explicitly refers to it.]"
            ).strip()
        else:
            content = clipped_text(content)

        if content:
            messages.append({"role": role, "content": content})

    if max_turns > 0:
        messages = messages[-max_turns * 2 :]
    # If the selected turns are still too large, drop the oldest messages while
    # keeping at least the most recent user/assistant exchange.
    while (
        len(messages) > 2
        and sum(len(render_content(message["content"])) for message in messages) > settings.MAX_MEMORY_CHARS
    ):
        messages.pop(0)
    return messages


def build_user_content(
    *,
    question: str,
    image_path: str | Path | None,
) -> list[dict[str, str]]:
    """Build the current user message with tool-use instructions."""

    text = (
        f"Question: {question}\n\n"
        "Use tools independently whenever they are useful. "
        "Use retrieve_plant_disease_info when supporting plant disease care and management information "
        "would improve the answer."
    )

    if image_path is not None:
        # Attach only the resolved path. The actual image is loaded by the MCP
        # prediction tool, not embedded into the LLM prompt.
        path = Path(image_path).resolve()
        text = (
            f"Current image path: {path}\n\n"
            + text
            + (
                "\n\nIf diagnosing or classifying this image, call predict_plant_disease with this exact image path. "
                "If the user asks for model comparison or the MobileNetV2 prediction is below 85% confidence, "
                "call compare_plant_disease_models with this exact image path. "
                "Choose prediction parameters yourself based on the user's request."
            )
        )
    else:
        text = (
            "No current image is attached to this message. Do not call predict_plant_disease "
            "unless the user explicitly asks about an earlier attached image.\n\n"
            + text
        )
    return [{"type": "text", "text": text}]
