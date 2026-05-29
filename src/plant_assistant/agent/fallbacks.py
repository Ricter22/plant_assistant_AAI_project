"""Fallback renderers used when the agent does not return final text."""

from __future__ import annotations

import json
from typing import Any

from plant_assistant.agent.content import message_content, message_type, render_content


def fallback_from_tool_json(text: str) -> str:
    """Convert structured tool JSON into readable Markdown if the LLM is silent."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    # Prediction tool fallback: summarize the top class and ranked labels.
    if isinstance(data, dict) and {"top_class", "top_confidence", "predictions"}.issubset(data):
        confidence = float(data["top_confidence"]) * 100
        lines = [
            f"The classifier predicts **{data['top_class']}** with {confidence:.1f}% confidence.",
        ]
        predictions = data.get("predictions") or []
        if predictions:
            lines.append("Top predictions:")
            lines.extend(
                f"- {item.get('class', 'Unknown')}: {float(item.get('confidence', 0.0)) * 100:.1f}%"
                for item in predictions
            )
        if isinstance(data.get("shap"), dict) and data["shap"].get("included"):
            lines.append("The SHAP visual explanation is attached below.")
        return "\n".join(lines)

    # Model-comparison fallback: preserve agreement and per-model confidence.
    if isinstance(data, dict) and {"agreement", "models", "confidence_gap", "summary"}.issubset(data):
        models = data.get("models") or []
        lines = [
            "The local models agree on the top prediction."
            if data.get("agreement")
            else "The local models disagree on the top prediction."
        ]
        for model_result in models:
            confidence = float(model_result.get("top_confidence", 0.0)) * 100
            lines.append(
                f"- **{model_result.get('model', 'Unknown model')}**: "
                f"{model_result.get('top_class', 'Unknown')} ({confidence:.1f}% confidence)"
            )
        lines.append(f"Confidence gap: {float(data.get('confidence_gap', 0.0)) * 100:.1f} percentage points.")
        return "\n".join(lines)

    # Retrieval fallback: show compact snippets from the local resource search.
    if isinstance(data, dict) and "results" in data:
        results = data.get("results") or []
        if not results:
            return "The local plant disease resources did not return any matching results."
        lines = ["I found these relevant local resource notes:"]
        for item in results[:3]:
            title = str(item.get("title") or item.get("source_url") or "Local resource")
            content = " ".join(str(item.get("content", "")).split())
            snippet = content[:260].rstrip()
            if len(content) > len(snippet):
                snippet += "..."
            lines.append(f"- **{title}**: {snippet}")
        return "\n".join(lines)

    return text.strip()


def fallback_answer_from_messages(messages: list[Any]) -> str:
    """Use the latest useful tool or assistant message when the final answer is empty."""

    for accepted_types, renderer in (
        ({"tool"}, fallback_from_tool_json),
        ({"ai", "assistant"}, lambda text: text),
    ):
        for message in reversed(messages):
            text = render_content(message_content(message)).strip()
            if text and message_type(message) in accepted_types:
                return renderer(text)

    return "The agent finished without returning text. Please retry the question or start a new chat."
