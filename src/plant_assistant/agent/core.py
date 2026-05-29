"""LangChain agent orchestration for the plant disease assistant."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from plant_assistant import settings
from plant_assistant.agent.content import (
    context_remaining_percent,
    describe_generated_messages,
    extract_agent_images,
    extract_prompt_tokens,
    render_content,
)
from plant_assistant.agent.fallbacks import fallback_answer_from_messages
from plant_assistant.agent.memory import (
    build_user_content,
    clipped_text,
    history_to_messages,
    latest_tool_memory_summary,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResponse:
    """Normalized response object consumed by the Streamlit interface."""

    answer: str
    images: list[dict[str, str]]
    prompt_tokens: int | None = None
    context_window_tokens: int = settings.OLLAMA_NUM_CTX
    context_remaining_percent: float | None = None
    store_in_history: bool = True
    memory_summary: str | None = None


# The system prompt defines when the LLM should use each local MCP tool and how
# to treat current versus previous image paths in the conversation memory.
SYSTEM_PROMPT = (
    "You are a concise plant owner assistant in an ongoing conversation. "
    "Use the conversation history to answer follow-up questions naturally. "
    "Use tools independently whenever they are useful. "
    "You do not receive image pixels in chat memory; current and earlier images are represented by local paths. "
    "Use predict_plant_disease when the user asks for plant disease classification, diagnosis, "
    "or likely disease labels from a local leaf image path. "
    "Use MobileNetV2 as the default prediction model. "
    "If the user asks for model comparison, model agreement, confidence, or uncertainty, "
    "call compare_plant_disease_models when an image path is available. "
    "If a MobileNetV2 prediction has top_confidence below 0.85, call compare_plant_disease_models "
    "before giving the final diagnosis. "
    "If compared models disagree, say that the local models disagree and avoid presenting the diagnosis as certain. "
    "After predicting a disease, call retrieve_plant_disease_info with the predicted plant and disease "
    "when giving symptoms, prevention, care, or management advice. "
    "Use retrieve_plant_disease_info for text-only plant disease questions when local support "
    "would improve the answer. "
    "Ground plant disease advice in retrieved results and mention source titles or URLs when available. "
    "If retrieval returns no useful support, say that the local resources did not contain enough information. "
    "If the user asks for a visual explanation, SHAP explanation, or why the model predicted a class, "
    "call predict_plant_disease with include_shap=true when an image path is available. "
    "When a current image path is provided, diagnose that current image, not an earlier image. "
    "Use earlier image paths only when the user explicitly refers to a previous image. "
    "Do not repeat a previous image diagnosis for unrelated text-only questions."
)


def mcp_connection() -> dict[str, str | list[str] | dict[str, str]]:
    """Return the MCP client configuration for stdio or HTTP transport."""

    if settings.MCP_TRANSPORT == "stdio":
        logger.info("Using MCP stdio transport")
        env_keys = [
            "MODEL_DIR",
            "CLASS_NAMES_PATH",
            "VECTOR_DB_DIR",
            "UPLOAD_DIR",
            "DEFAULT_MODEL_NAME",
            "DEFAULT_TOP_K",
            "DEFAULT_RETRIEVAL_TOP_K",
            "DEFAULT_SHAP_MAX_EVALS",
            "DEFAULT_SHAP_BATCH_SIZE",
        ]
        env = {
            "PYTHONPATH": str(settings.PROJECT_DIR / "src"),
            "MCP_TRANSPORT": "stdio",
        }
        # Pass through only the environment overrides needed by the MCP server
        # so subprocess behavior matches the parent app configuration.
        env.update({key: value for key in env_keys if (value := os.getenv(key))})
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "plant_assistant.mcp_server"],
            "cwd": str(settings.PROJECT_DIR),
            "env": env,
        }

    logger.info("Using MCP HTTP transport at %s", settings.MCP_SERVER_URL)
    return {
        "transport": "http",
        "url": settings.MCP_SERVER_URL,
        "terminate_on_close": False,
    }


async def answer_question(
    *,
    question: str,
    image_path: str | Path | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AgentResponse:
    """Answer one user turn by invoking the LLM agent and local MCP tools."""

    logger.info(
        "Starting agent request image=%s ollama=%s",
        image_path is not None,
        settings.OLLAMA_BASE_URL,
    )
    # ChatOllama owns model generation; the MCP client supplies the local image
    # classifier, model comparison, SHAP, and retrieval tools.
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=0.2,
        reasoning=False,
        base_url=settings.OLLAMA_BASE_URL,
        num_ctx=settings.OLLAMA_NUM_CTX,
        num_predict=settings.OLLAMA_NUM_PREDICT,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        client_kwargs={"timeout": settings.OLLAMA_REQUEST_TIMEOUT_SECONDS},
    )
    client = MultiServerMCPClient({"plant_disease": mcp_connection()})

    # Prepare previous turns first, then append the current message so generated
    # messages can be separated cleanly after the agent returns.
    messages = history_to_messages(history)
    current_content = build_user_content(question=question, image_path=image_path)
    logger.info(
        "Prepared %s prior history messages max_history_turns=%s memory_chars=%s current_chars=%s image=%s",
        len(messages),
        settings.MAX_HISTORY_TURNS,
        sum(len(render_content(message["content"])) for message in messages),
        len(render_content(current_content)),
        image_path is not None,
    )
    messages.append({"role": "user", "content": current_content})
    input_message_count = len(messages)

    try:
        # Load MCP tools lazily for each request, create a LangChain agent, and
        # enforce the configured timeout around the async invocation.
        tools = await client.get_tools()
        logger.info("Loaded %s MCP tools", len(tools))
        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        config = {"recursion_limit": settings.AGENT_RECURSION_LIMIT}
        invocation = agent.ainvoke({"messages": messages}, config=config)
        if settings.AGENT_TIMEOUT_SECONDS > 0:
            result = await asyncio.wait_for(invocation, timeout=settings.AGENT_TIMEOUT_SECONDS)
        else:
            result = await invocation
        result_messages = list(result["messages"])
        generated_messages = result_messages[input_message_count:]
        logger.info("Generated message diagnostics: %s", " | ".join(describe_generated_messages(generated_messages)))
        if not generated_messages:
            # Return a user-visible failure response without storing it as
            # conversation memory because it contains no useful assistant facts.
            logger.warning("Agent returned no generated messages")
            prompt_tokens = extract_prompt_tokens(result_messages)
            return AgentResponse(
                answer=(
                    "The agent did not produce a response for this message. "
                    "Please retry or start a new chat."
                ),
                images=[],
                prompt_tokens=prompt_tokens,
                context_remaining_percent=context_remaining_percent(prompt_tokens),
                store_in_history=False,
            )

        answer = render_content(generated_messages[-1].content).strip()
        memory_summary = latest_tool_memory_summary(generated_messages)
        store_in_history = True
        if not answer:
            # Some tool-heavy runs can end with an empty assistant message. In
            # that case, render the latest structured tool result directly.
            logger.warning("Agent returned empty final message; using fallback content from generated messages")
            answer = fallback_answer_from_messages(generated_messages)
            store_in_history = memory_summary is not None
        images = extract_agent_images(generated_messages)
        prompt_tokens = extract_prompt_tokens(result_messages)
        remaining_percent = context_remaining_percent(prompt_tokens)
        logger.info(
            "Agent request completed answer_chars=%s prompt_tokens=%s context_remaining_percent=%s",
            len(answer),
            prompt_tokens,
            f"{remaining_percent:.1f}" if remaining_percent is not None else None,
        )
        return AgentResponse(
            answer=answer,
            images=images,
            prompt_tokens=prompt_tokens,
            context_remaining_percent=remaining_percent,
            store_in_history=store_in_history,
            memory_summary=memory_summary or clipped_text(answer),
        )
    except Exception:
        logger.exception("Agent request failed")
        raise
