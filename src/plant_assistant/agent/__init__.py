"""Public agent API.

This package keeps the original ``plant_assistant.agent`` import path stable
while the implementation lives in smaller focused modules.
"""

from plant_assistant.agent.core import AgentResponse, answer_question


__all__ = ["AgentResponse", "answer_question"]
