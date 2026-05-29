from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from plant_assistant.agent import answer_question


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    PROJECT_DIR / "datasets/plant-diseases/test/test/AppleCedarRust1.JPG",
    PROJECT_DIR / "datasets/plant-diseases/test/test/PotatoEarlyBlight1.JPG",
]


def existing_default_images() -> list[Path]:
    return [path for path in DEFAULT_IMAGES if path.exists()]


async def run_replay(images: list[Path]) -> None:
    turns: list[tuple[str, Path | None]] = [
        ("What can you help with?", None),
        ("Give a short answer: what should I check on leaves before watering?", None),
    ]
    for image_path in images:
        turns.append(("Please diagnose this plant image briefly.", image_path))
        turns.append(("What are the most important next care steps?", None))
    turns.append(("Summarize the diagnoses in this chat.", None))

    history: list[dict[str, Any]] = []
    for index, (question, image_path) in enumerate(turns, start=1):
        response = await answer_question(question=question, image_path=image_path, history=history)
        fallback = not response.store_in_history
        print(
            f"turn={index} image={bool(image_path)} answer_chars={len(response.answer)} "
            f"prompt_tokens={response.prompt_tokens} context_left={response.context_remaining_percent} "
            f"store_in_history={response.store_in_history} fallback={fallback}"
        )
        print("  answer:", " ".join(response.answer.split())[:220])
        if response.memory_summary:
            print("  memory:", response.memory_summary[:220])

        user_message = {
            "role": "user",
            "content": question,
            "image_path": str(image_path) if image_path else None,
            "image_name": image_path.name if image_path else None,
        }
        history.append(user_message)
        if response.store_in_history:
            history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "memory_summary": response.memory_summary,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay repeated text/image turns and report memory metrics.")
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        dest="images",
        help="Image path to include. Can be passed more than once.",
    )
    args = parser.parse_args()

    images = [path.expanduser().resolve() for path in args.images] if args.images else existing_default_images()
    if not images:
        raise SystemExit("No replay images found. Pass --image PATH.")

    asyncio.run(run_replay(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
