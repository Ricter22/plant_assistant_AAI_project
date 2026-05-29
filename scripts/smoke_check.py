from __future__ import annotations

import argparse
from pathlib import Path

from plant_assistant import settings


def check_assets() -> None:
    expected = [
        settings.CLASS_NAMES_PATH,
        settings.MODEL_DIR / "plant_disease_mobilenetv2.keras",
        settings.MODEL_DIR / "plant_disease_custom_cnn.keras",
        settings.VECTOR_DB_DIR / "chroma.sqlite3",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime asset(s): " + ", ".join(str(path) for path in missing))


def check_retrieval() -> None:
    from plant_assistant.mcp_server import build_retrieval_result

    result = build_retrieval_result("How do I manage tomato early blight?", top_k=1)
    if result["result_count"] < 1:
        raise RuntimeError("Retrieval smoke check returned no results.")


def check_prediction(image_path: Path) -> None:
    from plant_assistant.mcp_server import build_prediction_result

    result, _, _ = build_prediction_result("MobileNetV2", image_path, top_k=1)
    if not result.get("top_class"):
        raise RuntimeError("Prediction smoke check did not return a top class.")


def check_model_comparison(image_path: Path) -> None:
    from plant_assistant.mcp_server import build_model_comparison_result

    result = build_model_comparison_result(image_path, top_k=1)
    if len(result.get("models") or []) != 2:
        raise RuntimeError("Model comparison smoke check did not return two model results.")
    if not all(model_result.get("top_class") for model_result in result["models"]):
        raise RuntimeError("Model comparison smoke check returned an empty top class.")
    if not isinstance(result.get("agreement"), bool):
        raise RuntimeError("Model comparison smoke check did not return a boolean agreement value.")
    if not isinstance(result.get("confidence_gap"), float):
        raise RuntimeError("Model comparison smoke check did not return a numeric confidence gap.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-image", type=Path)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    check_assets()
    if not args.skip_retrieval:
        check_retrieval()
    if args.prediction_image:
        check_prediction(args.prediction_image)
        check_model_comparison(args.prediction_image)
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
