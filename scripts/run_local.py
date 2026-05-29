from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO


PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
DEFAULT_MCP_PORT = 8000
DEFAULT_STREAMLIT_PORT = 8501
REQUIRED_MODULES = [
    "streamlit",
    "fastmcp",
    "langchain",
    "langchain_mcp_adapters",
    "langchain_ollama",
    "chromadb",
    "tensorflow",
]


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def venv_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_DIR / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else None


def reexec_into_venv() -> None:
    target = venv_python()
    if target is None:
        return
    if Path(sys.executable) == target:
        return
    print(f"[preflight] Re-running with project virtualenv: {target}")
    os.execv(str(target), [str(target), *sys.argv])


def merged_env(run_log: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(PROJECT_DIR / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]

    mcp_port = env.get("MCP_PORT", str(DEFAULT_MCP_PORT))
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_MODEL": env.get("OLLAMA_MODEL", "qwen3.6:latest"),
            "OLLAMA_NUM_CTX": env.get("OLLAMA_NUM_CTX", "32768"),
            "OLLAMA_NUM_PREDICT": env.get("OLLAMA_NUM_PREDICT", "1024"),
            "OLLAMA_KEEP_ALIVE": env.get("OLLAMA_KEEP_ALIVE", "2m"),
            "OLLAMA_REQUEST_TIMEOUT_SECONDS": env.get("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"),
            "AGENT_TIMEOUT_SECONDS": env.get("AGENT_TIMEOUT_SECONDS", "180"),
            "AGENT_RECURSION_LIMIT": env.get("AGENT_RECURSION_LIMIT", "12"),
            "MAX_HISTORY_TURNS": env.get("MAX_HISTORY_TURNS", "6"),
            "MAX_MEMORY_CHARS": env.get("MAX_MEMORY_CHARS", "6000"),
            "MAX_MESSAGE_MEMORY_CHARS": env.get("MAX_MESSAGE_MEMORY_CHARS", "700"),
            "MCP_TRANSPORT": "http",
            "MCP_HOST": env.get("MCP_HOST", "0.0.0.0"),
            "MCP_PORT": mcp_port,
            "MCP_PATH": env.get("MCP_PATH", "/mcp"),
            "MCP_SERVER_URL": env.get("MCP_SERVER_URL", f"http://localhost:{mcp_port}/mcp"),
            "STREAMLIT_PORT": env.get("STREAMLIT_PORT", str(DEFAULT_STREAMLIT_PORT)),
            "TF_CPP_MIN_LOG_LEVEL": env.get("TF_CPP_MIN_LOG_LEVEL", "2"),
            "TRANSFORMERS_VERBOSITY": env.get("TRANSFORMERS_VERBOSITY", "error"),
            "HF_HUB_VERBOSITY": env.get("HF_HUB_VERBOSITY", "error"),
            "HF_HUB_DISABLE_PROGRESS_BARS": env.get("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
            "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM", "false"),
            "MCP_PRELOAD_VECTOR_DB": env.get("MCP_PRELOAD_VECTOR_DB", "1"),
            "DEFAULT_SHAP_MAX_EVALS": env.get("DEFAULT_SHAP_MAX_EVALS", "100"),
            "DEFAULT_SHAP_BATCH_SIZE": env.get("DEFAULT_SHAP_BATCH_SIZE", "8"),
            "PLANT_ASSISTANT_LOG_FILE": str(run_log),
        }
    )
    return env


def fail(message: str) -> int:
    print(f"[preflight] ERROR: {message}")
    return 1


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_port(host: str, port: int, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def require_modules() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def check_assets(env: dict[str, str]) -> list[Path]:
    model_dir = Path(env.get("MODEL_DIR", PROJECT_DIR / "notebooks")).expanduser().resolve()
    class_names = Path(
        env.get("CLASS_NAMES_PATH", model_dir / "plant_disease_class_names.json")
    ).expanduser().resolve()
    vector_db = Path(env.get("VECTOR_DB_DIR", PROJECT_DIR / "rag" / "chroma_db")).expanduser().resolve()
    expected = [
        model_dir / "plant_disease_mobilenetv2.keras",
        model_dir / "plant_disease_custom_cnn.keras",
        class_names,
        vector_db / "chroma.sqlite3",
    ]
    return [path for path in expected if not path.exists()]


def ollama_tags(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama(env: dict[str, str]) -> str | None:
    base_url = env["OLLAMA_BASE_URL"]
    model = env["OLLAMA_MODEL"]
    try:
        data = ollama_tags(base_url)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return f"Cannot reach Ollama at {base_url}: {exc}"

    models = {item.get("name") for item in data.get("models", [])}
    models.update(item.get("model") for item in data.get("models", []))
    if model not in models:
        return f"Ollama model {model!r} is not installed. Run: ollama pull {model}"
    return None


def run_retrieval_smoke(env: dict[str, str]) -> int:
    print("[preflight] Checking Chroma vector DB retrieval...")
    result = subprocess.run(
        [sys.executable, "scripts/smoke_check.py"],
        cwd=PROJECT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    for line in result.stdout.splitlines():
        print(f"[preflight] {line}")
    return result.returncode


def tee_output(
    name: str,
    process: subprocess.Popen,
    log_file: TextIO,
    events: queue.Queue[tuple[str, int]],
) -> None:
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        text = line.rstrip()
        print(f"[{name}] {text}")
        log_file.write(line)
        log_file.flush()
    events.put((name, process.wait()))


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    events: queue.Queue[tuple[str, int]],
) -> tuple[subprocess.Popen, TextIO, threading.Thread]:
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"# {name} started at {dt.datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write(f"# command: {' '.join(command)}\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    thread = threading.Thread(
        target=tee_output,
        args=(name, process, log_file, events),
        daemon=True,
    )
    thread.start()
    return process, log_file, thread


def terminate(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def preflight(env: dict[str, str]) -> int:
    missing = require_modules()
    if missing:
        return fail(
            "Missing Python modules for this interpreter: "
            + ", ".join(missing)
            + ". Activate/install the project environment with: python -m pip install -r requirements-local.txt"
        )

    missing_assets = check_assets(env)
    if missing_assets:
        return fail("Missing runtime asset(s): " + ", ".join(str(path) for path in missing_assets))

    mcp_port = int(env["MCP_PORT"])
    streamlit_port = int(env["STREAMLIT_PORT"])
    for port, service in [(mcp_port, "MCP server"), (streamlit_port, "Streamlit")]:
        if not is_port_free("127.0.0.1", port):
            return fail(f"Port {port} is already in use by another process; needed for {service}.")

    ollama_error = check_ollama(env)
    if ollama_error:
        return fail(ollama_error)

    retrieval_status = run_retrieval_smoke(env)
    if retrieval_status != 0:
        return fail("Vector DB retrieval smoke check failed. See the preflight output above.")

    return 0


def main() -> int:
    reexec_into_venv()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log = LOG_DIR / f"run-local-{stamp}.log"
    with run_log.open("a", encoding="utf-8") as run_log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, run_log_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, run_log_file)  # type: ignore[assignment]
        try:
            env = merged_env(run_log)

            print(f"[preflight] Python: {sys.executable}")
            print(f"[preflight] Logs: {LOG_DIR}")
            print(f"[preflight] Aggregate log: {run_log}")
            preflight_status = preflight(env)
            if preflight_status != 0:
                return preflight_status

            events: queue.Queue[tuple[str, int]] = queue.Queue()
            processes: list[subprocess.Popen] = []
            log_files: list[TextIO] = []
            threads: list[threading.Thread] = []

            try:
                mcp_port = int(env["MCP_PORT"])
                mcp, mcp_log, mcp_thread = start_process(
                    "mcp",
                    [sys.executable, "-m", "plant_assistant.mcp_server"],
                    env,
                    LOG_DIR / f"mcp-server-{stamp}.log",
                    events,
                )
                processes.append(mcp)
                log_files.append(mcp_log)
                threads.append(mcp_thread)
                if not wait_for_port("127.0.0.1", mcp_port, mcp, timeout=45):
                    return fail(f"MCP server did not become ready on port {mcp_port}.")
                print(f"[mcp] Ready at {env['MCP_SERVER_URL']}")

                streamlit_port = int(env["STREAMLIT_PORT"])
                streamlit, streamlit_log, streamlit_thread = start_process(
                    "streamlit",
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "src/plant_assistant/ui/streamlit_app.py",
                        "--server.address=0.0.0.0",
                        f"--server.port={streamlit_port}",
                        "--server.headless=true",
                        "--server.fileWatcherType=none",
                        "--browser.gatherUsageStats=false",
                    ],
                    env,
                    LOG_DIR / f"streamlit-{stamp}.log",
                    events,
                )
                processes.append(streamlit)
                log_files.append(streamlit_log)
                threads.append(streamlit_thread)
                if not wait_for_port("127.0.0.1", streamlit_port, streamlit, timeout=45):
                    return fail(f"Streamlit did not become ready on port {streamlit_port}.")

                print(f"\nOpen http://localhost:{streamlit_port} in your browser.")
                print(f"Logs are in {LOG_DIR}\n")

                name, returncode = events.get()
                if returncode == 0:
                    print(f"[{name}] exited.")
                else:
                    print(f"[{name}] exited with code {returncode}. Check logs in {LOG_DIR}.")
                return returncode
            except KeyboardInterrupt:
                print("\n[run] Stopping local services...")
                return 130
            finally:
                terminate(processes)
                for thread in threads:
                    thread.join(timeout=2)
                for log_file in log_files:
                    log_file.close()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
