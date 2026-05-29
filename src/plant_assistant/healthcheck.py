"""TCP healthcheck utility for the local MCP server."""

from __future__ import annotations

import socket
import sys


def main() -> int:
    """Return success when the configured host and port accept a TCP connection."""

    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    with socket.create_connection((host, port), timeout=5):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
