"""Run one perf mock out of process: `python server.py openai --port 8090`.

Each kind is its own pod (or local process) with its own request log; the gateway talks to it over
HTTP exactly as it would to the real provider, and the test reads `/mock/requests` (providers) or
`/mock/calls` (guardrail) afterwards to join what was received against what each client saw.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Final

import uvicorn
from fastapi import FastAPI

import mock_anthropic
import mock_bedrock
import mock_guardrail
import mock_openai

KINDS: Final[dict[str, Callable[[], FastAPI]]] = {
    "openai": mock_openai.build_app,
    "anthropic": mock_anthropic.build_app,
    "bedrock": mock_bedrock.build_app,
    "guardrail": mock_guardrail.build_app,
}


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("kind", choices=sorted(KINDS))
    _ = parser.add_argument("--host", default="0.0.0.0")
    _ = parser.add_argument("--port", type=int, default=8090)
    args: Final = parser.parse_args()
    kind: Final = str(args.kind)  # pyright: ignore[reportAny]
    port: Final = int(args.port)  # pyright: ignore[reportAny]
    host: Final = str(args.host)  # pyright: ignore[reportAny]
    uvicorn.run(KINDS[kind](), host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
