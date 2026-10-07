from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from integration._support import responses_vendor as rv
from pydantic import JsonValue

from litellm.integrations.custom_logger import CustomLogger

OUT_ENVIRONMENT: Final = "AGENTIC_PROBE_OUT"


class AgenticProbe(CustomLogger):
    """Records every agentic-loop hook call the proxy makes, one JSON line per call, and never runs a loop."""

    async def async_should_run_agentic_loop(
        self,
        response: object,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: Mapping[str, object],
    ) -> tuple[bool, dict[str, object]]:
        out: Final = os.environ.get(OUT_ENVIRONMENT)
        if out:
            line: Final[Mapping[str, JsonValue]] = {
                "marker": rv.newest_marker(f"{messages!s} {response!s}"),
                "surface": str(kwargs.get("_agentic_loop_api_surface")),
                "response_type": type(response).__name__,
                "stream": stream,
                "model": model,
                "provider": custom_llm_provider,
            }
            with open(out, "a", encoding="utf-8") as sink:
                sink.write(json.dumps(line) + "\n")
        return False, {}


probe: Final = AgenticProbe()


def lines(path: Path, marker: str) -> tuple[Mapping[str, JsonValue], ...]:
    recorded: Final = tuple(rv.JSON_OBJECT.validate_json(line) for line in path.read_text().splitlines() if line)
    return tuple(line for line in recorded if line["marker"] == marker)
