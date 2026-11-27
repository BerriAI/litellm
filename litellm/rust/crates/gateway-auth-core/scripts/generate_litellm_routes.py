"""Write src/routes/litellm_routes.json from LiteLLMRoutes.

PYTHONPATH=<repo root> uv run --no-sync python litellm/rust/crates/gateway-auth-core/scripts/generate_litellm_routes.py
"""

import json
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLMRoutes

OUTPUT: Final = Path(__file__).resolve().parent.parent / "src" / "routes" / "litellm_routes.json"
ROUTES: Final = TypeAdapter[list[str] | tuple[str, ...] | frozenset[str]](list[str] | tuple[str, ...] | frozenset[str])


def routes(member: LiteLLMRoutes) -> list[str]:
    value: Final = ROUTES.validate_python(member.value)
    return sorted(value) if isinstance(value, frozenset) else list(value)


if __name__ == "__main__":
    _ = OUTPUT.write_text(json.dumps({member.name: routes(member) for member in LiteLLMRoutes}, indent=2) + "\n")
