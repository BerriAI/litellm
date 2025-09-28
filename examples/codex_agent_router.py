"""
Example: codex-agent via Router (sync + async options).

Prereqs:
  export LITELLM_ENABLE_CODEX_AGENT=1
  export CODEX_AGENT_API_BASE=http://127.0.0.1:8788  # mini-agent OpenAI shim
  # optional: export CODEX_AGENT_API_KEY=...

Run:
  python examples/codex_agent_router.py
"""
from __future__ import annotations

import os
from litellm import Router


def main() -> None:
    r = Router(
        model_list=[
            {
                "model_name": "codex-agent-1",
                "litellm_params": {
                    "model": "codex-agent/mini",
                    "api_base": os.getenv("CODEX_AGENT_API_BASE", "http://127.0.0.1:8788"),
                    "api_key": os.getenv("CODEX_AGENT_API_KEY", ""),
                },
            }
        ]
    )

    out = r.completion(
        model="codex-agent-1",
        messages=[{"role": "user", "content": "Say hello and finish."}],
    )
    print(getattr(out.choices[0].message, "content", "").strip())


if __name__ == "__main__":
    main()

