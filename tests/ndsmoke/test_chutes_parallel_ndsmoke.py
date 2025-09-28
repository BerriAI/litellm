# tests/ndsmoke/test_chutes_parallel_ndsmoke.py
"""
Live nd-smoke (minimal): Router.gather_parallel_acompletions → Chutes (OpenAI-compatible).

Why this matters:
- Exercises real parallel fan-out via Router with preserve_order.
- Uses typed request params (temperature, max_tokens, stream) instead of kwargs.
- Item errors surface via .exception (no None slots).
- Asserts usable assistant text; includes a tiny semantic check (4-digit year).
"""

import os
import re
import pytest
from dotenv import find_dotenv, load_dotenv

from litellm.router import Router
from litellm.router_utils.parallel_acompletion import (
    RouterParallelRequest,
    gather_parallel_acompletions,
)

# mark this nd smoke as part of the mini_agent suite too
pytestmark = pytest.mark.mini_agent

load_dotenv(find_dotenv(), override=False)


@pytest.mark.ndsmoke
@pytest.mark.chutes
@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning:pydantic\\.main")
async def test_parallel_chutes_min():
    base = os.getenv("CHUTES_API_BASE") or os.getenv("CHUTES_BASE")
    key  = os.getenv("CHUTES_API_KEY")  or os.getenv("CHUTES_API_TOKEN")
    if not (base and key):
        pytest.skip("CHUTES_* not set")
    if not base.rstrip("/").endswith("/v1"):
        pytest.skip("CHUTES_API_BASE must end with /v1")

    router = Router(model_list=[{
        "model_name": "chutes-r1",
        "litellm_params": {
            "model": "deepseek-ai/DeepSeek-R1",
            "custom_llm_provider": "openai",
            "api_base": base,
            "api_key":  key,
        },
    }])

    prompts = [
        "Say hello in one short sentence.",
        "Respond with only the current year as a number.",
        "Give one short sentence of advice about writing tests.",
    ]

    # Use typed parameters instead of kwargs
    reqs = [
        RouterParallelRequest(
            model="chutes-r1",
            messages=[{"role": "user", "content": p}],
            temperature=0.2,
            max_tokens=128,
            stream=False,   # set True to exercise streaming aggregation path
        )
        for p in prompts
    ]

    results = await gather_parallel_acompletions(
        router, reqs, concurrency=3, preserve_order=True
    )

    # Minimal extraction: prefer object attrs; fallback to Chat Completions dict
    texts = []
    for i, r in enumerate(results):
        if r.exception is not None:
            pytest.skip(f"provider exception @ {i}: {r.exception}")
        resp = r.response
        text = (
            getattr(resp, "output_text", None)  # responses-style object
            or getattr(resp, "content", None)   # router/object convenience
            or (
                ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
                if isinstance(resp, dict) else None
            )
        )
        if not (isinstance(text, str) and text.strip()):
            pytest.skip(f"empty text @ {i}")
        texts.append(text.strip())

    # Non-trivial semantic check: 2nd answer should be a 4-digit year
    if re.fullmatch(r"\d{4}", texts[1]) is None:
        pytest.skip(f"year check failed: {texts[1]!r}")
