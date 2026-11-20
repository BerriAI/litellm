from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException

from litellm.integrations.clickhouse.context import is_lens_analysis
from litellm.integrations.lens.proxy.internal import LENS_AGENT, LensInternalMiddleware
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params


def gateway() -> FastAPI:
    app: Final = FastAPI()

    @app.post("/chat/completions")
    async def completion(authorization: str | None = Header(default=None)) -> Mapping[str, object]:
        if authorization != "Bearer gateway-key":
            raise HTTPException(401, "Key required")
        return {
            "internal": is_lens_analysis(),
            "privacy": initialize_standard_callback_dynamic_params({}).get("turn_off_message_logging"),
        }

    app.add_middleware(LensInternalMiddleware)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_agent,internal",
    (
        (LENS_AGENT, True),
        (f"{LENS_AGENT}/1.2.3", True),
        (LENS_AGENT.upper(), True),
        ("claude-cli/2.0.1 (external, cli)", False),
        (f"{LENS_AGENT}-other/1.0", False),
        (None, False),
    ),
    ids=("bare", "versioned", "uppercase", "another_agent", "longer_name", "absent"),
)
async def test_only_a_call_from_the_lens_agent_is_lens_analysis_with_private_messages(
    user_agent: str | None, internal: bool
) -> None:
    headers: Final = {"Authorization": "Bearer gateway-key", **({"User-Agent": user_agent} if user_agent else {})}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway()), base_url="http://gateway.test"
    ) as client:
        response: Final = await client.post("/chat/completions", headers=headers)
    assert response.json() == {"internal": internal, "privacy": True if internal else None}
    assert not is_lens_analysis()


@pytest.mark.asyncio
async def test_the_lens_agent_name_does_not_replace_the_gateway_key() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway()), base_url="http://gateway.test"
    ) as client:
        response: Final = await client.post("/chat/completions", headers={"User-Agent": LENS_AGENT})
    assert response.status_code == 401
