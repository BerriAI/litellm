"""Live e2e: /v1/responses with store + metadata (LIT-1201 customer path).

Customers attach metadata and store=true through the OpenAI SDK, then continue
with previous_response_id. Both turns must succeed, and any Redis keys written
for the session must carry a positive TTL (not unbounded).
"""

from __future__ import annotations

import os
import socket
import time

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import require_env, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

INSTRUCTIONS = "You are a helpful assistant."


class RedisKeyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    ttl: int


def _redis_scan(marker: str) -> tuple[RedisKeyInfo, ...]:
    import redis

    (host,) = require_env("REDIS_HOST")
    port = int((os.environ.get("REDIS_PORT") or "6379").strip() or "6379")
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise AssertionError(
            f"REDIS_HOST={host!r}:{port} unreachable ({exc}); "
            "LIT-1201 TTL check needs Redis the proxy writes to."
        ) from exc

    client = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=5)
    found: list[RedisKeyInfo] = []
    for key in client.scan_iter(match=f"*{marker}*", count=200):
        found.append(RedisKeyInfo(key=str(key), ttl=int(client.ttl(key))))
    return tuple(found)


class TestResponsesMetadata:
    @pytest.mark.covers(
        "llm.responses.openai.basic.nonstream.works",
        "other.config.responses.metadata_redis_ttl_bounded",
        exercised_on=["responses"],
    )
    def test_store_metadata_continues_and_redis_keys_have_ttl(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        # Anthropic avoids OpenAI/Gemini quota flakes; Responses translation still
        # exercises store + metadata + previous_response_id on the proxy.
        marker = unique_marker()
        model = f"e2e-resp-meta-{marker}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5-20251001",
                api_key="os.environ/ANTHROPIC_API_KEY",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client = sdk.openai(resources.key())

        first = client.responses.create(
            model=model,
            input=f"Remember marker {marker}. Reply with one word.",
            store=True,
            metadata={"session_id": marker, "customer": "e2e"},
            instructions=INSTRUCTIONS,
        )
        assert first.id, f"responses must return an id: {first!r}"
        assert first.output_text.strip(), f"responses returned empty text: {first.output!r}"

        second = client.responses.create(
            model=model,
            input="Reply with the single word ok.",
            store=True,
            previous_response_id=first.id,
            metadata={"session_id": marker, "turn": "2"},
            instructions=INSTRUCTIONS,
        )
        assert second.output_text.strip(), (
            f"previous_response_id follow-up returned empty text: {second.output!r}"
        )

        time.sleep(1.0)
        keys = _redis_scan(marker)
        unbounded = tuple(k for k in keys if k.ttl == -1)
        assert not unbounded, (
            "responses metadata must not leave Redis keys without TTL (LIT-1201); "
            f"unbounded={unbounded}"
        )
