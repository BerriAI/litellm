"""Live e2e: /v1/responses with store + metadata (LIT-1201 customer path).

Customers attach metadata and store=true, then continue with previous_response_id.
Both turns must succeed, and any Redis keys written for the session must carry a
positive TTL (not unbounded).
"""

from __future__ import annotations

import os
import socket
import time

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, ResponsesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class ResponsesMetadataBody(BaseModel):
    model: str
    input: str
    store: bool = True
    metadata: dict[str, str]
    previous_response_id: str | None = None
    instructions: str | None = "You are a helpful assistant."


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
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        # Anthropic avoids OpenAI/Gemini quota flakes; Responses translation still
        # exercises store + metadata + previous_response_id on the proxy.
        marker = unique_marker()
        model = f"e2e-resp-meta-{marker}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5-20251001",
                api_key="os.environ/ANTHROPIC_API_KEY",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        first = endpoints_client.proxy.transport.send(
            "/v1/responses",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=ResponsesMetadataBody(
                model=model,
                input=f"Remember marker {marker}. Reply with one word.",
                metadata={"session_id": marker, "customer": "e2e"},
            ),
        )
        require_successful_call(first)
        parsed = ResponsesResult.model_validate_json(first.body)
        assert parsed.id, f"responses must return an id: {first.body[:300]}"
        assert parsed.text.strip(), f"responses returned empty text: {first.body[:300]}"

        second = endpoints_client.proxy.transport.send(
            "/v1/responses",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=ResponsesMetadataBody(
                model=model,
                input="Reply with the single word ok.",
                previous_response_id=parsed.id,
                metadata={"session_id": marker, "turn": "2"},
            ),
        )
        require_successful_call(second)
        second_parsed = ResponsesResult.model_validate_json(second.body)
        assert second_parsed.text.strip(), (
            f"previous_response_id follow-up returned empty text: {second.body[:300]}"
        )

        time.sleep(1.0)
        keys = _redis_scan(marker)
        unbounded = tuple(k for k in keys if k.ttl == -1)
        assert not unbounded, (
            "responses metadata must not leave Redis keys without TTL (LIT-1201); "
            f"unbounded={unbounded}"
        )
