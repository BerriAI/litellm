"""Storage pricing for Gemini explicit context caches created from a client ``cache_control`` marker.

Google bills a cachedContents object for storage per cached token per hour until it expires; LiteLLM picks the TTL
from the client and never deletes the cache. The spend LiteLLM records for the request that created the cache must
include that storage, and the TTL a client can buy must be bounded by the operator.
"""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ModelResponse, Usage

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.vertex_ai.context_caching.storage_pricing import (
    cache_lifetime_hours,
    cache_storage_price,
    cap_cache_ttl,
    recorded_cache_storage_cost,
)
from litellm.llms.vertex_ai.context_caching.transformation import (
    transform_openai_messages_to_gemini_context_caching,
)
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints

PRO_RATE = 4.5e-06  # USD per cached token per hour, Gemini 2.5 Pro (Vertex AI pricing, context cache storage)


@pytest.fixture(autouse=True)
def local_cost_map(monkeypatch):
    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    monkeypatch.delattr(litellm, "gemini_context_cache_max_ttl_seconds", raising=False)
    yield
    litellm.model_cost = original
    litellm.get_model_info.cache_clear()


def _messages(ttl: str | None) -> list:
    control = {"type": "ephemeral"} | ({"ttl": ttl} if ttl else {})
    return [
        {"role": "system", "content": [{"type": "text", "text": "long prefix " * 400, "cache_control": control}]},
        {"role": "user", "content": "question"},
    ]


def test_storage_rate_is_in_the_cost_map():
    assert cache_storage_price("gemini-2.5-pro", "gemini") == PRO_RATE
    assert cache_storage_price("gemini-2.5-pro", "vertex_ai") == PRO_RATE
    assert cache_storage_price("gemini-2.5-flash", "gemini") == 1e-06
    assert cache_storage_price("no-such-model", "gemini") is None


@pytest.mark.parametrize(
    "requested, cap, expected",
    [
        ("31536000s", None, "3600s"),
        ("600s", None, "600s"),
        (None, None, None),
        ("31536000s", 86400, "86400s"),
        (None, 1800, "1800s"),
    ],
)
def test_client_ttl_is_capped(monkeypatch, requested, cap, expected):
    if cap is not None:
        monkeypatch.setattr(litellm, "gemini_context_cache_max_ttl_seconds", cap, raising=False)
    assert cap_cache_ttl(requested) == expected


def test_transformed_cache_request_carries_the_capped_ttl():
    body = transform_openai_messages_to_gemini_context_caching(
        model="gemini-2.5-pro",
        messages=_messages("8760h")[:1],
        custom_llm_provider="gemini",
        cache_key="k",
        vertex_project=None,
        vertex_location=None,
    )
    assert body["ttl"] == "3600s"


def test_lifetime_comes_from_the_created_cache():
    created = {"createTime": "2026-10-14T00:00:00Z", "expireTime": "2026-10-14T05:00:00Z"}
    assert cache_lifetime_hours(created, "60s") == 5.0
    assert cache_lifetime_hours({}, "7200s") == 2.0
    assert cache_lifetime_hours({}, None) == 1.0


def _created_response(tokens: int, hours: float) -> httpx.Response:
    created = datetime(2026, 10, 14, tzinfo=timezone.utc)
    expires = datetime.fromtimestamp(created.timestamp() + hours * 3600, timezone.utc)
    body = {
        "name": "cachedContents/abc",
        "usageMetadata": {"totalTokenCount": tokens},
        "createTime": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expireTime": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return httpx.Response(200, json=body, request=httpx.Request("POST", "https://x/cachedContents"))


def _endpoints_call(model: str, client: MagicMock, logging_obj: Logging, ttl: str | None):
    with patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt",
        return_value=True,
    ):
        return ContextCachingEndpoints().check_and_create_cache(
            messages=_messages(ttl),
            optional_params={},
            api_key="k",
            api_base=None,
            model=model,
            client=client,
            timeout=10,
            logging_obj=logging_obj,
            custom_llm_provider="gemini",
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )


def _logging_obj() -> Logging:
    return Logging(
        model="gemini/gemini-2.5-pro",
        messages=[],
        stream=False,
        call_type="completion",
        start_time=datetime.now(timezone.utc),
        litellm_call_id="c",
        function_id="f",
    )


def test_created_cache_storage_is_recorded_and_priced():
    client = MagicMock(spec=HTTPHandler)
    client.get.return_value = httpx.Response(200, json={}, request=httpx.Request("GET", "https://x"))
    client.post.return_value = _created_response(tokens=2000, hours=1.0)
    logging_obj = _logging_obj()
    _, _, name = _endpoints_call("gemini-2.5-pro", client, logging_obj, "8760h")
    assert name == "cachedContents/abc"
    assert client.post.call_args.kwargs["json"]["ttl"] == "3600s"
    assert math.isclose(recorded_cache_storage_cost(logging_obj), 2000 * 1.0 * PRO_RATE, rel_tol=1e-9)

    response = ModelResponse(
        model="gemini-2.5-pro",
        usage=Usage(prompt_tokens=2100, completion_tokens=10, total_tokens=2110, cache_read_input_tokens=2000),
    )
    with_storage = litellm.completion_cost(
        completion_response=response,
        model="gemini/gemini-2.5-pro",
        custom_llm_provider="gemini",
        litellm_logging_obj=logging_obj,
    )
    without = litellm.completion_cost(
        completion_response=response,
        model="gemini/gemini-2.5-pro",
        custom_llm_provider="gemini",
        litellm_logging_obj=_logging_obj(),
    )
    assert math.isclose(with_storage - without, 2000 * PRO_RATE, rel_tol=1e-6)


def test_model_without_storage_price_gets_no_explicit_cache():
    client = MagicMock(spec=HTTPHandler)
    messages, _, name = _endpoints_call("gemini-unpriced-test-model", client, _logging_obj(), "8760h")
    assert name is None
    client.post.assert_not_called()
    assert len(messages) == 2
