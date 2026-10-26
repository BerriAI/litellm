from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.model_inventory import get_inventory_model_metadata, get_provider_model_inventory
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.types.llms.openrouter import OPENROUTER_REASONING_EFFORTS, OpenRouterProviderPolicy
from litellm.types.proxy.model_inventory import SupplierModelInventory


def _endpoint(tag: str = "supplier/fp8", **fields: object) -> Mapping[str, object]:
    return {
        "tag": tag,
        "context_length": 1200,
        "max_prompt_tokens": None,
        "max_completion_tokens": 300,
        "supported_parameters": ["tools", "reasoning", "reasoning_effort"],
        **fields,
    }


async def _inventory(
    handler: AsyncHTTPHandler,
    cache: InMemoryCache,
    *,
    only: tuple[str, ...] = ("supplier/fp8",),
    key: str = "fixture-key",
    base: str = "https://supplier.test/custom/v1",
    refresh: bool = False,
    previous: SupplierModelInventory | None = None,
    model: str = "org/model",
) -> tuple[SupplierModelInventory, Mapping[str, object]]:
    inventory: Final = await get_provider_model_inventory(
        provider="openrouter",
        api_base=base,
        headers={"authorization": key},
        client=handler,
        cache=cache,
        endpoint_models=(model, model),
        force_refresh=refresh,
        previous=previous,
    )
    assert isinstance(inventory, SupplierModelInventory)
    return inventory, get_inventory_model_metadata(
        inventory, model, OpenRouterProviderPolicy(provider="openrouter", only=only)
    )


def _catalog(reasoning: Mapping[str, object] | None = None) -> Mapping[str, object]:
    return {
        "data": [
            {
                "id": "org/model",
                "context_length": 9000,
                "top_provider": {"context_length": 8000, "max_completion_tokens": 7000},
                "supported_parameters": ["tools", "reasoning_effort"],
                "reasoning": {"supported_efforts": ["max", "high", "low"], "default_effort": "high"}
                if reasoning is None
                else reasoning,
            },
            {"id": "org/second"},
        ]
    }


@pytest.mark.parametrize(
    ("endpoints", "only", "context", "output", "input_limit", "efforts", "default", "reasoning"),
    (
        (
            [_endpoint(), _endpoint("other", context_length=9000)],
            ("supplier/fp8",),
            1200,
            300,
            None,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [_endpoint(), _endpoint("supplier/fp8/eu", context_length=900, max_completion_tokens=200)],
            ("supplier",),
            900,
            200,
            None,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [_endpoint(max_prompt_tokens=500), _endpoint("second", max_prompt_tokens=400)],
            ("supplier", "second"),
            1200,
            300,
            400,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [_endpoint(), _endpoint("second", supported_parameters=["reasoning"])],
            ("supplier", "second"),
            1200,
            300,
            None,
            None,
            None,
            True,
        ),
        ([_endpoint(supported_parameters=[])], ("supplier",), 1200, 300, None, None, None, False),
        (
            [_endpoint(supported_parameters=None, context_length=None, max_completion_tokens=None)],
            ("supplier",),
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        ([_endpoint()], ("supplier/fp4",), None, None, None, None, None, None),
        ([], ("supplier",), None, None, None, None, None, None),
        (
            [_endpoint(), _endpoint("supplier/fast", context_length=10)],
            ("supplier",),
            1200,
            300,
            None,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [_endpoint(), _endpoint("supplier/fast", context_length=10)],
            ("supplier/fast",),
            10,
            300,
            None,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [_endpoint(status=-1, uptime_last_30m=0)],
            ("supplier",),
            1200,
            300,
            None,
            ["max", "high", "low"],
            "high",
            True,
        ),
        (
            [
                _endpoint(reasoning_effort_levels=["high", "low"]),
                _endpoint("second", reasoning_effort_levels=["max", "high"], default_reasoning_effort="max"),
            ],
            ("supplier", "second"),
            1200,
            300,
            None,
            ["high"],
            None,
            True,
        ),
    ),
)
async def test_pinned_metadata_guarantees_only_eligible_endpoint_contracts(
    endpoints: list[Mapping[str, object]],
    only: tuple[str, ...],
    context: int | None,
    output: int | None,
    input_limit: int | None,
    efforts: list[str] | None,
    default: str | None,
    reasoning: bool | None,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "fixture-key"
        if request.url.path.endswith("/endpoints"):
            assert request.url.path == "/custom/v1/models/org/model/endpoints"
            return httpx.Response(200, json={"data": {"id": "org/model", "endpoints": endpoints}})
        assert request.url.path == "/custom/v1/models"
        return httpx.Response(200, json=_catalog())

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        inventory, metadata = await _inventory(handler, InMemoryCache(), only=only)
        assert metadata.get("context_window") == context
        assert metadata.get("max_output_tokens") == output
        assert metadata.get("max_input_tokens") == input_limit
        assert metadata.get("reasoning_effort_levels") == efforts
        assert metadata.get("default_reasoning_effort") == default
        assert metadata.get("supports_reasoning") == reasoning
        assert all(value is not None for value in metadata.values())
        assert get_inventory_model_metadata(inventory, "org/model")["context_window"] == 8000
        assert get_inventory_model_metadata(inventory, "org/model")["max_output_tokens"] == 7000


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (
        ({}, {}),
        (
            {"supported_efforts": None, "mandatory": True, "default_effort": "none"},
            {"reasoning_effort_levels": [level for level in OPENROUTER_REASONING_EFFORTS if level != "none"]},
        ),
        (
            {"supported_efforts": ["high", "none"], "default_effort": "none", "default_enabled": False},
            {"reasoning_effort_levels": ["high", "none"], "default_reasoning_effort": "none"},
        ),
        ({"supported_efforts": ["future-effort", "high"]}, {"reasoning_effort_levels": ["future-effort", "high"]}),
        (
            {"supported_efforts": None, "default_effort": "high"},
            {"reasoning_effort_levels": list(OPENROUTER_REASONING_EFFORTS), "default_reasoning_effort": "high"},
        ),
        ({"supported_efforts": [], "default_effort": "high"}, {"reasoning_effort_levels": []}),
        (
            {"supported_efforts": ["xhigh", "high"], "default_effort": "xhigh", "default_enabled": False},
            {"reasoning_effort_levels": ["xhigh", "high"]},
        ),
        ({"supported_efforts": ["max", "high"], "default_effort": "low"}, {"reasoning_effort_levels": ["max", "high"]}),
    ),
)
async def test_model_reasoning_contract_is_not_invented_from_boolean_support(
    reasoning: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"data": {"id": "org/model", "endpoints": [_endpoint()]}}
                if request.url.path.endswith("/endpoints")
                else _catalog(reasoning),
            )
        )
    ) as client:
        handler.client = client
        _, metadata = await _inventory(handler, InMemoryCache())
        assert {
            k: v for k, v in metadata.items() if k in ("reasoning_effort_levels", "default_reasoning_effort")
        } == expected


@pytest.mark.parametrize(
    "failure",
    (
        httpx.Response(401),
        httpx.Response(503),
        httpx.Response(302, headers={"location": "https://foreign.test"}),
        httpx.Response(200, content=b"bad json"),
        httpx.Response(200, json={"data": {"id": "wrong/model", "endpoints": [_endpoint()]}}),
        httpx.Response(200, json={"data": {"id": "org/model", "endpoints": [_endpoint(context_length=True)]}}),
        httpx.Response(200, json={"data": {"id": "org/model", "endpoints": [_endpoint(max_prompt_tokens=-1)]}}),
        httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1)),
        httpx.ReadTimeout("fixture timeout"),
    ),
)
async def test_endpoint_failure_retains_same_identity_metadata_but_never_imports_aggregate_limits(
    failure: httpx.Response | Exception,
) -> None:
    endpoint_response: Final = Mock(
        return_value=httpx.Response(200, json={"data": {"id": "org/model", "endpoints": [_endpoint()]}})
    )

    def respond(request: httpx.Request) -> httpx.Response:
        return endpoint_response() if request.url.path.endswith("/endpoints") else httpx.Response(200, json=_catalog())

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        first, metadata = await _inventory(handler, cache)
        endpoint_response.side_effect = failure if isinstance(failure, Exception) else None
        endpoint_response.return_value = failure
        _, retained = await _inventory(handler, cache, refresh=True, previous=first)
        assert retained == metadata
        _, unavailable = await _inventory(handler, cache)
        assert unavailable == {}
        assert endpoint_response.call_count == 2
        _, other_identity = await _inventory(handler, cache, key="other-key", previous=first)
        assert other_identity == {}


async def test_endpoint_cache_separates_models_credentials_and_bases_and_refreshes() -> None:
    clock: Final = Mock(return_value=0)
    responder: Final = Mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "data": {
                    "id": "org/second" if "/org/second/" in request.url.path else "org/model",
                    "endpoints": [_endpoint()],
                }
            }
            if request.url.path.endswith("/endpoints")
            else _catalog(),
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        cache: Final = InMemoryCache(clock=clock)
        await _inventory(handler, cache)
        await _inventory(handler, cache, only=("supplier",))
        assert responder.call_count == 2
        await _inventory(handler, cache, model="org/second")
        assert responder.call_count == 3
        await _inventory(handler, cache, key="other")
        assert responder.call_count == 5
        await _inventory(handler, cache, base="https://other.test/v1")
        assert responder.call_count == 7
        await _inventory(handler, cache, refresh=True)
        assert responder.call_count == 9
        clock.return_value = MODEL_INFO_REFRESH_SECONDS + 1
        await _inventory(handler, cache)
        assert responder.call_count == 11


@pytest.mark.parametrize("endpoints", ([], [_endpoint("other")]))
async def test_successful_endpoint_absence_clears_last_known_metadata(endpoints: list[Mapping[str, object]]) -> None:
    response: Final = Mock(return_value=[_endpoint()])
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"data": {"id": "org/model", "endpoints": response()}}
                if request.url.path.endswith("/endpoints")
                else _catalog(),
            )
        )
    ) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        first, metadata = await _inventory(handler, cache)
        assert metadata["context_window"] == 1200
        response.return_value = endpoints
        current, absent = await _inventory(handler, cache, refresh=True, previous=first)
        assert absent == {}
        assert "org/model" in current.models


@pytest.mark.parametrize("model", ("org/../escape", "../model", "org/.."))
async def test_inventory_identifiers_cannot_change_endpoint_url_structure(model: str) -> None:
    responder: Final = Mock(return_value=httpx.Response(200, json={"data": [{"id": model}]}))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        _, metadata = await _inventory(handler, InMemoryCache(), model=model)
        assert metadata == {}
        assert responder.call_count == 1
