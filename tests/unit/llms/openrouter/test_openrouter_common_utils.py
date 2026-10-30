from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openrouter.common_utils import (
    OpenRouterModelInfo,
    aget_openrouter_catalog_limits,
    openrouter_catalog_limits_for_model,
)

DEFAULT_BASE: Final = "https://openrouter.ai/api/v1"
KEY_SOURCES: Final = ("api_key", "litellm.api_key", "litellm.openrouter_key", "OPENROUTER_API_KEY", "OR_API_KEY")
BASE_SOURCES: Final = ("api_base", "litellm.api_base", "OPENROUTER_API_BASE")


@pytest.fixture(autouse=True)
def _no_ambient_openrouter_settings(monkeypatch: pytest.MonkeyPatch):
    for env_var in ("OPENROUTER_API_BASE", "OPENROUTER_API_KEY", "OR_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "openrouter_key", None)


def _catalog(*model_ids: str) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": model_id, "name": model_id} for model_id in model_ids]})


def _configure(monkeypatch: pytest.MonkeyPatch, source: str, value: str) -> None:
    if source.startswith("litellm."):
        monkeypatch.setattr(litellm, source.removeprefix("litellm."), value)
    else:
        monkeypatch.setenv(source, value)


def test_get_models_lists_only_the_chat_catalog_under_openrouter_keeping_the_org_namespace():
    with respx.mock() as respx_mock:
        respx_mock.get(f"{DEFAULT_BASE}/models/user").mock(
            return_value=_catalog("openai/gpt-5.4-mini", "anthropic/claude-haiku-5-5")
        )

        assert OpenRouterModelInfo().get_models(api_key="sk-or-test") == [
            "openrouter/openai/gpt-5.4-mini",
            "openrouter/anthropic/claude-haiku-5-5",
        ]
        assert [call.request.url.path for call in respx_mock.calls] == ["/api/v1/models/user"]


@respx.mock
def test_get_models_drops_batch_only_ids_and_keeps_every_other_variant():
    respx.get(f"{DEFAULT_BASE}/models/user").mock(
        return_value=_catalog("openai/gpt-5.4-mini:batch", "openai/gpt-5.4-mini", "qwen/qwen3-coder:free")
    )

    assert OpenRouterModelInfo().get_models(api_key="sk-or-test") == [
        "openrouter/openai/gpt-5.4-mini",
        "openrouter/qwen/qwen3-coder:free",
    ]


@pytest.mark.parametrize(
    "api_base",
    [
        "https://gateway.example/openrouter/api/v1/",
        "https://gateway.example/openrouter/api/v1/chat/completions",
        "https://gateway.example/openrouter/api/v1/chat/completions/",
    ],
)
@respx.mock
def test_get_models_keeps_the_api_base_path_and_sends_the_deployment_key(api_base: str):
    chat_route: Final = respx.get("https://gateway.example/openrouter/api/v1/models/user").mock(
        return_value=_catalog("openai/gpt-5.4-mini")
    )

    assert OpenRouterModelInfo().get_models(api_key="sk-or-deployment", api_base=api_base) == [
        "openrouter/openai/gpt-5.4-mini"
    ]
    assert chat_route.calls.last.request.headers["authorization"] == "Bearer sk-or-deployment"


@pytest.mark.parametrize("winner", range(len(KEY_SOURCES)), ids=KEY_SOURCES)
@respx.mock
def test_get_models_sends_the_first_configured_key_in_the_chat_paths_order(
    monkeypatch: pytest.MonkeyPatch, winner: int
):
    for source in KEY_SOURCES[winner + 1 :]:
        _configure(monkeypatch, source, f"sk-or-from-{source}")
    if winner > 0:
        _configure(monkeypatch, KEY_SOURCES[winner], "sk-or-winner")
    chat_route: Final = respx.get(f"{DEFAULT_BASE}/models/user").mock(return_value=_catalog("qwen/qwen3-coder"))

    assert OpenRouterModelInfo().get_models(api_key="sk-or-winner" if winner == 0 else None) == [
        "openrouter/qwen/qwen3-coder"
    ]
    assert chat_route.calls.last.request.headers["authorization"] == "Bearer sk-or-winner"


@pytest.mark.parametrize("winner", range(len(BASE_SOURCES) + 1), ids=(*BASE_SOURCES, "default"))
@respx.mock
def test_get_models_calls_the_first_configured_base_in_the_chat_paths_order(
    monkeypatch: pytest.MonkeyPatch, winner: int
):
    for source in BASE_SOURCES[winner + 1 :]:
        _configure(monkeypatch, source, f"https://{source.lower().replace('.', '-')}.example/api/v1")
    if 0 < winner < len(BASE_SOURCES):
        _configure(monkeypatch, BASE_SOURCES[winner], "https://winner.example/api/v1")
    expected_base: Final = "https://winner.example/api/v1" if winner < len(BASE_SOURCES) else DEFAULT_BASE
    respx.get(f"{expected_base}/models/user").mock(return_value=_catalog("qwen/qwen3-coder"))

    assert OpenRouterModelInfo().get_models(
        api_key="sk-or-test", api_base="https://winner.example/api/v1" if winner == 0 else None
    ) == ["openrouter/qwen/qwen3-coder"]


def test_get_models_without_any_key_raises_before_calling_openrouter():
    with respx.mock(assert_all_called=False) as respx_mock:
        chat_route: Final = respx_mock.get(f"{DEFAULT_BASE}/models/user").mock(
            return_value=_catalog("openai/gpt-5.4-mini")
        )

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterModelInfo().get_models()

    assert not chat_route.called


@respx.mock
def test_get_models_raises_when_the_catalog_answers_an_error():
    respx.get(f"{DEFAULT_BASE}/models/user").mock(return_value=httpx.Response(401, json={"error": {"code": 401}}))

    with pytest.raises(httpx.HTTPStatusError):
        OpenRouterModelInfo().get_models(api_key="sk-or-revoked")


def test_validate_environment_adds_the_resolved_key_and_keeps_the_callers_headers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    headers: Final = {"X-Title": "my-app"}

    assert OpenRouterModelInfo().validate_environment(
        headers=headers, model="openrouter/openai/gpt-5.4-mini", messages=(), optional_params={}, litellm_params={}
    ) == {"X-Title": "my-app", "Authorization": "Bearer sk-or-env"}
    assert headers == {"X-Title": "my-app"}


def test_validate_environment_without_any_key_sends_no_authorization():
    assert OpenRouterModelInfo().validate_environment(
        headers={"X-Title": "my-app"},
        model="openrouter/openai/gpt-5.4-mini",
        messages=(),
        optional_params={},
        litellm_params={},
    ) == {"X-Title": "my-app"}


@pytest.mark.parametrize(
    ("model", "base_model"),
    [("openrouter/openai/gpt-5.4-mini", "openai/gpt-5.4-mini"), ("openai/gpt-5.4-mini", "openai/gpt-5.4-mini")],
)
def test_get_base_model_drops_only_the_openrouter_prefix(model: str, base_model: str):
    assert OpenRouterModelInfo.get_base_model(model) == base_model


@respx.mock
def test_get_models_keeps_listing_when_a_catalog_row_carries_unusable_limits():
    respx.get(f"{DEFAULT_BASE}/models/user").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "vendor/junk-limits", "context_length": "many", "top_provider": "none"},
                    {"id": "vendor/junk-cap", "top_provider": {"max_completion_tokens": "many"}},
                    {"id": "vendor/no-limits"},
                ]
            },
        )
    )

    assert OpenRouterModelInfo().get_models(api_key="sk-or-test") == [
        "openrouter/vendor/junk-limits",
        "openrouter/vendor/junk-cap",
        "openrouter/vendor/no-limits",
    ]


@pytest.mark.parametrize(
    ("row", "expected"),
    (
        (
            {"context_length": 4096, "top_provider": {"max_completion_tokens": 1024}},
            {"max_tokens": 1024, "max_input_tokens": 4096, "max_output_tokens": 1024},
        ),
        (
            {"context_length": 4096, "top_provider": {"max_completion_tokens": None}},
            {"max_tokens": 4096, "max_input_tokens": 4096},
        ),
        ({"context_length": 4096}, {"max_tokens": 4096, "max_input_tokens": 4096}),
        (
            {"context_length": 4096, "top_provider": {"context_length": 8192, "max_completion_tokens": 1024}},
            {"max_tokens": 1024, "max_input_tokens": 4096, "max_output_tokens": 1024},
        ),
        ({"top_provider": {"max_completion_tokens": 1024}}, {"max_tokens": 1024, "max_output_tokens": 1024}),
        ({"context_length": 0, "top_provider": {"max_completion_tokens": True}}, {}),
        ({"context_length": "many", "top_provider": "none"}, {}),
    ),
)
async def test_catalog_row_becomes_the_published_limits_and_never_invents_an_output_cap(
    row: Mapping[str, object], expected: Mapping[str, int]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{DEFAULT_BASE}/models/user"
        assert request.headers["authorization"] == "Bearer sk-or-test"
        return httpx.Response(200, json={"data": [{"id": "vendor/model", **row}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        catalog: Final = await aget_openrouter_catalog_limits(
            api_base=DEFAULT_BASE,
            headers={"Authorization": "Bearer sk-or-test"},
            client=handler,
            cache=InMemoryCache(),
        )
    assert catalog == {"vendor/model": expected}


async def test_a_failed_catalog_fetch_is_cached_so_the_next_refresh_does_not_call_again() -> None:
    responder: Final = Mock(return_value=httpx.Response(503))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        for _ in range(2):
            assert (
                await aget_openrouter_catalog_limits(api_base=DEFAULT_BASE, headers={}, client=handler, cache=cache)
                == {}
            )
    assert responder.call_count == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("openrouter/vendor/model", {"max_input_tokens": 1}),
        ("vendor/model", {"max_input_tokens": 1}),
        ("openrouter/vendor/model:nitro", {"max_input_tokens": 1}),
        ("openrouter/vendor/model:free", {"max_input_tokens": 2}),
        ("openrouter/auto", {"max_input_tokens": 3}),
        ("openrouter/vendor/unlisted", {}),
        ("openrouter/vendor/unlisted:nitro", {}),
    ),
)
def test_catalog_lookup_finds_a_model_by_its_id_then_by_the_id_without_its_variant(
    model: str, expected: Mapping[str, int]
) -> None:
    catalog: Final = {
        "vendor/model": {"max_input_tokens": 1},
        "vendor/model:free": {"max_input_tokens": 2},
        "openrouter/auto": {"max_input_tokens": 3},
    }

    assert openrouter_catalog_limits_for_model(catalog, model) == expected
