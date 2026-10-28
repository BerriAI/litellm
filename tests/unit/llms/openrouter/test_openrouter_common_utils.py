from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.openrouter.common_utils import OpenRouterModelInfo

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


@respx.mock
def test_get_models_lists_both_catalogs_once_each_under_openrouter_keeping_the_org_namespace():
    respx.get(f"{DEFAULT_BASE}/models/user").mock(
        return_value=_catalog("openai/gpt-5.4-mini", "anthropic/claude-haiku-5-5")
    )
    respx.get(f"{DEFAULT_BASE}/embeddings/models").mock(
        return_value=_catalog("openai/text-embedding-3-small", "openai/gpt-5.4-mini")
    )

    assert OpenRouterModelInfo().get_models(api_key="sk-or-test") == [
        "openrouter/openai/gpt-5.4-mini",
        "openrouter/anthropic/claude-haiku-5-5",
        "openrouter/openai/text-embedding-3-small",
    ]


@respx.mock
def test_get_models_keeps_the_api_base_path_and_sends_the_deployment_key():
    chat_route: Final = respx.get("https://gateway.example/openrouter/api/v1/models/user").mock(
        return_value=_catalog("openai/gpt-5.4-mini")
    )
    embeddings_route: Final = respx.get("https://gateway.example/openrouter/api/v1/embeddings/models").mock(
        return_value=_catalog()
    )

    assert OpenRouterModelInfo().get_models(
        api_key="sk-or-deployment", api_base="https://gateway.example/openrouter/api/v1/"
    ) == ["openrouter/openai/gpt-5.4-mini"]
    assert [route.calls.last.request.headers["authorization"] for route in (chat_route, embeddings_route)] == [
        "Bearer sk-or-deployment",
        "Bearer sk-or-deployment",
    ]


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
    respx.get(f"{DEFAULT_BASE}/embeddings/models").mock(return_value=_catalog())

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
    respx.get(f"{expected_base}/embeddings/models").mock(return_value=_catalog())

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


@pytest.mark.parametrize("failing_route", ["models/user", "embeddings/models"])
def test_get_models_raises_when_either_catalog_answers_an_error(failing_route: str):
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get(f"{DEFAULT_BASE}/models/user").mock(return_value=_catalog("openai/gpt-5.4-mini"))
        respx_mock.get(f"{DEFAULT_BASE}/embeddings/models").mock(return_value=_catalog())
        respx_mock.get(f"{DEFAULT_BASE}/{failing_route}").mock(
            return_value=httpx.Response(401, json={"error": {"code": 401}})
        )

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
