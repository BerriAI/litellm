from typing import Final

import httpx
import pytest
import respx
import time_machine
from pydantic import ValidationError

import litellm
from litellm.constants import FAILED_MODEL_INFO_LOOKUP_RETRY_SECONDS
from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig

LEMONADE_BASE = "http://lemonade.test/api/v1"
MODEL = "Qwen3-4B-GGUF"


def _expected_model_info(**overrides: object) -> dict[str, object]:
    return {
        "key": f"lemonade/{MODEL}",
        "litellm_provider": "lemonade",
        "mode": "chat",
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        **overrides,
    }


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            {"object": "list", "data": [{"id": MODEL, "recipe_options": {"ctx_size": [4096]}}, {"id": "nomic-embed"}]},
            [f"lemonade/{MODEL}", "lemonade/nomic-embed"],
            id="listed-models",
        ),
        pytest.param({"data": []}, [], id="empty-list"),
        pytest.param({"object": "list"}, [], id="data-key-missing"),
        pytest.param({"data": ""}, [], id="data-empty-string"),
        pytest.param({"data": {}}, [], id="data-empty-object"),
        pytest.param({"data": [{"id": ""}]}, ["lemonade/"], id="empty-id"),
    ],
)
def test_get_models_prefixes_each_listed_id(
    respx_mock: respx.MockRouter, body: dict[str, object], expected: list[str]
) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models").respond(200, json=body)

    assert LemonadeChatConfig().get_models(api_base=LEMONADE_BASE) == expected


def test_get_models_raises_key_error_for_entry_without_id(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models").respond(200, json={"data": [{"id": MODEL}, {"name": "no-id"}]})

    with pytest.raises(KeyError, match="id"):
        LemonadeChatConfig().get_models(api_base=LEMONADE_BASE)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param([{"id": MODEL}], id="top-level-list"),
        pytest.param({"data": None}, id="data-null"),
        pytest.param({"data": [MODEL]}, id="entry-not-an-object"),
        pytest.param({"data": [{"id": 7}]}, id="id-not-a-string"),
    ],
)
def test_get_models_rejects_malformed_listing(respx_mock: respx.MockRouter, body: object) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models").respond(200, json=body)

    with pytest.raises(ValidationError):
        LemonadeChatConfig().get_models(api_base=LEMONADE_BASE)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param({}, _expected_model_info(), id="empty-object"),
        pytest.param(
            {"max_tokens": "2048", "max_output_tokens": 4096, "max_input_tokens": 8192},
            _expected_model_info(max_tokens=2048, max_output_tokens=4096, max_input_tokens=8192),
            id="numeric-string-and-ints",
        ),
        pytest.param(
            {"max_output_tokens": 4096},
            _expected_model_info(max_tokens=4096, max_output_tokens=4096),
            id="max-tokens-falls-back-to-output",
        ),
        pytest.param(
            {"max_tokens": True, "max_output_tokens": 0, "max_input_tokens": 12.5},
            _expected_model_info(),
            id="bool-zero-and-float-ignored",
        ),
        pytest.param(
            {"max_tokens": None, "max_output_tokens": "many", "recipe_options": [1], "provider_specific_entry": "x"},
            _expected_model_info(provider_specific_entry={"recipe_options": [1]}),
            id="null-text-and-non-object-options-ignored",
        ),
        pytest.param(
            {"provider_specific_entry": {"recipe_options": {"ctx_size": "32768"}, "labels": ["hot"]}},
            _expected_model_info(
                max_input_tokens=32768,
                provider_specific_entry={"recipe_options": {"ctx_size": "32768"}, "labels": ["hot"]},
            ),
            id="nested-entry-kept",
        ),
    ],
)
def test_get_model_info_reads_token_limits_from_server_payload(
    respx_mock: respx.MockRouter, body: dict[str, object], expected: dict[str, object]
) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models/{MODEL}").respond(200, json=body)

    assert LemonadeChatConfig().get_model_info(model=f"lemonade/{MODEL}", api_base=LEMONADE_BASE) == expected


@pytest.mark.parametrize("body", [pytest.param([{"max_tokens": 5}], id="list"), pytest.param("loaded", id="string")])
def test_get_model_info_raises_for_non_object_payload_instead_of_defaulting(
    respx_mock: respx.MockRouter, body: object
) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models/{MODEL}").respond(200, json=body)

    with pytest.raises(ValidationError):
        LemonadeChatConfig().get_model_info(model=f"lemonade/{MODEL}", api_base=LEMONADE_BASE)


def test_get_model_info_defaults_when_server_errors(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{LEMONADE_BASE}/models/{MODEL}").respond(503, json={"max_tokens": 5})

    assert (
        LemonadeChatConfig().get_model_info(model=f"lemonade/{MODEL}", api_base=LEMONADE_BASE) == _expected_model_info()
    )


def test_per_call_lookups_retry_a_failed_lookup_at_the_call_api_base_with_its_key(
    respx_mock: respx.MockRouter,
) -> None:
    from litellm.utils import get_model_info_for_call

    lookups: Final = respx_mock.get(f"{LEMONADE_BASE}/models/{MODEL}").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"max_input_tokens": 4096})]
    )

    def max_input_tokens() -> object:
        return get_model_info_for_call(f"lemonade/{MODEL}", None, LEMONADE_BASE, "lemonade-call-key")["max_input_tokens"]

    litellm.get_model_info.cache_clear()
    try:
        with time_machine.travel(0, tick=False) as clock:
            failed: Final = max_input_tokens()
            within_window: Final = max_input_tokens()
            clock.shift(FAILED_MODEL_INFO_LOOKUP_RETRY_SECONDS)
            retried: Final = max_input_tokens()
            cached: Final = max_input_tokens()
    finally:
        litellm.get_model_info.cache_clear()

    assert (failed, within_window, retried, cached) == (None, None, 4096, 4096)
    assert [call.request.headers.get("authorization") for call in lookups.calls] == ["Bearer lemonade-call-key"] * 2


@pytest.mark.respx(assert_all_called=False)
def test_completion_looks_up_the_model_next_to_the_call_api_base(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from litellm.utils import cached_get_model_info_helper

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    respx_mock.post(f"{LEMONADE_BASE}/chat/completions").respond(
        200,
        json={
            "id": "chatcmpl-lemonade",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )
    lookups: Final = respx_mock.get(f"{LEMONADE_BASE}/models/{MODEL}").respond(200, json={"max_input_tokens": 4096})
    litellm.get_model_info.cache_clear()
    cached_get_model_info_helper.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as success_logging:
            monkeypatch.setattr("litellm.utils.executor", success_logging)
            litellm.completion(
                model=f"lemonade/{MODEL}", messages=[{"role": "user", "content": "hi"}], api_base=LEMONADE_BASE
            )
    finally:
        litellm.get_model_info.cache_clear()
        cached_get_model_info_helper.cache_clear()

    assert lookups.called
    assert {call.request.url.path for call in respx_mock.calls} == {
        "/api/v1/chat/completions",
        f"/api/v1/models/{MODEL}",
    }


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        pytest.param({}, {}, id="nothing-supplied"),
        pytest.param(
            {"repeat_penalty": 0.0, "n": 0, "stop": "", "tools": []},
            {"repeat_penalty": 0.0, "n": 0, "stop": "", "tools": []},
            id="falsy-values-are-kept",
        ),
        pytest.param({"top_k": 40, "top_p": None, "max_tokens": None}, {"top_k": 40}, id="none-is-skipped"),
    ],
)
def test_init_stores_each_supplied_value_except_none_as_class_level_config(
    supplied: dict[str, object], stored: dict[str, object]
) -> None:
    class ScopedLemonadeConfig(LemonadeChatConfig):
        pass

    ScopedLemonadeConfig(**supplied)

    assert ScopedLemonadeConfig.get_config() == stored


def test_init_stores_the_supplied_object_itself_rather_than_a_copy() -> None:
    class ScopedLemonadeConfig(LemonadeChatConfig):
        pass

    logit_bias = {"50256": -100}

    ScopedLemonadeConfig(logit_bias=logit_bias)

    assert ScopedLemonadeConfig.get_config()["logit_bias"] is logit_bias


def test_init_called_again_with_none_keeps_the_value_stored_earlier() -> None:
    class ScopedLemonadeConfig(LemonadeChatConfig):
        pass

    ScopedLemonadeConfig(top_k=40)
    ScopedLemonadeConfig(top_k=None, n=2)

    assert ScopedLemonadeConfig.get_config() == {"top_k": 40, "n": 2}


@pytest.mark.respx(assert_all_called=False)
def test_the_max_token_check_looks_up_a_mapped_model_at_the_call_api_base(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from litellm.utils import cached_get_model_info_helper

    mapped_model: Final = next(
        name.removeprefix("lemonade/")
        for name, entry in litellm.model_cost.items()
        if entry.get("litellm_provider") == "lemonade" and entry.get("max_output_tokens")
    )
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LEMONADE_API_BASE", "http://lemonade-default.test/api/v1")
    monkeypatch.setattr(litellm, "modify_params", True)
    respx_mock.post(f"{LEMONADE_BASE}/chat/completions").respond(
        200,
        json={
            "id": "chatcmpl-lemonade",
            "object": "chat.completion",
            "created": 0,
            "model": mapped_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )
    respx_mock.get(url__regex=r"/models/").respond(200, json={"max_input_tokens": 4096})
    litellm.get_model_info.cache_clear()
    cached_get_model_info_helper.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as success_logging:
            monkeypatch.setattr("litellm.utils.executor", success_logging)
            litellm.completion(
                model=f"lemonade/{mapped_model}",
                messages=[{"role": "user", "content": "hi"}],
                api_base=LEMONADE_BASE,
                max_tokens=16,
            )
    finally:
        litellm.get_model_info.cache_clear()
        cached_get_model_info_helper.cache_clear()

    assert {call.request.url.host for call in respx_mock.calls} == {"lemonade.test"}
