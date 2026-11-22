import pytest
import respx
from pydantic import ValidationError

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
