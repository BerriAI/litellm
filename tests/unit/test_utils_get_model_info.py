import json
import re
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import Final, Literal, cast

import httpx
import pytest

import litellm
from litellm import CustomLLM
from litellm.llms.bedrock.common_utils import BedrockModelInfo
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ModelInfoBase
from litellm.utils import _invalidate_model_cost_lowercase_map, supports_function_calling


@pytest.fixture(autouse=True)
def isolate_model_info_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    monkeypatch.setattr(litellm, "custom_provider_map", list(litellm.custom_provider_map))
    yield
    _invalidate_model_cost_lowercase_map()


def test_get_model_info_simple_model_name() -> None:
    model: Final = "claude-opus-5-5"
    info: Final = litellm.get_model_info(model)

    assert info["key"]
    assert info["litellm_provider"] == "anthropic"


def test_get_model_info_custom_llm_with_model_name() -> None:
    model: Final = "anthropic/claude-opus-5-5"
    info: Final = litellm.get_model_info(model)

    assert info["key"]
    assert info["litellm_provider"] == "anthropic"


def test_get_model_info_custom_llm_with_same_name_vllm() -> None:
    model: Final = "command-r-plus"
    provider: Final = "openai"
    litellm.register_model(
        {
            "openai/command-r-plus": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        },
        persist_across_reloads=False,
    )
    info: Final = litellm.get_model_info(model, custom_llm_provider=provider)

    assert info["input_cost_per_token"] == 0.0


def test_get_model_info_ollama_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.ollama.completion.transformation import OllamaConfig

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload: Final = cast(dict[str, object], json.loads(request.content))
        assert request.url.path == "/api/show"
        assert payload == {"name": "unknown-model"}
        return httpx.Response(
            200,
            json={
                "model_info": {"llama.context_length": 32768},
                "template": "tools",
            },
            request=request,
        )

    transport: Final = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as http_client:
        monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=http_client))
        config_info: Final = OllamaConfig().get_model_info("unknown-model")
        assert config_info is not None
        model_info: Final = litellm.get_model_info("ollama/unknown-model")

    assert config_info["supports_function_calling"] is True
    assert config_info["max_tokens"] == 32768
    assert model_info["supports_function_calling"] is True
    assert model_info["max_tokens"] == 32768


def test_get_model_info_bedrock_region(monkeypatch: pytest.MonkeyPatch) -> None:
    regional_model: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost_without_regional_entry: Final = {
        key: value for key, value in litellm.get_model_cost_map(url="").items() if key != regional_model
    }
    monkeypatch.setattr(litellm, "model_cost", model_cost_without_regional_entry)
    _invalidate_model_cost_lowercase_map()
    info: Final = litellm.get_model_info(model=regional_model, custom_llm_provider="bedrock")

    assert info["key"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
    assert info["litellm_provider"] == "bedrock_converse"


@pytest.mark.parametrize(
    "model",
    [
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "ft:gpt-4-0613:my-org:custom_suffix:id",
        "ft:davinci-002:my-org:custom_suffix:id",
        "ft:babbage-002:my-org:custom_suffix:id",
        "gpt-35-turbo",
        "ada",
    ],
)
def test_get_model_info_completion_cost_unit_tests(model: str) -> None:
    info: Final = litellm.get_model_info(model)

    assert info["key"]


def test_get_model_info_ft_model_with_provider_prefix() -> None:
    info: Final = litellm.get_model_info(
        model="openai/ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        custom_llm_provider="openai",
    )

    assert info["key"] == "ft:gpt-3.5-turbo"


def _enforce_bedrock_converse_models(
    model_cost: Mapping[str, ModelInfoBase], whitelist_models: Collection[str]
) -> None:
    for model, info in model_cost.items():
        if (
            info.get("litellm_provider") == "bedrock"
            and info.get("mode") == "chat"
            and model not in whitelist_models
            and not (
                (base_model := BedrockModelInfo.get_base_model(model)) != model
                and model_cost.get(base_model, {}).get("litellm_provider") == "bedrock_converse"
                and BedrockModelInfo.get_bedrock_route(model) == "converse"
            )
        ):
            raise AssertionError(f"Unlisted Bedrock chat model does not route to Converse: {model}")


def _read_whitelisted_bedrock_models() -> tuple[str, ...]:
    path: Final = Path(__file__).resolve().parents[2] / "whitelisted_bedrock_models.txt"
    return tuple(path.read_text().splitlines())


def _normalize_bedrock_model_key(model_key: str) -> str:
    without_wildcard: Final = model_key.replace("*/", "")
    return re.sub(r"(?:1-month-commitment|3-month-commitment|6-month-commitment)/", "", without_wildcard)


def test_model_info_bedrock_converse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost: Final = cast(Mapping[str, ModelInfoBase], litellm.get_model_cost_map(url=""))
    assert any(info.get("litellm_provider") == "bedrock_converse" for info in model_cost.values())

    _enforce_bedrock_converse_models(
        model_cost=model_cost,
        whitelist_models=_read_whitelisted_bedrock_models(),
    )


def test_model_info_bedrock_converse_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost: Final = cast(Mapping[str, ModelInfoBase], litellm.get_model_cost_map(url=""))
    model_cost_with_unlisted: Final = {
        **model_cost,
        "fake.bedrock-chat-model": cast(
            ModelInfoBase,
            {
                "litellm_provider": "bedrock",
                "mode": "chat",
            },
        ),
    }

    with pytest.raises(AssertionError, match=re.escape("fake.bedrock-chat-model")):
        _enforce_bedrock_converse_models(
            model_cost=model_cost_with_unlisted,
            whitelist_models=_read_whitelisted_bedrock_models(),
        )


@pytest.mark.parametrize("region", ("us-gov-east-1", "us-gov-west-1"))
@pytest.mark.parametrize("base_provider", ("bedrock_converse", "bedrock"))
def test_regional_bedrock_alias_requires_canonical_converse_metadata(
    region: str, base_provider: Literal["bedrock_converse", "bedrock"]
) -> None:
    base_model: Final = next(
        model for model in sorted(litellm.bedrock_converse_models) if BedrockModelInfo.get_base_model(model) == model
    )
    model: Final = f"bedrock/{region}/{base_model}"
    model_cost: Final[Mapping[str, ModelInfoBase]] = {
        model: {"litellm_provider": "bedrock", "mode": "chat"},
        base_model: {"litellm_provider": base_provider, "mode": "chat"},
    }
    assert BedrockModelInfo.get_bedrock_route(model) == "converse"
    if base_provider == "bedrock":
        with pytest.raises(AssertionError, match=re.escape(model)):
            _enforce_bedrock_converse_models(model_cost, ())
        return
    _enforce_bedrock_converse_models(model_cost, ())


def test_get_model_info_bedrock_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost: Final = cast(Mapping[str, ModelInfoBase], litellm.get_model_cost_map(url=""))
    bedrock_entries: Final = tuple(
        (model_key, model_info)
        for model_key, model_info in model_cost.items()
        if model_info.get("litellm_provider") == "bedrock"
    )

    assert bedrock_entries
    for model_key, model_info in bedrock_entries:
        normalized_key: Final = _normalize_bedrock_model_key(model_key)
        base_model: Final = BedrockModelInfo.get_base_model(normalized_key)
        base_model_key: Final = (
            base_model if base_model in model_cost else f"bedrock/{base_model}"
        )
        if base_model_key not in model_cost or "invoke/" in normalized_key:
            continue
        base_model_info: Final = model_cost[base_model_key]
        capability_values: Final = tuple(
            (key, value) for key, value in base_model_info.items() if key.startswith("supports_")
        )
        for capability, value in capability_values:
            assert capability in model_info, f"{capability} is not in model cost map for {model_key}"
            assert model_info[capability] == value, f"{capability} differs for model {model_key}"


def _cross_region_base_model_key(
    model_key: str,
    prefixes: tuple[str, ...],
    model_cost: Mapping[str, Mapping[str, object]],
) -> str | None:
    matched_prefix: Final = next((prefix for prefix in prefixes if model_key.startswith(prefix)), None)
    if matched_prefix is None:
        return None
    base_model_key: Final = model_key[len(matched_prefix) :]
    return base_model_key if base_model_key in model_cost else None


def _cross_region_profiles(
    model_cost: Mapping[str, Mapping[str, object]],
    prefixes: tuple[str, ...],
) -> tuple[tuple[str, Mapping[str, object], str], ...]:
    return tuple(
        (model_key, model_info, base_model_key)
        for model_key, model_info in model_cost.items()
        if str(model_info.get("litellm_provider", "")).startswith("bedrock")
        and (
            base_model_key := _cross_region_base_model_key(model_key, prefixes, model_cost)
        )
        is not None
    )


def test_get_model_info_bedrock_cross_region_capability_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost: Final[Mapping[str, Mapping[str, object]]] = litellm.get_model_cost_map(url="")
    profiles: Final = _cross_region_profiles(model_cost, ("us.", "eu.", "apac.", "us-gov."))

    assert profiles, "no cross-region bedrock profiles found"
    for model_key, model_info, base_model_key in profiles:
        base_model_info: Final = model_cost[base_model_key]
        capabilities: Final = tuple(
            (key, value) for key, value in base_model_info.items() if key.startswith("supports_")
        )
        for capability, base_value in capabilities:
            assert capability in model_info, f"{capability} is on {base_model_key} but missing from {model_key}"
            assert model_info.get(capability) == base_value, f"{capability} differs for {model_key}"


def _is_positive_cost(value: object) -> bool:
    return isinstance(value, (int, float)) and value > 0


def test_get_model_info_bedrock_priced_cross_region_profile_has_priced_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    model_cost: Final[Mapping[str, Mapping[str, object]]] = litellm.get_model_cost_map(url="")
    profiles: Final = _cross_region_profiles(model_cost, ("us.", "eu.", "apac.", "us-gov.", "au.", "global."))

    assert profiles, "no cross-region bedrock profiles found"
    for model_key, model_info, base_model_key in profiles:
        base_model_info: Final = model_cost[base_model_key]
        for cost_key in ("input_cost_per_token", "output_cost_per_token"):
            if _is_positive_cost(model_info.get(cost_key)):
                assert _is_positive_cost(base_model_info.get(cost_key)), (
                    f"{model_key} charges {cost_key} but its base {base_model_key} is free"
                )


def test_get_model_info_case_insensitive_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.register_model(
        {
            "together_ai/Qwen/Qwen3-Next-80B-A3B-Thinking": {
                "input_cost_per_token": 0.0001,
                "output_cost_per_token": 0.0002,
                "litellm_provider": "together_ai",
                "supports_function_calling": True,
            }
        },
        persist_across_reloads=False,
    )

    model_names: Final = (
        "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "qwen/qwen3-next-80b-a3b-thinking",
        "QWEN/qwen3-NEXT-80b-a3b-thinking",
    )
    for model_name in model_names:
        info: Final = litellm.get_model_info(model=model_name, custom_llm_provider="together_ai")
        assert info["supports_function_calling"] is True


def test_get_model_info_case_insensitive_supports_function_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.register_model(
        {
            "test_provider/TestModel-ABC": {
                "input_cost_per_token": 0.0001,
                "output_cost_per_token": 0.0002,
                "litellm_provider": "test_provider",
                "supports_function_calling": True,
            }
        },
        persist_across_reloads=False,
    )

    assert supports_function_calling("TestModel-ABC", custom_llm_provider="test_provider") is True
    assert supports_function_calling("testmodel-abc", custom_llm_provider="test_provider") is True


def test_get_model_info_custom_model_router() -> None:
    from litellm import Router

    Router(
        model_list=[
            {
                "model_name": "ma-summary",
                "litellm_params": {
                    "api_base": "http://ma-mix-llm-serving.cicero.svc.cluster.local/v1",
                    "input_cost_per_token": 1,
                    "output_cost_per_token": 1,
                    "model": "openai/meta-llama/Meta-Llama-3-8B-Instruct",
                },
                "model_info": {
                    "id": "c20d603e-1166-4e0f-aa65-ed9c476ad4ca",
                },
            }
        ]
    )
    info: Final = litellm.get_model_info("c20d603e-1166-4e0f-aa65-ed9c476ad4ca")

    assert info["key"] == "c20d603e-1166-4e0f-aa65-ed9c476ad4ca"
    assert info["input_cost_per_token"] == 1
    assert info["output_cost_per_token"] == 1


def test_get_model_info_custom_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomProviderHandler:
        def completion(self, *_args: object, **_kwargs: object) -> litellm.ModelResponse:
            return litellm.ModelResponse(
                id="mock-completion",
                created=0,
                model="gpt-3.5-turbo",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi!"},
                    }
                ],
            )

    custom_handler: Final = cast(CustomLLM, CustomProviderHandler())
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": "my-custom-llm", "custom_handler": custom_handler}],
    )
    response: Final = litellm.completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )
    assert response.choices[0].message.content == "Hi!"

    litellm.register_model(
        {"my-custom-llm/my-fake-model": {"max_tokens": 2048}},
        persist_across_reloads=False,
    )
    info: Final = litellm.get_model_info(model="my-custom-llm/my-fake-model")

    assert info["max_tokens"] == 2048
