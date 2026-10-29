import json
from pathlib import Path
from typing import Final

import pytest

from litellm.llms.bedrock.common_utils import get_bedrock_base_model
from litellm.proxy.common_utils.token_count_settings import (
    ProviderCountFailure,
    TokenCountModel,
    TokenCountSettings,
    count_source_fields,
    resolve_token_count_settings,
)
from litellm.types.utils import TokenCountResponse

_OPUS_US_EAST_1: Final = TokenCountModel(model="anthropic.claude-opus-5-5", region="us-east-1")
_DEFAULT: Final = {
    "token_count_framing": 97,
    "token_count_model": {"model": "anthropic.claude-opus-5-5", "region": "us-east-1"},
}
_COST_MAP: Final = Path(__file__).parents[4] / "model_prices_and_context_window.json"


def _in_us_west_2() -> str:
    return "us-west-2"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1)),
        (
            {"token_count_framing": 5, "token_count_model": {"model": "custom-model", "region": "ap-southeast-4"}},
            TokenCountSettings(
                framing=5,
                count_model=TokenCountModel(model="custom-model", region="ap-southeast-4", configured=True),
            ),
        ),
        (
            {"token_count_model": {"model": "custom-model", "litellm_provider": "bedrock_mantle"}},
            TokenCountSettings(
                framing=97,
                count_model=TokenCountModel(
                    model="custom-model", region=None, litellm_provider="bedrock_mantle", configured=True
                ),
            ),
        ),
        (
            {"token_count_model": {"model": "custom-model"}},
            TokenCountSettings(
                framing=97, count_model=TokenCountModel(model="custom-model", region=None, configured=True)
            ),
        ),
        (
            {"token_count_model": {"model": "custom-model", "region": " "}},
            TokenCountSettings(
                framing=97, count_model=TokenCountModel(model="custom-model", region=None, configured=True)
            ),
        ),
        ({"token_count_framing": 0}, TokenCountSettings(framing=0, count_model=_OPUS_US_EAST_1)),
        (
            {"token_count_framing": None, "token_count_model": None},
            TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1),
        ),
        (
            {"token_count_framing": True, "token_count_model": "custom-model"},
            TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1),
        ),
        ({"token_count_framing": 78.0}, TokenCountSettings(framing=78, count_model=_OPUS_US_EAST_1)),
        ({"token_count_framing": 1.5}, TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1)),
        ({"token_count_framing": -1}, TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1)),
        ({"token_count_framing": "78"}, TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1)),
        ({"token_count_model": {}}, TokenCountSettings(framing=97, count_model=None)),
        ({"token_count_model": ""}, TokenCountSettings(framing=97, count_model=_OPUS_US_EAST_1)),
        (
            {"token_count_model": {"model": "  ", "region": "us-east-1"}},
            TokenCountSettings(framing=97, count_model=None),
        ),
    ],
)
def test_config_overrides_the_cost_map_default_key_by_key(
    configured: dict[str, object], expected: TokenCountSettings
) -> None:
    assert (
        resolve_token_count_settings(configured=configured, default=_DEFAULT, deployment_region=_in_us_west_2)
        == expected
    )


@pytest.mark.parametrize(
    ("deployment_region", "configured", "expected"),
    [
        ("us-west-2", {}, _OPUS_US_EAST_1),
        ("eu-central-1", {}, TokenCountModel(model="anthropic.claude-opus-5-5", region=None)),
        ("ca-central-1", {}, TokenCountModel(model="anthropic.claude-opus-5-5", region=None)),
        ("us-gov-west-1", {}, TokenCountModel(model="anthropic.claude-opus-5-5", region=None)),
        (
            "eu-central-1",
            {"token_count_model": {"model": "anthropic.claude-opus-5-5", "region": "us-east-1"}},
            TokenCountModel(model="anthropic.claude-opus-5-5", region="us-east-1", configured=True),
        ),
    ],
)
def test_shipped_count_region_never_takes_the_prompt_out_of_the_deployment_geography(
    deployment_region: str, configured: dict[str, object], expected: TokenCountModel
) -> None:
    settings: Final = resolve_token_count_settings(
        configured=configured, default=_DEFAULT, deployment_region=lambda: deployment_region
    )
    assert settings.count_model == expected


@pytest.mark.parametrize("prefix", ("", "bedrock/", "bedrock_mantle/"))
def test_shipped_global_count_model_keeps_its_region_for_a_deployment_in_any_geography(prefix: str) -> None:
    model: Final = f"{prefix}global.anthropic.claude-opus-5-5"
    settings: Final = resolve_token_count_settings(
        configured={},
        default={"token_count_model": {"model": model, "region": "us-east-1"}},
        deployment_region=lambda: "eu-west-1",
    )
    assert settings.count_model == TokenCountModel(model=model, region="us-east-1")


def test_shipped_australian_count_region_stays_with_an_australian_deployment() -> None:
    settings: Final = resolve_token_count_settings(
        configured={},
        default={"token_count_model": {"model": "au.anthropic.claude-opus-5-5", "region": "ap-southeast-4"}},
        deployment_region=lambda: "ap-southeast-2",
    )
    assert settings.count_model == TokenCountModel(model="au.anthropic.claude-opus-5-5", region="ap-southeast-4")


_ABSENT: Final = object()


@pytest.mark.parametrize(
    "key", ["token_count_tool_framing", "token_count_responses_framing", "token_count_responses_tool_framing"]
)
@pytest.mark.parametrize(
    ("configured", "default", "expected"),
    [
        (_ABSENT, 151, 151),
        (40, 151, 40),
        (0, 151, 0),
        (40.0, _ABSENT, 40),
        (-1, 151, 151),
        (True, 151, 151),
        ("40", _ABSENT, None),
    ],
)
def test_each_request_kind_framing_resolves_like_framing_and_apart_from_the_others(
    key: str, configured: object, default: object, expected: int | None
) -> None:
    settings: Final = resolve_token_count_settings(
        configured={} if configured is _ABSENT else {key: configured},
        default={"token_count_framing": 97, **({} if default is _ABSENT else {key: default})},
        deployment_region=_in_us_west_2,
    )
    resolved: Final = {
        "token_count_tool_framing": settings.tool_framing,
        "token_count_responses_framing": settings.responses_framing,
        "token_count_responses_tool_framing": settings.responses_tool_framing,
    }
    assert settings.framing == 97
    assert resolved == {name: expected if name == key else None for name in resolved}


_CHAT_ONLY: Final = TokenCountSettings(framing=185, count_model=None, tool_framing=214)
_PER_API: Final = TokenCountSettings(
    framing=149, count_model=None, tool_framing=300, responses_framing=145, responses_tool_framing=104
)


@pytest.mark.parametrize(
    ("settings", "carries_tools", "responses_api", "expected"),
    [
        (TokenCountSettings(framing=None, count_model=None), True, True, 0),
        (TokenCountSettings(framing=23, count_model=None), True, False, 23),
        (TokenCountSettings(framing=23, count_model=None), True, True, 23),
        (_CHAT_ONLY, False, False, 185),
        (_CHAT_ONLY, True, False, 214),
        (_CHAT_ONLY, False, True, 185),
        (_CHAT_ONLY, True, True, 214),
        (_PER_API, False, False, 149),
        (_PER_API, True, False, 300),
        (_PER_API, False, True, 145),
        (_PER_API, True, True, 104),
        (TokenCountSettings(framing=149, count_model=None, tool_framing=0), True, False, 0),
        (TokenCountSettings(framing=149, count_model=None, responses_tool_framing=0), True, True, 0),
        (TokenCountSettings(framing=149, count_model=None, responses_framing=145), True, True, 149),
    ],
)
def test_hidden_prompt_is_the_framing_set_for_the_request_api_and_tools_falling_back_to_chat(
    settings: TokenCountSettings, carries_tools: bool, responses_api: bool, expected: int
) -> None:
    assert settings.hidden_prompt_tokens(carries_tools=carries_tools, responses_api=responses_api) == expected


def test_no_setting_anywhere_means_the_model_counts_itself_without_framing() -> None:
    assert resolve_token_count_settings(
        configured={}, default={}, deployment_region=_in_us_west_2
    ) == TokenCountSettings(framing=None, count_model=None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("kimi_k3", "kimi_k3"), ("cl100k_base", None), (None, None), (True, None)],
)
def test_only_a_tokenizer_litellm_ships_is_taken_from_the_cost_map(value: object, expected: str | None) -> None:
    settings: Final = resolve_token_count_settings(
        configured={}, default={"token_count_tokenizer": value}, deployment_region=_in_us_west_2
    )
    assert settings.tokenizer == expected


def _counted(source: str | None, model_used: str = "us.anthropic.claude-opus-5-5") -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=7,
        request_model="sonnet",
        model_used=model_used,
        tokenizer_type="bedrock_api",
        count_source=source,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, {}),
        ("provider", {"litellm_count_source": "provider"}),
        ("local_estimate", {"litellm_count_source": "local_estimate"}),
        (
            "token_count_model",
            {"litellm_count_source": "token_count_model", "litellm_count_model": "us.anthropic.claude-opus-5-5"},
        ),
    ],
)
def test_count_source_fields_name_the_counting_model_only_when_token_count_model_counted(
    source: str | None, expected: dict[str, str]
) -> None:
    assert count_source_fields(_counted(source)) == expected


def test_every_shipped_token_count_model_resolves_to_a_mapped_aws_model() -> None:
    cost_map: Final = json.loads(_COST_MAP.read_text())
    aws_models: Final = frozenset(
        get_bedrock_base_model(name)
        for name, entry in cost_map.items()
        if isinstance(entry, dict) and entry.get("litellm_provider") in ("bedrock", "bedrock_converse")
    )
    count_models: Final = {
        name: entry["token_count_model"]
        for name, entry in cost_map.items()
        if isinstance(entry, dict) and "token_count_model" in entry
    }
    assert count_models
    for name, count_model in count_models.items():
        model: Final = count_model["model"]
        assert get_bedrock_base_model(model) in aws_models
        assert cost_map[name]["litellm_provider"] in ("bedrock", "bedrock_converse", "bedrock_mantle")
        assert "region" not in count_model or count_model["region"].strip()


_UPSTREAM_404: Final = {
    "type": "error",
    "request_id": "req_upstream",
    "error": {"type": "not_found_error", "message": "The model 'anthropic.claude-opus-5-5' does not exist"},
}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            json.dumps(_UPSTREAM_404),
            {
                **_UPSTREAM_404,
                "error": {
                    "type": "not_found_error",
                    "message": "The model 'anthropic.claude-opus-5-5' does not exist. Enable eu-west-1",
                },
            },
        ),
        (json.dumps({"message": "Too many tokens"}), "Too many tokens. Enable eu-west-1"),
        ("scripted 404", "scripted 404. Enable eu-west-1"),
        (
            "Credential should be scoped to a valid region. ",
            "Credential should be scoped to a valid region. Enable eu-west-1",
        ),
    ],
)
def test_hint_lands_in_the_upstream_error_message_and_keeps_its_envelope(message: str, expected: object) -> None:
    hinted: Final = ProviderCountFailure(status_code=404, message=message).with_hint("Enable eu-west-1")

    assert (json.loads(hinted.message) if isinstance(expected, dict) else hinted.message) == expected
    assert hinted.status_code == 404


def test_hint_stays_off_an_error_raised_before_any_provider_call() -> None:
    message: Final = "Invalid request for token counting: LiteLLM cannot count a tool of type 'mcp' for this model"
    failure: Final = ProviderCountFailure(status_code=400, message=message, rejected_before_provider_call=True)

    assert failure.with_hint("Enable eu-west-1").message == message
