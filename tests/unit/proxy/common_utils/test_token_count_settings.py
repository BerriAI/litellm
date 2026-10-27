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
            TokenCountSettings(framing=5, count_model=TokenCountModel(model="custom-model", region="ap-southeast-4")),
        ),
        (
            {"token_count_model": {"model": "custom-model", "litellm_provider": "bedrock_mantle"}},
            TokenCountSettings(
                framing=97,
                count_model=TokenCountModel(model="custom-model", region=None, litellm_provider="bedrock_mantle"),
            ),
        ),
        (
            {"token_count_model": {"model": "custom-model"}},
            TokenCountSettings(framing=97, count_model=TokenCountModel(model="custom-model", region=None)),
        ),
        (
            {"token_count_model": {"model": "custom-model", "region": " "}},
            TokenCountSettings(framing=97, count_model=TokenCountModel(model="custom-model", region=None)),
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
    ("deployment_region", "configured", "expected_region"),
    [
        ("us-west-2", {}, "us-east-1"),
        ("eu-central-1", {}, None),
        ("ca-central-1", {}, None),
        ("us-gov-west-1", {}, None),
        (
            "eu-central-1",
            {"token_count_model": {"model": "anthropic.claude-opus-5-5", "region": "us-east-1"}},
            "us-east-1",
        ),
    ],
)
def test_shipped_count_region_never_takes_the_prompt_out_of_the_deployment_geography(
    deployment_region: str, configured: dict[str, object], expected_region: str | None
) -> None:
    settings: Final = resolve_token_count_settings(
        configured=configured, default=_DEFAULT, deployment_region=lambda: deployment_region
    )
    assert settings.count_model == TokenCountModel(model="anthropic.claude-opus-5-5", region=expected_region)


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
