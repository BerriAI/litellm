import hashlib
from typing import Final

import pytest

from litellm.router_utils.clientside_credential_handler import (
    FORWARDED_API_KEY_SCOPE_METADATA_KEY,
    ForwardedApiKeyScope,
    deployment_audience,
    dispatched_audiences,
    forwarded_api_key_scope,
    headers_without_forwarded_api_key,
    is_forwarded_api_key,
    stamped_forwarded_api_key_scope,
)
from litellm.types.router import LiteLLM_Params

_CLIENT_KEY: Final = "sk-ant-api03-client-key"
_ANTHROPIC: Final = {"model": "anthropic/claude-haiku-4-5"}
_ANTHROPIC_GATEWAY: Final = {"model": "anthropic/claude-haiku-4-5", "api_base": "https://gateway.example/anthropic"}
_BEDROCK: Final = {"model": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"}
_ANTHROPIC_SCOPE: Final = ForwardedApiKeyScope(
    audiences=(("anthropic", ""),), key_sha256=hashlib.sha256(_CLIENT_KEY.encode()).hexdigest()
)


@pytest.mark.parametrize(
    "deployment_params",
    [
        (_ANTHROPIC, {"model": "auto_router/claude-router"}),
        (_ANTHROPIC, {"model": "not-a-known-provider-model"}),
        ({"model": "auto_router/complexity_router"},),
        ({"model": "auto_router/adaptive_router"},),
        (),
    ],
)
def test_forwarded_api_key_scope_is_not_stamped_when_audiences_are_unknown_up_front(
    deployment_params: tuple[dict[str, str], ...],
) -> None:
    assert forwarded_api_key_scope(_CLIENT_KEY, deployment_params) is None


def test_forwarded_api_key_scope_names_provider_and_normalized_api_base_with_the_key_hash() -> None:
    scope: Final = forwarded_api_key_scope(
        _CLIENT_KEY, (_BEDROCK, {**_ANTHROPIC_GATEWAY, "api_base": "https://gateway.example/anthropic/"})
    )

    assert scope == ForwardedApiKeyScope(
        audiences=(("anthropic", "https://gateway.example/anthropic"), ("bedrock", "")),
        key_sha256=_ANTHROPIC_SCOPE.key_sha256,
    )


@pytest.mark.parametrize(
    "request_kwargs, expected_audiences",
    [
        ({}, {("anthropic", "")}),
        ({"api_key": _CLIENT_KEY}, {("anthropic", "")}),
        ({"api_base": "https://other-gateway.example/"}, {("anthropic", "https://other-gateway.example")}),
        (
            {"api_base": "https://other-gateway.example/", "api_key": _CLIENT_KEY},
            {("anthropic", "https://other-gateway.example")},
        ),
        (
            {"base_url": "https://other-gateway.example"},
            {("anthropic", ""), ("anthropic", "https://other-gateway.example")},
        ),
    ],
)
def test_dispatched_audiences_apply_the_request_api_base_and_base_url_to_the_deployment(
    request_kwargs: dict[str, object], expected_audiences: set[tuple[str, str]]
) -> None:
    assert dispatched_audiences(_ANTHROPIC, request_kwargs) == expected_audiences


def test_forwarded_api_key_scope_includes_the_api_base_the_client_chose() -> None:
    scope: Final = forwarded_api_key_scope(_CLIENT_KEY, (_ANTHROPIC,), {"api_base": "https://client-gateway.example"})

    assert scope == ForwardedApiKeyScope(
        audiences=(("anthropic", "https://client-gateway.example"),), key_sha256=_ANTHROPIC_SCOPE.key_sha256
    )


@pytest.mark.parametrize(
    "request_kwargs, expected_scope",
    [
        ({"litellm_metadata": {FORWARDED_API_KEY_SCOPE_METADATA_KEY: _ANTHROPIC_SCOPE}}, _ANTHROPIC_SCOPE),
        ({"metadata": {FORWARDED_API_KEY_SCOPE_METADATA_KEY: _ANTHROPIC_SCOPE}}, _ANTHROPIC_SCOPE),
        (
            {"metadata": {FORWARDED_API_KEY_SCOPE_METADATA_KEY: [[["anthropic", ""]], _ANTHROPIC_SCOPE.key_sha256]}},
            None,
        ),
        ({"metadata": {}}, None),
        ({}, None),
    ],
)
def test_stamped_forwarded_api_key_scope_reads_only_a_proxy_built_stamp(
    request_kwargs: dict[str, object], expected_scope: ForwardedApiKeyScope | None
) -> None:
    assert stamped_forwarded_api_key_scope(request_kwargs) is expected_scope


@pytest.mark.parametrize("value, expected", [(_CLIENT_KEY, True), ("admin-fallback-key", False), (None, False)])
def test_is_forwarded_api_key_matches_only_the_stamped_key(value: object, expected: bool) -> None:
    assert is_forwarded_api_key(value, _ANTHROPIC_SCOPE) is expected


@pytest.mark.parametrize(
    "litellm_params, expected_audience",
    [
        (_BEDROCK, ("bedrock", "")),
        (_ANTHROPIC_GATEWAY, ("anthropic", "https://gateway.example/anthropic")),
        (LiteLLM_Params(**_ANTHROPIC_GATEWAY), ("anthropic", "https://gateway.example/anthropic")),
        (
            {"model": "anthropic/claude-haiku-4-5", "api_base": "https://api.anthropic.com/"},
            ("anthropic", "https://api.anthropic.com"),
        ),
    ],
)
def test_deployment_audience_is_provider_and_api_base_without_trailing_slash(
    litellm_params: dict[str, str] | LiteLLM_Params, expected_audience: tuple[str, str]
) -> None:
    assert deployment_audience(litellm_params) == expected_audience


def test_headers_without_forwarded_api_key_drops_only_the_forwarded_keys_x_api_key() -> None:
    request_headers: Final = {"X-Api-Key": _CLIENT_KEY, "anthropic-version": "2023-06-01"}
    request_kwargs: Final = {
        "headers": request_headers,
        "extra_headers": {"x-api-key": "admin-configured-key"},
        "api_key": _CLIENT_KEY,
    }

    stripped: Final = headers_without_forwarded_api_key(request_kwargs, _ANTHROPIC_SCOPE)

    assert stripped == {
        "headers": {"anthropic-version": "2023-06-01"},
        "extra_headers": {"x-api-key": "admin-configured-key"},
    }
    assert request_headers == {"X-Api-Key": _CLIENT_KEY, "anthropic-version": "2023-06-01"}
