import hashlib

import pytest

from litellm.router_utils.clientside_credential_handler import (
    FORWARDED_OAUTH_CREDENTIAL_KEY,
    forwarded_oauth_credential_fingerprint,
    get_dynamic_litellm_params,
    is_clientside_credential,
)

_TOKEN = "sk-ant-oat01-caller-seat"
_FINGERPRINT = hashlib.sha256(_TOKEN.encode()).hexdigest()


def _scoped(providers: str, headers: dict[str, str]) -> dict[str, object]:
    return {"custom_llm_provider": providers, "extra_headers": headers}


@pytest.mark.parametrize(
    ("provider_specific_header", "custom_llm_provider", "expected"),
    [
        pytest.param(
            _scoped("anthropic", {"authorization": f"Bearer {_TOKEN}"}),
            "anthropic",
            _FINGERPRINT,
            id="anthropic bearer",
        ),
        pytest.param(
            _scoped("anthropic", {"Authorization": f"Bearer {_TOKEN}"}),
            "anthropic",
            _FINGERPRINT,
            id="header name case",
        ),
        pytest.param(
            [
                _scoped("anthropic,bedrock,vertex_ai", {"anthropic-beta": "oauth-2025-04-20"}),
                _scoped("anthropic", {"authorization": f"Bearer {_TOKEN}"}),
            ],
            "anthropic",
            _FINGERPRINT,
            id="scoped list as the proxy sends it",
        ),
        pytest.param(
            _scoped("anthropic", {"authorization": f"Bearer {_TOKEN}"}), "bedrock", None, id="bedrock deployment"
        ),
        pytest.param(
            _scoped("bedrock", {"authorization": f"Bearer {_TOKEN}"}),
            "anthropic",
            None,
            id="header scoped to another provider",
        ),
        pytest.param(
            _scoped("anthropic", {"authorization": "Bearer sk-ant-api03-key"}),
            "anthropic",
            None,
            id="api key, not oauth",
        ),
        pytest.param(
            [
                _scoped("anthropic", {"authorization": f"Bearer {_TOKEN}"}),
                _scoped("anthropic", {"authorization": "Bearer sk-ant-oat01-other"}),
            ],
            "anthropic",
            None,
            id="two bearers",
        ),
        pytest.param({"custom_llm_provider": "anthropic"}, "anthropic", None, id="malformed entry"),
        pytest.param(None, "anthropic", None, id="no header"),
    ],
)
def test_forwarded_oauth_credential_fingerprint(
    provider_specific_header: object, custom_llm_provider: str, expected: str | None
) -> None:
    fingerprint = forwarded_oauth_credential_fingerprint(
        {"provider_specific_header": provider_specific_header}, custom_llm_provider
    )

    assert fingerprint == expected


def test_forwarded_oauth_bearer_gets_its_own_deployment_identity_without_storing_the_token() -> None:
    request = {"provider_specific_header": _scoped("anthropic", {"authorization": f"Bearer {_TOKEN}"})}

    dynamic_params = get_dynamic_litellm_params(
        litellm_params={"model": "anthropic/claude-opus-5-5"}, request_kwargs=request, custom_llm_provider="anthropic"
    )

    assert is_clientside_credential(request, custom_llm_provider="anthropic") is True
    assert is_clientside_credential(request, custom_llm_provider="bedrock") is False
    assert dynamic_params == {"model": "anthropic/claude-opus-5-5", FORWARDED_OAUTH_CREDENTIAL_KEY: _FINGERPRINT}
    assert _TOKEN not in repr(dynamic_params)
