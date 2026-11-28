"""No caller credential and no internal header reaches a provider when header forwarding is switched on.

Contract (litellm/proxy/litellm_pre_call_utils.py `clean_headers`, `_get_forwardable_headers`,
`add_headers_to_llm_call`, `get_sanitized_user_information_from_key`, `add_headers_to_llm_call_by_model_group`):

* `general_settings.forward_client_headers_to_llm_api: true` copies every `x-*` request header (except
  `x-stainless-*`) and `anthropic-beta*` to the provider, after `clean_headers` removed the headers the proxy
  reads credentials from: `Authorization`, `API-Key`, `x-api-key`, `x-goog-api-key`,
  `Ocp-Apim-Subscription-Key`, `x-litellm-api-key`, the MCP headers, and the custom
  `general_settings.litellm_key_header_name`. Whatever header carried the caller's key, the provider sees only
  the deployment key.
* `litellm_settings.add_user_information_to_llm_headers: true` adds `x-litellm-user_api_key_*` headers: the key
  appears only as its SHA-256 (`x-litellm-user_api_key_hash`), never raw.
* `forward_llm_provider_auth_headers: true` forwards a provider `x-api-key` the caller adds on purpose, but never
  the header the caller authenticated with.
* `model_group_settings.forward_client_headers_to_llm_api: [<model>]` forwards for the listed model only.

What `x-` headers in general are forwarded, and the Rust peer header, are pinned in
`compatibility/test_dec1_http_forwarding_settings.py`; this file covers the credential carriers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.wire import Request
from integration.security.dec1_authsec_support import (
    CHAT,
    DEPLOYMENT_KEY,
    OWNED_MODEL,
    SECOND_MODEL,
    Door,
    assert_served,
    chat_body,
    digest,
    header_values_carrying,
    owned_door,
    raw_post,
)

CUSTOM_KEY_HEADER: Final = "x-dec1-proxy-key"

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 180),
    pytest.mark.xdist_group("dec1-authsec-forwarded-credentials"),
]


@pytest.fixture(scope="module")
def forwarding(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Door]:
    with owned_door(
        tmp_path_factory.mktemp("dec1-authsec-forwarding"),
        general_settings={"forward_client_headers_to_llm_api": True, "litellm_key_header_name": CUSTOM_KEY_HEADER},
        litellm_settings={"add_user_information_to_llm_headers": True},
    ) as door:
        yield door


@pytest.fixture(scope="module")
def provider_auth(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Door]:
    with owned_door(
        tmp_path_factory.mktemp("dec1-authsec-provider-auth"),
        general_settings={"forward_client_headers_to_llm_api": True, "forward_llm_provider_auth_headers": True},
    ) as door:
        yield door


@pytest.fixture(scope="module")
def per_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Door]:
    with owned_door(
        tmp_path_factory.mktemp("dec1-authsec-per-model"),
        litellm_settings={"model_group_settings": {"forward_client_headers_to_llm_api": [OWNED_MODEL]}},
    ) as door:
        yield door


def _one_call(door: Door) -> Request:
    (call,) = door.calls()
    return call


def _served_through(door: Door, headers: dict[str, str], model: str = OWNED_MODEL) -> Request:
    door.calls()
    assert_served(raw_post(door.gateway, CHAT, chat_body(model), headers))
    return _one_call(door)


@pytest.mark.parametrize(
    "header",
    ["Authorization", "API-Key", "x-api-key", "x-goog-api-key", "Ocp-Apim-Subscription-Key", CUSTOM_KEY_HEADER],
)
def test_the_key_never_reaches_the_provider_whatever_header_carried_it(forwarding: Door, header: str) -> None:
    """With forwarding on, a virtual key is sent in one credential header (as `Bearer <key>` for `Authorization`
    and the custom key header, raw for the others). The call must be served, the provider must receive
    `Authorization: Bearer <deployment key>`, and the virtual key must not appear in any provider header value.
    The `x-` names are the dangerous ones: the forwarding rule would copy them if `clean_headers` missed them."""
    with forwarding.gateway.scenario() as scenario:
        key: Final = scenario.key()
        value: Final = f"Bearer {key}" if header in ("Authorization", CUSTOM_KEY_HEADER) else key
        call: Final = _served_through(forwarding, {header: value})
    assert call.headers.get("authorization") == f"Bearer {DEPLOYMENT_KEY}"
    assert [name for name, raw in call.headers.items() if key in raw] == []


def test_the_custom_key_header_is_not_forwarded_by_name(forwarding: Door) -> None:
    """With `litellm_key_header_name: x-dec1-proxy-key` and forwarding on, the caller authenticates with that
    header. The provider must receive no `x-dec1-proxy-key` header at all."""
    with forwarding.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(forwarding, {CUSTOM_KEY_HEADER: f"Bearer {key}"})
    assert CUSTOM_KEY_HEADER not in call.headers


def test_mcp_credential_headers_are_not_forwarded_to_an_llm_provider(forwarding: Door) -> None:
    """With forwarding on, the caller adds `x-mcp-auth: dec1-mcp-secret` and `x-mcp-servers: dec1` to a chat
    call. The provider must receive neither header: both are in `SpecialHeaders`, which `clean_headers` drops."""
    with forwarding.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(
            forwarding, {"Authorization": f"Bearer {key}", "x-mcp-auth": "dec1-mcp-secret", "x-mcp-servers": "dec1"}
        )
    assert "x-mcp-auth" not in call.headers and "x-mcp-servers" not in call.headers
    assert [name for name, raw in call.headers.items() if "dec1-mcp-secret" in raw] == []


def test_user_information_headers_carry_the_key_hash_never_the_key(forwarding: Door) -> None:
    """With `add_user_information_to_llm_headers: true`, a key with alias `dec1-authsec-alias` calls a chat model.
    The provider must receive `x-litellm-user_api_key_hash` equal to the key's SHA-256 and
    `x-litellm-user_api_key_alias: dec1-authsec-alias`, and the raw key must not appear in any header value."""
    with forwarding.gateway.scenario() as scenario:
        key: Final = scenario.key(key_alias="dec1-authsec-alias")
        call: Final = _served_through(forwarding, {"Authorization": f"Bearer {key}"})
    assert call.headers.get("x-litellm-user_api_key_hash") == digest(key)
    assert call.headers.get("x-litellm-user_api_key_alias") == "dec1-authsec-alias"
    assert [name for name, raw in call.headers.items() if key in raw] == []


def test_user_information_headers_strip_callback_secrets_from_key_metadata(forwarding: Door) -> None:
    """A key whose metadata holds a logging callback config with a Langfuse secret
    (`metadata.logging: [{"callback_name": "langfuse", "callback_vars": {"langfuse_secret_key": <secret>}}]`) calls a
    chat model with user information headers on. No provider header value may contain the secret
    (`strip_callback_config` before `x-litellm-user_api_key_auth_metadata` is built). UNVERIFIED: that
    `/key/generate` accepts this metadata shape on the suite proxy."""
    secret: Final = "sk-lf-dec1-authsec-callback-secret"
    with forwarding.gateway.scenario() as scenario:
        key: Final = scenario.key(
            metadata={
                "logging": [
                    {
                        "callback_name": "langfuse",
                        "callback_type": "success",
                        "callback_vars": {"langfuse_public_key": "pk-lf-dec1", "langfuse_secret_key": secret},
                    }
                ]
            }
        )
        call: Final = _served_through(forwarding, {"Authorization": f"Bearer {key}"})
    assert header_values_carrying(call, secret) == []


def test_the_master_key_never_reaches_the_provider_with_forwarding_on(forwarding: Door) -> None:
    """The master key calls a chat model through each of `Authorization` and `x-api-key` with forwarding and user
    information headers on. Neither the master key nor its SHA-256 may appear in any provider header value; the
    master key's logged identity is an alias, not its hash."""
    for headers in ({"Authorization": f"Bearer {forwarding.gateway.key}"}, {"x-api-key": forwarding.gateway.key}):
        call = _served_through(forwarding, headers)
        assert header_values_carrying(call, forwarding.gateway.key) == [], headers


def test_a_provider_x_api_key_is_forwarded_when_the_caller_authenticated_with_authorization(
    provider_auth: Door,
) -> None:
    """With `forward_llm_provider_auth_headers: true`, the caller authenticates with `Authorization: Bearer <key>`
    and adds `x-api-key: sk-dec1-byok-provider`. The provider must receive `Authorization: Bearer
    sk-dec1-byok-provider` (the client key overrides the deployment key as `api_key`), and the virtual key must not
    appear in any header value."""
    with provider_auth.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(
            provider_auth, {"Authorization": f"Bearer {key}", "x-api-key": "sk-dec1-byok-provider"}
        )
    assert call.headers.get("authorization") == "Bearer sk-dec1-byok-provider"
    assert [name for name, raw in call.headers.items() if key in raw] == []


def test_the_x_api_key_used_to_authenticate_is_not_forwarded_as_the_provider_key(provider_auth: Door) -> None:
    """With `forward_llm_provider_auth_headers: true`, the caller authenticates with `x-api-key: <virtual key>`
    and sends nothing else. The provider must receive the deployment key, and the virtual key must not appear in
    any provider header value: `clean_headers` drops the header the caller authenticated with."""
    with provider_auth.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(provider_auth, {"x-api-key": key})
    assert call.headers.get("authorization") == f"Bearer {DEPLOYMENT_KEY}"
    assert [name for name, raw in call.headers.items() if key in raw] == []


def test_the_callers_authorization_is_never_forwarded_even_with_provider_auth_forwarding(
    provider_auth: Door,
) -> None:
    """With `forward_llm_provider_auth_headers: true`, a virtual key calls through `Authorization` and also sends
    `x-goog-api-key: dec1-goog-provider-key`. The provider's `authorization` must be the deployment key, and the
    virtual key must not appear anywhere in the provider headers."""
    with provider_auth.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(
            provider_auth, {"Authorization": f"Bearer {key}", "x-goog-api-key": "dec1-goog-provider-key"}
        )
    assert call.headers.get("authorization") == f"Bearer {DEPLOYMENT_KEY}"
    assert [name for name, raw in call.headers.items() if key in raw] == []


def test_per_model_forwarding_applies_to_the_listed_model(per_model: Door) -> None:
    """With `model_group_settings.forward_client_headers_to_llm_api: [<owned model>]`, the caller sends
    `x-dec1-trace: listed` on a call to that model. The provider must receive `x-dec1-trace: listed`."""
    call: Final = _served_through(
        per_model, {"Authorization": f"Bearer {per_model.gateway.key}", "x-dec1-trace": "listed"}
    )
    assert call.headers.get("x-dec1-trace") == "listed"


def test_per_model_forwarding_does_not_apply_to_another_model(per_model: Door) -> None:
    """Same proxy; the caller sends `x-dec1-trace: unlisted` on a call to the second model, which is not listed.
    The provider must not receive `x-dec1-trace`."""
    call: Final = _served_through(
        per_model, {"Authorization": f"Bearer {per_model.gateway.key}", "x-dec1-trace": "unlisted"}, SECOND_MODEL
    )
    assert "x-dec1-trace" not in call.headers


def test_per_model_forwarding_never_forwards_the_key_header(per_model: Door) -> None:
    """Same proxy; a virtual key calls the listed model through `x-litellm-api-key: Bearer <key>`. The provider
    must receive no `x-litellm-api-key` header and the key must not appear in any header value."""
    with per_model.gateway.scenario() as scenario:
        key: Final = scenario.key()
        call: Final = _served_through(per_model, {"x-litellm-api-key": f"Bearer {key}"})
    assert "x-litellm-api-key" not in call.headers
    assert [name for name, raw in call.headers.items() if key in raw] == []
