"""Where the proxy looks for a virtual key on an LLM route, and which header wins when several are sent.

Contract (litellm/proxy/auth/user_api_key_auth.py `get_api_key`, `get_bearer_token`):

* The key is read from the first of these that is present, in this order: `x-litellm-api-key`,
  `Authorization`, `API-Key`, `x-api-key`, `x-goog-api-key`, `Ocp-Apim-Subscription-Key`. Later headers
  are ignored once an earlier one is present, even when the earlier one holds a bad key.
* `Authorization` must carry a `Bearer `, `bearer ` or `Basic ` prefix (or an AWS SigV4 credential that
  embeds `Bearer <key>`). Any other shape is a 401 "Malformed API Key".
* `x-litellm-api-key` accepts the key with or without a `Bearer ` prefix. The provider style headers take
  the raw key only.
* The virtual key is never forwarded upstream. The upstream sees the deployment's own provider key.

The Rust `gateway-auth` crate currently ships a `Bearer` only extractor, so every non `Authorization`
form in this file is a behaviour the port has to add.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_keys_support import (
    MALFORMED_KEY_MESSAGE,
    NO_KEY_MESSAGE,
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    assert_served,
    chat_body,
    marker,
    send_with_headers,
    upstream_requests,
)

_UNKNOWN_KEY: Final = "sk-dec1-keys-this-key-was-never-issued"
_RAW_KEY_HEADERS: Final = ("API-Key", "x-api-key", "x-goog-api-key", "Ocp-Apim-Subscription-Key", "x-litellm-api-key")


def test_authorization_bearer_serves_the_request(gateway: Gateway) -> None:
    """A key is created and sent as `Authorization: Bearer <key>` on chat completions.

    The proxy must answer 200 with one choice. This is the baseline every other header form is compared to.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"Authorization": f"Bearer {key}"}, chat_body(model)))


def test_authorization_with_a_lowercase_bearer_prefix_serves_the_request(gateway: Gateway) -> None:
    """The caller sends `Authorization: bearer <key>` with a lowercase scheme.

    The proxy must strip the lowercase prefix and answer 200. A strict RFC parser in a rewrite that only
    accepts `Bearer` would turn working clients into 401s.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"Authorization": f"bearer {key}"}, chat_body(model)))


def test_authorization_with_an_uppercase_bearer_scheme_is_malformed(gateway: Gateway) -> None:
    """The caller sends `Authorization: BEARER <key>` for a valid key.

    Only `Bearer ` and `bearer ` are recognised, so the proxy must answer 401 with type `auth_error` and
    the malformed key message, even though the key itself is valid.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(gateway, {"Authorization": f"BEARER {key}"}, chat_body(model))
        assert_denied(refused, status=401, error_type="auth_error", message=MALFORMED_KEY_MESSAGE)


def test_authorization_without_a_scheme_prefix_is_malformed(gateway: Gateway) -> None:
    """The caller sends the raw key as the whole `Authorization` value, with no `Bearer ` prefix.

    The proxy must answer 401 with exactly
    `{"error": {"message": "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
    "type": "auth_error", "param": null, "code": "401"}}` and must not call the upstream.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = marker()
        refused: Final = send_with_headers(gateway, {"Authorization": key}, chat_body(model, text))
        assert refused.status_code == 401, refused.text
        assert refused.json() == {
            "error": {"message": MALFORMED_KEY_MESSAGE, "type": "auth_error", "param": None, "code": "401"}
        }, refused.text
        assert upstream_requests(gateway, text) == []


def test_authorization_basic_prefix_is_accepted_as_a_key_carrier(gateway: Gateway) -> None:
    """The caller sends `Authorization: Basic <key>` where the value after `Basic ` is the raw virtual key.

    The proxy strips the `Basic ` prefix without base64 decoding and authenticates the remainder, so the
    request must be served with 200. This exists for Langfuse style clients and is easy to lose in a port.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"Authorization": f"Basic {key}"}, chat_body(model)))


def test_an_aws_sigv4_authorization_carrying_a_bearer_credential_serves_the_request(gateway: Gateway) -> None:
    """The caller sends an AWS SigV4 `Authorization` whose `Credential=` field is `Bearer <key>/date/region/...`.

    The proxy must extract the key from the credential field and answer 200. LangChain Bedrock clients
    pointed at the proxy authenticate this way.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        signed: Final = (
            f"AWS4-HMAC-SHA256 Credential=Bearer {key}/20260101/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-date, Signature=0000"
        )
        assert_served(send_with_headers(gateway, {"Authorization": signed}, chat_body(model)))


def test_an_aws_sigv4_authorization_with_a_bare_credential_serves_the_request(gateway: Gateway) -> None:
    """The caller sends an AWS SigV4 `Authorization` whose `Credential=` field starts with the raw key, no `Bearer`.

    The proxy falls back to the text between `Credential=` and the first `/` and must answer 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        signed: Final = (
            f"AWS4-HMAC-SHA256 Credential={key}/20260101/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-date, Signature=0000"
        )
        assert_served(send_with_headers(gateway, {"Authorization": signed}, chat_body(model)))


@pytest.mark.parametrize("header", _RAW_KEY_HEADERS)
def test_each_provider_style_header_carries_the_raw_key(gateway: Gateway, header: str) -> None:
    """The caller sends no `Authorization` and puts the raw key in one provider style header.

    `API-Key` (Azure), `x-api-key` (Anthropic), `x-goog-api-key` (Google AI Studio),
    `Ocp-Apim-Subscription-Key` (Azure APIM) and `x-litellm-api-key` must each authenticate a chat
    completion with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {header: key}, chat_body(model)))


@pytest.mark.parametrize("header", _RAW_KEY_HEADERS)
def test_each_provider_style_header_refuses_an_unknown_key_with_the_generic_401(gateway: Gateway, header: str) -> None:
    """The caller sends a never issued `sk-` key in one provider style header and no `Authorization`.

    Each header must lead to the same 401 as `Authorization` does: type `token_not_found_in_db`, param
    `key`, code "401" and the generic message that does not echo the key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        refused: Final = send_with_headers(gateway, {header: _UNKNOWN_KEY}, chat_body(model))
        error: Final = assert_denied(
            refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE
        )
        assert _UNKNOWN_KEY not in str(error)


def test_x_litellm_api_key_accepts_a_bearer_prefixed_value(gateway: Gateway) -> None:
    """The caller sends `x-litellm-api-key: Bearer <key>`.

    Unlike the provider style headers, this header strips an optional `Bearer ` prefix, so the request
    must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"x-litellm-api-key": f"Bearer {key}"}, chat_body(model)))


def test_x_api_key_does_not_strip_a_bearer_prefix(gateway: Gateway) -> None:
    """The caller sends `x-api-key: Bearer <key>` for a valid key.

    Provider style headers are used verbatim, so the value does not start with `sk-` and the proxy must
    answer 401 with type `auth_error` and a message that starts with `LiteLLM Virtual Key expected.`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(gateway, {"x-api-key": f"Bearer {key}"}, chat_body(model))
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert str(error["message"]).startswith("LiteLLM Virtual Key expected. Received="), refused.text
        assert key not in refused.text


def test_x_litellm_api_key_wins_over_a_valid_authorization_header(gateway: Gateway) -> None:
    """The caller sends an unknown key in `x-litellm-api-key` and a valid key in `Authorization`.

    `x-litellm-api-key` supersedes `Authorization`, so the proxy must answer the generic unknown key 401
    and must not fall back to the valid `Authorization` key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(
            gateway, {"x-litellm-api-key": _UNKNOWN_KEY, "Authorization": f"Bearer {key}"}, chat_body(model)
        )
        assert_denied(refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE)


def test_a_valid_x_litellm_api_key_is_used_even_when_authorization_holds_an_unknown_key(gateway: Gateway) -> None:
    """The caller sends a valid key in `x-litellm-api-key` and an unknown key in `Authorization`.

    The request must be served with 200. Clients that must send a provider credential in `Authorization`
    rely on this to pass the proxy key separately.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(
            send_with_headers(
                gateway, {"x-litellm-api-key": key, "Authorization": f"Bearer {_UNKNOWN_KEY}"}, chat_body(model)
            )
        )


def test_authorization_wins_over_x_api_key(gateway: Gateway) -> None:
    """The caller sends an unknown key in `Authorization` and a valid key in `x-api-key`.

    `Authorization` is read before `x-api-key`, so the proxy must answer the generic unknown key 401.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(
            gateway, {"Authorization": f"Bearer {_UNKNOWN_KEY}", "x-api-key": key}, chat_body(model)
        )
        assert_denied(refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE)


def test_a_malformed_authorization_is_not_rescued_by_a_valid_x_api_key(gateway: Gateway) -> None:
    """The caller sends `Authorization: <garbage without Bearer>` and a valid key in `x-api-key`.

    A non empty `Authorization` is the chosen source even when unusable, so the proxy must answer the
    malformed key 401 and must not fall through to `x-api-key`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(
            gateway, {"Authorization": "Token not-a-bearer-value", "x-api-key": key}, chat_body(model)
        )
        assert_denied(refused, status=401, error_type="auth_error", message=MALFORMED_KEY_MESSAGE)


def test_an_empty_authorization_falls_through_to_x_api_key(gateway: Gateway) -> None:
    """The caller sends an empty `Authorization` header and a valid key in `x-api-key`.

    An empty `Authorization` counts as absent, so the proxy must use `x-api-key` and answer 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"Authorization": "", "x-api-key": key}, chat_body(model)))


def test_api_key_header_wins_over_x_api_key(gateway: Gateway) -> None:
    """The caller sends an unknown key in `API-Key` and a valid key in `x-api-key`, with no `Authorization`.

    The Azure style `API-Key` is read before the Anthropic style `x-api-key`, so the proxy must answer
    the generic unknown key 401.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(gateway, {"API-Key": _UNKNOWN_KEY, "x-api-key": key}, chat_body(model))
        assert_denied(refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE)


def test_x_api_key_wins_over_x_goog_api_key(gateway: Gateway) -> None:
    """The caller sends a valid key in `x-api-key` and an unknown key in `x-goog-api-key`.

    `x-api-key` is read before `x-goog-api-key`, so the request must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(gateway, {"x-api-key": key, "x-goog-api-key": _UNKNOWN_KEY}, chat_body(model)))


def test_a_key_query_parameter_is_ignored_on_openai_routes(gateway: Gateway) -> None:
    """The caller sends no credential header and puts a valid key in `?key=` on `/v1/chat/completions`.

    `?key=` is only honoured on Google generateContent routes, so the proxy must answer 401 with type
    `auth_error` and "Authentication Error, No api key passed in.".
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(gateway, {}, chat_body(model), params={"key": key})
        assert_denied(refused, status=401, error_type="auth_error", message=NO_KEY_MESSAGE)


def test_a_key_in_the_request_body_is_not_a_credential(gateway: Gateway) -> None:
    """The caller sends no credential header and puts a valid key in the JSON body as `api_key`.

    The body is never a credential source, so the proxy must answer the "No api key passed in." 401 and
    must not call the upstream.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = marker()
        refused: Final = send_with_headers(gateway, {}, chat_body(model, text, api_key=key))
        assert_denied(refused, status=401, error_type="auth_error", message=NO_KEY_MESSAGE)
        assert upstream_requests(gateway, text) == []


def test_the_virtual_key_never_reaches_the_upstream_provider(gateway: Gateway) -> None:
    """A served chat completion is inspected on the scripted upstream.

    The upstream must see `Authorization: Bearer integration-provider-key` (the deployment's own key) and
    neither the raw virtual key nor its SHA-256 anywhere in the forwarded request.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = marker()
        assert_served(send_with_headers(gateway, {"Authorization": f"Bearer {key}"}, chat_body(model, text)))
        forwarded: Final = upstream_requests(gateway, text)
        assert len(forwarded) == 1, forwarded
        assert forwarded[0]["authorization"] == "Bearer integration-provider-key", forwarded
        assert key not in str(forwarded[0]), forwarded


def test_the_virtual_key_in_x_api_key_is_not_forwarded_as_the_provider_key(gateway: Gateway) -> None:
    """A chat completion authenticated through `x-api-key` is inspected on the scripted upstream.

    The upstream request must carry the deployment's provider key and must not contain the virtual key in
    its `authorization` or `api_key` observation. A pass-through of client headers would leak it.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = marker()
        assert_served(send_with_headers(gateway, {"x-api-key": key}, chat_body(model, text)))
        forwarded: Final = upstream_requests(gateway, text)
        assert len(forwarded) == 1, forwarded
        assert (forwarded[0]["authorization"], forwarded[0]["api_key"]) == (
            "Bearer integration-provider-key",
            "",
        ), forwarded
