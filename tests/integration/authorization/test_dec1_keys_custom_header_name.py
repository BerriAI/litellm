"""`general_settings.litellm_key_header_name`: an operator chosen header that carries the virtual key.

Contract (litellm/proxy/auth/user_api_key_auth.py, `user_api_key_auth_builder` and
`get_api_key_from_custom_header`):

* When the configured header is present on a request, its value replaces whatever key was read from the
  standard headers. The header name is matched case insensitively.
* The value must be `Bearer <key>` (or `bearer `, `Basic `). A raw key without a prefix, or an empty value,
  is a 401 "Malformed API Key" even when `Authorization` holds a valid key.
* When the configured header is absent, the standard headers keep working.

These tests boot one extra proxy for the module with `litellm_key_header_name: X-Dec1-Proxy-Key`, sharing the
suite database and upstream.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration.authorization.dec1_keys_support import (
    MALFORMED_KEY_MESSAGE,
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    assert_served,
    chat_body,
    config_with_general_settings,
    marker,
    send_with_headers,
    upstream_requests,
)

_HEADER: Final = "X-Dec1-Proxy-Key"
_UNKNOWN_KEY: Final = "sk-dec1-keys-custom-header-never-issued"


@pytest.fixture(scope="module")
def custom_header_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-keys-custom-header")
    config: Final = config_with_general_settings(directory, {"litellm_key_header_name": _HEADER})
    with gateway_from_environment() as shared, owned_proxy(shared, directory / "proxy", {}, config=config) as owned:
        yield owned


def test_the_configured_header_with_a_bearer_value_serves_the_request(custom_header_proxy: Gateway) -> None:
    """The caller sends only `X-Dec1-Proxy-Key: Bearer <key>` to a proxy configured with that header name.

    The proxy must authenticate from the configured header and answer 200 with no `Authorization` at all.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(custom_header_proxy, {_HEADER: f"Bearer {key}"}, chat_body(model)))


def test_the_configured_header_is_matched_case_insensitively(custom_header_proxy: Gateway) -> None:
    """The caller sends the configured header in lower case, `x-dec1-proxy-key: Bearer <key>`.

    Header names are case insensitive, so the request must be served with 200.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(custom_header_proxy, {_HEADER.lower(): f"Bearer {key}"}, chat_body(model)))


def test_the_configured_header_requires_a_bearer_prefix(custom_header_proxy: Gateway) -> None:
    """The caller sends the raw key in the configured header without a `Bearer ` prefix.

    Unlike `x-litellm-api-key`, the configured header goes through the strict bearer parser, so the proxy
    must answer 401 with type `auth_error` and the malformed key message, and must not call the upstream.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = marker()
        refused: Final = send_with_headers(custom_header_proxy, {_HEADER: key}, chat_body(model, text))
        assert_denied(refused, status=401, error_type="auth_error", message=MALFORMED_KEY_MESSAGE)
        assert upstream_requests(custom_header_proxy, text) == []


def test_the_configured_header_wins_over_a_valid_authorization(custom_header_proxy: Gateway) -> None:
    """The caller sends an unknown key in the configured header and a valid key in `Authorization`.

    The configured header replaces the `Authorization` key, so the proxy must answer the generic unknown
    key 401 with type `token_not_found_in_db`.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(
            custom_header_proxy,
            {_HEADER: f"Bearer {_UNKNOWN_KEY}", "Authorization": f"Bearer {key}"},
            chat_body(model),
        )
        assert_denied(refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE)


def test_a_valid_configured_header_is_used_when_authorization_holds_a_provider_credential(
    custom_header_proxy: Gateway,
) -> None:
    """The caller sends a valid key in the configured header and an unrelated bearer token in `Authorization`.

    The request must be served with 200. This is the reason the setting exists: the client keeps
    `Authorization` for something else and the proxy key travels in the configured header.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(
            send_with_headers(
                custom_header_proxy,
                {_HEADER: f"Bearer {key}", "Authorization": f"Bearer {_UNKNOWN_KEY}"},
                chat_body(model),
            )
        )


def test_an_empty_configured_header_is_malformed_even_with_a_valid_authorization(custom_header_proxy: Gateway) -> None:
    """The caller sends the configured header with an empty value and a valid key in `Authorization`.

    A present but empty configured header yields an empty key, so the proxy must answer the malformed key
    401 and must not fall back to `Authorization`.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        refused: Final = send_with_headers(
            custom_header_proxy, {_HEADER: "", "Authorization": f"Bearer {key}"}, chat_body(model)
        )
        assert_denied(refused, status=401, error_type="auth_error", message=MALFORMED_KEY_MESSAGE)


def test_authorization_still_works_when_the_configured_header_is_absent(custom_header_proxy: Gateway) -> None:
    """The caller sends only `Authorization: Bearer <key>` to the proxy that has a configured key header.

    Configuring a custom header does not disable the standard one, so the request must be served with 200.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(custom_header_proxy, {"Authorization": f"Bearer {key}"}, chat_body(model)))


def test_x_api_key_still_works_when_the_configured_header_is_absent(custom_header_proxy: Gateway) -> None:
    """The caller sends only `x-api-key: <key>` to the proxy that has a configured key header.

    The provider style headers stay available, so the request must be served with 200.
    """
    with custom_header_proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(send_with_headers(custom_header_proxy, {"x-api-key": key}, chat_body(model)))


def test_the_master_key_is_accepted_in_the_configured_header(custom_header_proxy: Gateway) -> None:
    """The admin sends the master key as `X-Dec1-Proxy-Key: Bearer <master key>` on `/key/generate`.

    The configured header applies to every route, so the management call must answer 200 and return a new
    key, which the test then deletes.
    """
    created: Final = send_with_headers(
        custom_header_proxy, {_HEADER: f"Bearer {custom_header_proxy.key}"}, {}, path="/key/generate"
    )
    assert created.status_code == 200, created.text
    custom_header_proxy.post("/key/delete", {"keys": [created.json()["key"]]})
