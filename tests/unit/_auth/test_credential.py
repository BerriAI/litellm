from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.credential import (
    KeySource,
    PresentedKeys,
    SelectedKey,
    WebSocketDenial,
    WebSocketKeys,
    litellm_header_token,
    scheme_credential,
    select_api_key,
    select_websocket_api_key,
)

AWS4_BEARER: Final = (
    "AWS4-HMAC-SHA256 Credential=Bearer token/date/region/service/aws4_request, SignedHeaders=host, Signature=sig"
)
AWS4_RAW: Final = (
    "AWS4-HMAC-SHA256 Credential=token/date/region/service/aws4_request, SignedHeaders=host, Signature=sig"
)

PRECEDENCE: Final = (
    ("custom_header", KeySource.CUSTOM_HEADER, "Bearer {}"),
    ("litellm_header", KeySource.LITELLM_HEADER, "Bearer {}"),
    ("authorization", KeySource.AUTHORIZATION, "Bearer {}"),
    ("api_key", KeySource.API_KEY, "{}"),
    ("x_api_key", KeySource.X_API_KEY, "{}"),
    ("google", KeySource.GOOGLE, "{}"),
    ("apim", KeySource.APIM, "{}"),
    ("query", KeySource.QUERY, "{}"),
    ("pass_through", KeySource.PASS_THROUGH, "{}"),
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Bearer token", "token"),
        ("bearer token", "token"),
        ("Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        (AWS4_BEARER, "token"),
        (AWS4_RAW, "token"),
        ("Bearer ", ""),
        ("Bearer  token", " token"),
        ("BEARER token", "token"),
        ("BASIC dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("AWS4-HMAC-SHA256 SignedHeaders=host", None),
        ("token", None),
        ("Bearer", None),
        ("Token token", None),
        ("Bearertoken", None),
    ],
)
def test_scheme_credential_strips_a_known_scheme_in_any_case(value: str, expected: str | None):
    assert scheme_credential(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("token", "token"),
        ("Bearer token", "token"),
        (AWS4_BEARER, "token"),
        ("AWS4-HMAC-SHA256 malformed", "AWS4-HMAC-SHA256 malformed"),
        ("Bearer ", ""),
    ],
)
def test_litellm_header_strips_a_known_scheme_or_keeps_the_raw_value(value: str, expected: str):
    assert litellm_header_token(value) == expected


@pytest.mark.parametrize("rank", range(len(PRECEDENCE)))
def test_higher_priority_source_wins_over_every_lower_one(rank: int):
    higher, source, template = PRECEDENCE[rank]
    winner: Final = template.format(higher)
    losers: Final = {field: lower_template.format("loser") for field, _, lower_template in PRECEDENCE[rank + 1 :]}
    assert select_api_key(PresentedKeys(**losers, **{higher: winner})) == SelectedKey(
        source=source, presented=winner, token=higher
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (PresentedKeys(custom_header="custom", x_api_key="lower"), ""),
        (PresentedKeys(custom_header="", x_api_key="lower"), ""),
        (PresentedKeys(authorization="token", x_api_key="lower"), ""),
        (PresentedKeys(authorization="Token x", x_api_key="lower"), ""),
    ],
)
def test_selected_source_without_a_known_scheme_yields_an_empty_token(keys: PresentedKeys, expected: str):
    selected: Final = select_api_key(keys)
    assert selected is not None
    assert selected.token == expected


@pytest.mark.parametrize(
    ("keys", "expected_source"),
    [
        (PresentedKeys(authorization="", x_api_key="lower"), KeySource.X_API_KEY),
        (PresentedKeys(query="", apim="lower"), KeySource.APIM),
        (PresentedKeys(litellm_header=""), KeySource.LITELLM_HEADER),
        (PresentedKeys(api_key=""), KeySource.API_KEY),
    ],
)
def test_only_empty_authorization_and_query_fall_through(keys: PresentedKeys, expected_source: KeySource):
    selected: Final = select_api_key(keys)
    assert selected is not None
    assert selected.source is expected_source


@pytest.mark.parametrize("keys", [PresentedKeys(), PresentedKeys(authorization=""), PresentedKeys(query="")])
def test_nothing_presented_selects_nothing(keys: PresentedKeys):
    assert select_api_key(keys) is None


def test_presented_keeps_the_raw_value_of_the_selected_source():
    keys: Final = PresentedKeys(authorization=AWS4_BEARER)
    assert select_api_key(keys) == SelectedKey(KeySource.AUTHORIZATION, AWS4_BEARER, "token")
    assert select_api_key(replace(keys, custom_header="Basic abc")) == SelectedKey(
        KeySource.CUSTOM_HEADER, "Basic abc", "abc"
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (WebSocketKeys(authorization="Bearer token"), "token"),
        (WebSocketKeys(authorization="bearer token "), "token"),
        (
            WebSocketKeys(authorization="Bearer token", api_key="other", subprotocols="openai-insecure-api-key.x"),
            "token",
        ),
        (WebSocketKeys(authorization="Basic token"), WebSocketDenial.INVALID_AUTHORIZATION),
        (WebSocketKeys(authorization="token"), WebSocketDenial.INVALID_AUTHORIZATION),
        (WebSocketKeys(authorization="Basic token", api_key="other"), WebSocketDenial.INVALID_AUTHORIZATION),
        (WebSocketKeys(api_key="token"), "token"),
        (WebSocketKeys(authorization="", api_key="token"), "token"),
        (WebSocketKeys(api_key="token", subprotocols="openai-insecure-api-key.other"), "token"),
        (WebSocketKeys(subprotocols="realtime, openai-insecure-api-key.token, openai-beta.realtime-v1"), "token"),
        (WebSocketKeys(api_key="", subprotocols="openai-insecure-api-key.token"), "token"),
        (WebSocketKeys(subprotocols="realtime"), WebSocketDenial.NO_API_KEY),
        (WebSocketKeys(subprotocols="openai-insecure-api-key."), WebSocketDenial.NO_API_KEY),
        (WebSocketKeys(), WebSocketDenial.NO_API_KEY),
    ],
)
def test_websocket_takes_a_bearer_authorization_then_api_key_then_subprotocol(
    keys: WebSocketKeys, expected: str | WebSocketDenial
):
    assert select_websocket_api_key(keys) == expected
