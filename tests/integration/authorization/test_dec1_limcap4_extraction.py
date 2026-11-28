"""Extraction requests in the key token window: input only, no output reservation, no implicit cap.

``POST /v1/extraction`` runs with call type ``aextraction`` (``litellm/proxy/decisions_endpoints/endpoints.py:295``)
through the same pre-call hooks as chat. The v3 limiter (``litellm/proxy/hooks/parallel_request_limiter_v3.py``)
estimates its input from ``state + "\\n" + json.dumps(fields)`` (``_extraction_input_text``, :3861) at one token per
four characters, reserves no output tokens for it (``generates_no_output``, :1437), and never writes an implicit
``max_tokens`` into it (``EXTRACTION_API_CALL_TYPES`` in the cap exemption, :1211). A chat request of the same size
would also reserve an output floor of a quarter of the token limit.

Every request here uses a 76 character ``state`` and a ``fields`` map that serializes to 60 characters, so the text
the limiter measures is 137 characters, an estimate of 34 tokens. The provider is a Cloudflare Clairic deployment on a
wire that answers ``usage.input_tokens: 412`` (the shape of ``providers/test_cloudflare_clairic_extraction_wire.py``).
"""

from __future__ import annotations

import json
import uuid
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    posts,
    scope_headers,
)

CLAIRIC: Final = "cloudflare/@cf/cloudflare/clairic"
PROVIDER_KEY: Final = "synthetic-cloudflare-key"
FIELDS: Final[dict[str, JsonValue]] = {"vendor": {"type": "text", "instructions": "Who sent it?"}}
ESTIMATE: Final = 34
USAGE_INPUT: Final = 412


def _state() -> str:
    text: Final = f"Invoice from Acme Tools, total $174. marker-{uuid.uuid4().hex}"
    assert len(text) == 76 and len(json.dumps(FIELDS)) == 60
    return text


def _clairic(request: Request) -> Reply:
    state: Final = JSON_OBJECT.validate_json(request.body).get("state")
    start: Final = str(state).find("Acme Tools")
    return Reply(
        body=json.dumps(
            {
                "result": {
                    "model": "clairic",
                    "answers": {
                        "vendor": {
                            "type": "text",
                            "value": "Acme Tools",
                            "confidence": 0.9,
                            "span": [start, start + len("Acme Tools")],
                        }
                    },
                    "truncated": False,
                    "usage": {"input_tokens": USAGE_INPUT, "output_tokens": 0},
                },
                "success": True,
                "errors": [],
                "messages": [],
            }
        ).encode()
    )


def _extract(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request("POST", "/v1/extraction", {"model": model, "state": _state(), "fields": FIELDS}, key=key)


def test_an_extraction_reserves_only_its_input_estimate(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 1000`` sends one extraction. The response reports
    ``x-ratelimit-api_key-remaining-tokens: 966``, which is 1000 minus the 34 token input estimate and no output.
    UNVERIFIED: that the extraction route attaches the limiter headers like chat does
    (``async_post_call_success_hook``)."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        answered: Final = _extract(gateway, model, key)
        assert answered.status_code == 200, answered.text
        assert scope_headers(answered, "api_key")["remaining-tokens"] == str(1000 - ESTIMATE), dict(answered.headers)


def test_an_extraction_larger_than_the_key_token_limit_is_refused_before_the_provider(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 33`` sends an extraction estimated at 34 tokens. It is refused 429 with ``Rate limit
    exceeded for api_key: <hash>. Limit type: tokens. Current limit: 33, Remaining: 33`` and the provider receives
    nothing."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        key: Final = scenario.key(models=[model], tpm_limit=33)
        wire.drain()
        assert_limited(_extract(gateway, model, key), "api_key", hashed(key), "tokens", 33, 33)
        assert posts(wire) == ()


def test_an_extraction_that_fits_its_input_is_served_where_a_chat_of_the_same_size_is_not(gateway: Gateway) -> None:
    """Two keys with ``tpm_limit: 34``. An extraction measured at 137 characters (34 tokens) is served. A chat whose
    message is the same 137 characters is refused for ``api_key`` tokens: chat adds an output floor of
    ``int(34 x 0.25) = 8`` tokens (``no_max_tokens_output_floor``, :1060) while extraction reserves none."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        chat_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        extractor: Final = scenario.key(models=[model], tpm_limit=ESTIMATE)
        chatter: Final = scenario.key(models=[chat_model], tpm_limit=ESTIMATE)
        assert _extract(gateway, model, extractor).status_code == 200
        text: Final = _state() + "\n" + json.dumps(FIELDS)
        assert_limited(chat(gateway, chat_model, chatter, text=text), "api_key", hashed(chatter), "tokens", 34, 34)


def test_an_extraction_under_a_small_token_limit_gets_no_implicit_max_tokens(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 1000`` would give a chat request an implicit ``max_tokens: 250``. Its extraction
    reaches the provider without any ``max_tokens`` or ``max_output_tokens`` field: the cap exemption covers
    ``EXTRACTION_API_CALL_TYPES``."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        wire.drain()
        assert _extract(gateway, model, key).status_code == 200
        sent: Final = posts(wire)
        assert len(sent) == 1
        assert {"max_tokens", "max_output_tokens"} & set(sent[0]) == set(), sent[0]


def test_extractions_count_against_the_key_request_window(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` sends two extractions. The second is refused 429 for ``api_key: <hash>`` with
    ``Limit type: requests. Current limit: 1``."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert _extract(gateway, model, key).status_code == 200
        assert_limited(_extract(gateway, model, key), "api_key", hashed(key), "requests", 1)


def test_an_extraction_does_not_take_the_chat_output_floor_from_a_project_output_window(gateway: Gateway) -> None:
    """The key's project has ``model_otpm_limit: {<model>: 4}``. A chat request without ``max_tokens`` would reserve
    an output floor of 1 against it; an extraction reserves 0 output tokens and is served twice in the same minute.
    UNVERIFIED: that the project OTPM path (``reserve_io_tokens``) uses the same ``generates_no_output`` rule as the
    combined estimate."""
    with wire_server(_clairic) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=CLAIRIC, api_base=wire.url, api_key=PROVIDER_KEY)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 4})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert [_extract(gateway, model, key).status_code for _ in range(2)] == [200, 200]
