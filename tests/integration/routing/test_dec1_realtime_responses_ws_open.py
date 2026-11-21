"""Contract: opening a Responses API websocket (``/v1/responses`` in websocket mode) before any turn is routed.

``responses_websocket_endpoint`` (litellm/proxy/response_api_endpoints/endpoints.py) authenticates the upgrade
with the same header precedence as ``/v1/realtime`` (Authorization Bearer, then ``api-key``, then an
``openai-insecure-api-key.<key>`` subprotocol), accepts with the first offered subprotocol, and then reads the
first frame (``_read_ws_model_from_first_frame``):

- not JSON: error ``First message is not valid JSON.`` and close 1008 ``Invalid JSON in first message``;
- JSON that is not an object with ``type`` ``response.create``: error ``First message must be a response.create
  JSON object.`` and close 1008 ``Invalid first message``;
- no ``?model=`` and no ``model`` (flat or under ``response``): error ``No model provided. Supply ?model=<model>
  in the URL or include 'model' in the first response.create event.`` and close 1008 ``No model provided``;
- a read failure other than a disconnect (for example a binary frame): close 1011 ``Internal server error``.

Every error frame is ``{"type": "error", "error": {"type": "invalid_request_error", "message": ...}}`` without a
``status``. A rate limit refusal at open is ``turn_refusal_frame`` plus close 1008 ``Rate limit exceeded``; any
other pre-call failure is an ``invalid_request_error`` frame plus close 1008 ``Pre-call error``. These checks
need no provider, so they run against the shared proxy. Turn admission, session limits and turn billing are
pinned by ``test_responses_websocket_turn_admission.py``, ``providers/test_responses_websocket_session_limit.py``
and ``spend/test_responses_websocket_turn_rows.py``.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration.providers.dec1_realtime_support import AwaitClose, Outcome, Send, realtime

pytestmark: Final = [pytest.mark.timeout(120), pytest.mark.shares_worker_state]

NOT_JSON: Final = "First message is not valid JSON."
NOT_CREATE: Final = "First message must be a response.create JSON object."
NO_MODEL: Final = (
    "No model provided. Supply ?model=<model> in the URL or include 'model' in the first response.create event."
)


def _open(
    gateway: Gateway,
    first: dict[str, object] | str | bytes,
    *,
    query: str = "",
    key: str | None = None,
    headers: dict[str, str] | None = None,
    subprotocols: tuple[str, ...] | None = None,
    path: str = "/v1/responses",
) -> Outcome:
    return realtime(
        gateway,
        query,
        key=key,
        headers=headers,
        subprotocols=subprotocols,
        path=path,
        steps=(Send(first), AwaitClose()),
    )


def _error_bodies(outcome: Outcome) -> list[dict[str, object]]:
    return [object_value(frame["error"]) for frame in outcome.of_type("error")]


@pytest.mark.parametrize("path", ("/v1/responses", "/responses"))
def test_a_first_frame_that_is_not_json_is_refused_with_1008(gateway: Gateway, path: str) -> None:
    """A valid key upgrades ``<path>`` and sends the text ``{not json``. It receives exactly one error frame
    ``{"type": "invalid_request_error", "message": "First message is not valid JSON."}`` and the close 1008
    ``Invalid JSON in first message``, on both route aliases."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, "{not json", key=key, path=path)
        assert _error_bodies(outcome) == [{"type": "invalid_request_error", "message": NOT_JSON}], outcome
        assert (outcome.close_code, outcome.close_reason) == (1008, "Invalid JSON in first message"), outcome


@pytest.mark.parametrize(
    "first",
    (
        {"type": "response.cancel"},
        {"type": "session.update", "session": {}},
        {"model": "gpt-4o"},
        "[1, 2, 3]",
        '"response.create"',
    ),
    ids=("cancel", "session_update", "no_type", "array", "string"),
)
def test_a_first_frame_that_is_not_a_response_create_object_is_refused(gateway: Gateway, first: object) -> None:
    """A first frame that is valid JSON but not an object whose ``type`` is ``response.create`` (another event
    type, a missing type, a JSON array, a JSON string) gets the error ``First message must be a response.create
    JSON object.`` and close 1008 ``Invalid first message``."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, first, key=key)  # type: ignore[arg-type]
        assert _error_bodies(outcome) == [{"type": "invalid_request_error", "message": NOT_CREATE}], outcome
        assert (outcome.close_code, outcome.close_reason) == (1008, "Invalid first message"), outcome


def test_a_response_create_without_any_model_is_refused_with_no_model_provided(gateway: Gateway) -> None:
    """No ``?model=`` and a ``response.create`` with neither ``model`` nor ``response.model``: error frame with the
    exact ``No model provided...`` message and close 1008 ``No model provided``."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, {"type": "response.create", "input": "hi"}, key=key)
        assert _error_bodies(outcome) == [{"type": "invalid_request_error", "message": NO_MODEL}], outcome
        assert (outcome.close_code, outcome.close_reason) == (1008, "No model provided"), outcome


def test_an_empty_model_in_the_frame_counts_as_no_model(gateway: Gateway) -> None:
    """``{"type": "response.create", "model": "", "response": {"model": ""}}`` resolves to no model (both values
    are falsy) and gets the ``No model provided`` refusal, not a routing attempt on an empty name."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, {"type": "response.create", "model": "", "response": {"model": ""}}, key=key)
        assert (outcome.close_code, outcome.close_reason) == (1008, "No model provided"), outcome


def test_a_binary_first_frame_closes_with_1011_internal_server_error(gateway: Gateway) -> None:
    """The first frame is binary. ``receive_text`` fails with something other than a disconnect, so the proxy
    closes 1011 ``Internal server error`` without an error frame."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, b'{"type": "response.create"}', key=key)
        assert outcome.of_type("error") == (), outcome
        assert (outcome.close_code, outcome.close_reason) == (1011, "Internal server error"), outcome


def test_an_api_key_header_authenticates_the_responses_socket(gateway: Gateway) -> None:
    """A key sent only as ``api-key`` passes the upgrade: the socket is accepted and the first-frame validation
    answers (``Invalid first message`` for a ``ping`` frame) instead of a 403 handshake."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, {"type": "ping"}, headers={"api-key": key})
        assert outcome.handshake_status is None, outcome
        assert outcome.close_reason == "Invalid first message", outcome


def test_a_subprotocol_key_authenticates_the_responses_socket_and_the_first_protocol_is_echoed(
    gateway: Gateway,
) -> None:
    """A browser-style client offers ``realtime`` and ``openai-insecure-api-key.<key>`` with no headers. The
    socket is accepted with subprotocol ``realtime`` and the first frame is validated."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, {"type": "ping"}, subprotocols=("realtime", f"openai-insecure-api-key.{key}"))
        assert outcome.subprotocol == "realtime", outcome
        assert outcome.close_reason == "Invalid first message", outcome


def test_a_lowercase_bearer_scheme_is_refused_on_the_responses_socket(gateway: Gateway) -> None:
    """``Authorization: bearer <valid key>`` (lowercase scheme) is refused before accept with HTTP 403: the
    websocket auth requires the exact prefix ``Bearer `` even though HTTP routes accept ``bearer``
    (``get_bearer_token``). A unified Rust auth would change this on purpose only."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = _open(gateway, {"type": "ping"}, headers={"Authorization": f"bearer {key}"})
        assert outcome.handshake_status == 403, outcome


def test_a_query_model_the_key_cannot_call_is_refused_before_accept(gateway: Gateway) -> None:
    """A key restricted to one chat model upgrades ``/v1/responses?model=<other model>``. The model is known at
    the upgrade, auth refuses it, and the client sees HTTP 403 without any frame."""
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        outcome: Final = _open(gateway, {"type": "response.create"}, query=f"model={other}", key=key)
        assert outcome.handshake_status == 403, outcome


def test_a_frame_model_the_key_cannot_call_is_refused_after_accept_with_pre_call_error(gateway: Gateway) -> None:
    """Without ``?model=`` the key is checked against the model of the first frame
    (``_enforce_responses_ws_first_frame_model_auth``). A key restricted to one model naming another in
    ``response.model`` gets one ``invalid_request_error`` frame and close 1008 ``Pre-call error``."""
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        outcome: Final = _open(gateway, {"type": "response.create", "response": {"model": other}}, key=key)
        assert [body.get("type") for body in _error_bodies(outcome)] == ["invalid_request_error"], outcome
        assert (outcome.close_code, outcome.close_reason) == (1008, "Pre-call error"), outcome


def test_the_query_model_wins_over_a_conflicting_frame_model(gateway: Gateway) -> None:
    """A key allowed only model A upgrades ``?model=A`` and sends a first frame naming model B. The query model
    is used (``query_model or _extract_model_from_first_ws_event``), so the frame's B is never access checked:
    the socket is NOT closed with 1008 ``Pre-call error``. UNVERIFIED: model A is a chat deployment on the plain
    HTTP scripted upstream, so the session later fails at the provider (close 1011) or is bridged; only the
    absence of the access refusal is pinned."""
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        outcome: Final = _open(
            gateway, {"type": "response.create", "model": other, "input": "hi"}, query=f"model={allowed}", key=key
        )
        assert (outcome.close_code, outcome.close_reason) != (1008, "Pre-call error"), outcome


def test_a_key_over_its_rpm_is_refused_when_the_socket_opens(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` spends its request on a chat call, then opens ``/v1/responses?model=<m>``.
    It is refused: either at the upgrade (HTTP 403 from the pre-body check on a cached key, #51428) or after the
    first frame with a ``turn_refusal_frame`` (``status`` 429, ``error.type`` ``rate_limit_exceeded``) and close
    1008 ``Rate limit exceeded``. UNVERIFIED which of the two the websocket path takes."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(rpm_limit=1)
        gateway.chat(model, key=key)
        outcome: Final = _open(gateway, {"type": "response.create", "input": "hi"}, query=f"model={model}", key=key)
        if outcome.handshake_status is None:
            refusals: Final = outcome.of_type("error")
            assert [frame.get("status") for frame in refusals] == [429], outcome
            assert object_value(refusals[0]["error"]).get("type") == "rate_limit_exceeded", outcome
            assert (outcome.close_code, outcome.close_reason) == (1008, "Rate limit exceeded"), outcome
        else:
            assert outcome.handshake_status in (403, 429), outcome


def test_an_unknown_frame_model_is_refused_without_reaching_any_provider(gateway: Gateway) -> None:
    """The master key sends a first frame naming ``dec1-missing-<uuid>``. The session never routes: the client
    gets one error frame and the socket closes with 1008 (pre-call) or 1011 (router failure, with a ``status``
    400 or 500 frame). UNVERIFIED which code; both mean no provider was dialled."""
    outcome: Final = _open(
        gateway, {"type": "response.create", "model": f"dec1-missing-{uuid.uuid4().hex}"}, key=gateway.key
    )
    assert len(outcome.of_type("error")) == 1, outcome
    assert outcome.close_code in (1008, 1011), outcome
