"""Contract: the ``/v1/realtime`` upgrade handshake, how the proxy authenticates it and how it picks the model.

Authentication (``user_api_key_auth_websocket_for_model``, litellm/proxy/auth/user_api_key_auth.py): the key is
read from ``Authorization: Bearer <key>`` first; only when that header is absent from ``api-key``; only when both
are absent from a ``Sec-WebSocket-Protocol`` entry ``openai-insecure-api-key.<key>`` (browser clients). A present
``Authorization`` header that does not start with exactly ``Bearer `` refuses the upgrade even when a valid
``api-key`` is also sent. There is no query string key. A refusal before accept is an HTTP 403 handshake answer.

Model selection (``realtime_websocket_endpoint`` and ``resolve_realtime_route_model``, proxy_server.py and
http_parsing_utils.py): ``?model=`` names the public model; without it ``intent=transcription`` resolves to
``gpt-realtime-whisper`` and the ``/realtime/translations`` routes resolve to ``gpt-realtime-translate``;
otherwise the upgrade is refused with close 1008 ``model query parameter is required`` before accept (403).
A model the key may not call is refused before accept. On accept the proxy echoes the FIRST subprotocol the
client offered, whatever it is.

A Rust front door that re-implements the upgrade must keep the precedence, the subprotocol echo and the
default models, or browser and SDK clients break silently.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration.providers.dec1_realtime_support import (
    REALTIME_MODEL,
    Await,
    AwaitClose,
    Send,
    done_event,
    openai_deployment,
    realtime,
    scripted,
    upgrade_query,
    upgrades,
)

pytestmark: Final = [pytest.mark.timeout(180), pytest.mark.shares_worker_state]

BROWSER_PROTOCOLS: Final = ("realtime", "openai-beta.realtime-v1")


def test_a_bearer_key_opens_a_session_and_the_upstream_sees_the_deployment_key_not_the_virtual_key(
    gateway: Gateway,
) -> None:
    """A deployment on the scripted upstream and a virtual key are created; the client upgrades with
    ``Authorization: Bearer <virtual key>``. The client receives ``session.created`` and the upstream handshake
    carries ``Authorization: Bearer <deployment api_key>``; the virtual key never leaves the proxy."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}", key=key)
        assert outcome.types[:1] == ("session.created",), outcome
        seen: Final = upgrades(gateway, handle.scenario_id)
        assert [entry["authorization"] for entry in seen] == [f"Bearer {handle.scenario_id}"], seen
        assert key not in str(seen), seen


def test_an_api_key_header_alone_opens_a_session(gateway: Gateway) -> None:
    """The client sends only ``api-key: <virtual key>`` (the Azure SDK shape) and no Authorization header.
    The upgrade is accepted and ``session.created`` arrives, read from code at user_api_key_auth.py
    ``user_api_key_auth_websocket_for_model`` (api-key is the second source)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}", headers={"api-key": key})
        assert outcome.handshake_status is None, outcome
        assert outcome.types[:1] == ("session.created",), outcome


def test_a_browser_subprotocol_key_opens_a_session_and_the_proxy_echoes_the_first_offered_protocol(
    gateway: Gateway,
) -> None:
    """A browser client cannot set headers, so it offers ``realtime``, ``openai-insecure-api-key.<key>`` and
    ``openai-beta.realtime-v1`` as subprotocols with no Authorization or api-key header. The proxy reads the key
    from the subprotocol entry, accepts, and answers ``Sec-WebSocket-Protocol: realtime`` (the first offered)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            subprotocols=("realtime", f"openai-insecure-api-key.{key}", "openai-beta.realtime-v1"),
        )
        assert outcome.subprotocol == "realtime", outcome
        assert outcome.types[:1] == ("session.created",), outcome


def test_a_key_offered_as_the_first_subprotocol_is_echoed_back_in_the_handshake_answer(gateway: Gateway) -> None:
    """A client offers ``openai-insecure-api-key.<key>`` as its FIRST subprotocol. The proxy accepts with that
    exact string as the selected subprotocol (``requested_protocols[0]``), so the virtual key appears in the 101
    response headers. Pinned as current behaviour so a rewrite that stops echoing secrets does it on purpose."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}", subprotocols=(f"openai-insecure-api-key.{key}",))
        assert outcome.subprotocol == f"openai-insecure-api-key.{key}", outcome


def test_a_client_that_offers_no_subprotocol_gets_none_selected(gateway: Gateway) -> None:
    """With a Bearer key and no ``Sec-WebSocket-Protocol`` header, the proxy accepts with no subprotocol (the
    handshake answer carries no ``Sec-WebSocket-Protocol``), so strict clients do not fail the upgrade."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}", key=key)
        assert outcome.subprotocol is None, outcome


def test_a_malformed_authorization_header_is_refused_even_with_a_valid_api_key_header(gateway: Gateway) -> None:
    """The client sends ``Authorization: Token <key>`` together with a valid ``api-key: <key>``. Because the
    Authorization header is present it is the only source consulted, it does not start with ``Bearer `` and the
    upgrade is refused with HTTP 403 (``Invalid Authorization header format``); the api-key is never tried."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}", headers={"Authorization": f"Token {key}", "api-key": key})
        assert outcome.handshake_status == 403, outcome
        assert upgrades(gateway, handle.scenario_id) == ()


def test_an_api_key_header_wins_over_a_subprotocol_key(gateway: Gateway) -> None:
    """The client sends ``api-key: <unknown key>`` and offers ``openai-insecure-api-key.<valid key>``. The
    api-key header is consulted first, the unknown key fails auth, and the upgrade is refused with 403; the
    subprotocol key is not a fallback for a bad header key."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            headers={"api-key": f"sk-{uuid.uuid4().hex}"},
            subprotocols=("realtime", f"openai-insecure-api-key.{key}"),
        )
        assert outcome.handshake_status == 403, outcome


@pytest.mark.parametrize("parameter", ("api_key", "api-key", "key"))
def test_a_key_in_the_query_string_does_not_authenticate_an_upgrade(gateway: Gateway, parameter: str) -> None:
    """A valid virtual key passed only as ``?<parameter>=<key>`` next to ``model`` (no header, no subprotocol) is
    refused with HTTP 403 ``No API key provided``: the websocket auth reads headers and subprotocols only, and
    the upstream sees no handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        outcome: Final = realtime(gateway, f"model={model}&{parameter}={key}")
        assert outcome.handshake_status == 403, outcome
        assert upgrades(gateway, handle.scenario_id) == ()


def test_the_master_key_opens_a_session_on_any_model(gateway: Gateway) -> None:
    """The proxy master key (the gateway's admin key) upgrades on a freshly added deployment with no key-level
    model list and receives ``session.created``."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key)
        assert outcome.types[:1] == ("session.created",), outcome


def test_a_key_without_access_to_the_model_is_refused_before_accept(gateway: Gateway) -> None:
    """A key restricted to another model upgrades with ``?model=<realtime deployment>``.
    ``can_key_call_resolved_model`` refuses, the socket is closed before accept, so the client sees HTTP 403 and
    the upstream sees no handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        other: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[other])
        outcome: Final = realtime(gateway, f"model={model}", key=key)
        assert outcome.handshake_status == 403, outcome
        assert upgrades(gateway, handle.scenario_id) == ()


def test_an_unknown_model_name_never_reaches_any_upstream(gateway: Gateway) -> None:
    """A master key upgrades with ``?model=<name no deployment has>``. Either the access check refuses before
    accept (403) or routing fails after accept with an ``error`` event of type ``server_error`` and close 1011.
    UNVERIFIED which of the two applies for the master key; both are accepted, the invariant is that the session
    never opens and no ``session.created`` is sent."""
    outcome: Final = realtime(gateway, f"model=dec1-rt-missing-{uuid.uuid4().hex}", key=gateway.key)
    assert "session.created" not in outcome.types, outcome
    assert outcome.handshake_status == 403 or outcome.close_code == 1011, outcome


def test_no_model_and_no_intent_is_refused_with_403(gateway: Gateway) -> None:
    """A valid key upgrades ``/v1/realtime`` with no query string at all. ``resolve_realtime_route_model``
    returns None, the session is rejected with close 1008 ``model query parameter is required`` before accept,
    which the client sees as HTTP 403."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = realtime(gateway, "", key=key)
        assert outcome.handshake_status == 403, outcome


def test_intent_transcription_without_a_model_routes_to_gpt_realtime_whisper(gateway: Gateway) -> None:
    """A deployment is registered under the public name ``gpt-realtime-whisper``. A client upgrades with only
    ``?intent=transcription``; the proxy resolves the model to ``gpt-realtime-whisper`` (http_parsing_utils.py
    ``resolve_realtime_route_model``), and that deployment's upstream receives the handshake with the
    ``intent=transcription`` pair and no ``model`` pair (the client gave none).
    UNVERIFIED: a second test proxy may already serve a deployment named ``gpt-realtime-whisper``."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),), session_type="transcription")
        scenario.model(
            name="gpt-realtime-whisper",
            model="openai/gpt-realtime-whisper",
            api_key=handle.scenario_id,
            api_base=gateway.upstream_url.rstrip("/"),
        )
        outcome: Final = realtime(gateway, "intent=transcription", key=gateway.key)
        assert outcome.types[:1] == ("session.created",), outcome
        seen: Final = upgrades(gateway, handle.scenario_id)
        assert [upgrade_query(entry) for entry in seen] == [[["intent", "transcription"]]], seen


@pytest.mark.parametrize("path", ("/v1/realtime", "/realtime", "/openai/v1/realtime"))
def test_every_realtime_route_alias_reaches_the_upstream_at_v1_realtime(gateway: Gateway, path: str) -> None:
    """The same deployment is opened through each websocket alias the proxy registers. Every alias delivers
    ``session.created`` and the upstream handshake path is always ``/v1/realtime``
    (``OpenAIRealtime._construct_url`` overwrites the path)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, path=path)
        assert outcome.types[:1] == ("session.created",), outcome
        assert [entry["path"] for entry in upgrades(gateway, handle.scenario_id)] == ["/v1/realtime"]


def test_a_public_alias_is_replaced_by_the_provider_model_in_the_upstream_query(gateway: Gateway) -> None:
    """The deployment's public name is a random alias and its ``litellm_params.model`` is
    ``openai/gpt-4o-realtime-preview``. The client upgrades with ``?model=<alias>``; the upstream handshake query
    is exactly ``[["model", "gpt-4o-realtime-preview"]]`` (``_arealtime`` rewrites ``query_params["model"]`` to
    the normalized provider model), so no proxy alias leaks to the provider."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(gateway, f"model={model}", key=gateway.key)
        assert [upgrade_query(entry) for entry in upgrades(gateway, handle.scenario_id)] == [
            [["model", REALTIME_MODEL]]
        ]


def test_unrecognized_client_query_parameters_are_not_forwarded_upstream(gateway: Gateway) -> None:
    """The client upgrades with ``?model=<alias>&foo=bar&temperature=0.2&guardrails=x``. Only ``model`` (and
    ``intent`` when present) are forwarded: the upstream query is ``[["model", "gpt-4o-realtime-preview"]]``
    (``_realtime_query_params_template`` keeps only the declared keys). ``guardrails`` is consumed by the proxy.
    UNVERIFIED: an unknown guardrail name may refuse the session at pre-call; then no upgrade is observed and
    the test only checks that nothing but model reached the upstream."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        realtime(gateway, f"model={model}&foo=bar&temperature=0.2", key=gateway.key)
        queries: Final = [upgrade_query(entry) for entry in upgrades(gateway, handle.scenario_id)]
        assert queries == [[["model", REALTIME_MODEL]]], queries


def test_model_given_twice_in_the_query_resolves_to_the_last_value(gateway: Gateway) -> None:
    """``?model=<first>&model=<second>`` with two deployments on two scenarios: FastAPI's ``Query`` reads the
    last value, so only the second deployment's upstream receives a handshake."""
    with gateway.scenario() as scenario:
        first_handle: Final = scripted(scenario, (done_event(),))
        second_handle: Final = scripted(scenario, (done_event(),))
        first: Final = openai_deployment(scenario, first_handle)
        second: Final = openai_deployment(scenario, second_handle)
        outcome: Final = realtime(gateway, f"model={first}&model={second}", key=gateway.key)
        assert outcome.types[:1] == ("session.created",), outcome
        assert upgrades(gateway, first_handle.scenario_id) == ()
        assert len(upgrades(gateway, second_handle.scenario_id)) == 1


def test_a_team_model_alias_routes_the_session_to_the_aliased_deployment(gateway: Gateway) -> None:
    """A team has ``model_aliases: {"voice": <deployment name>}`` and a key of that team upgrades with
    ``?model=voice``. Since #52124 (``_route_realtime_websocket_session`` no longer overrides ``model``) the
    alias resolves in pre-call and the aliased deployment's upstream receives the handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        alias: Final = f"voice-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(models=[model, alias], model_aliases={alias: model})
        key: Final = scenario.key(team_id=team, models=[alias, model])
        outcome: Final = realtime(gateway, f"model={alias}", key=key)
        assert outcome.types[:1] == ("session.created",), outcome
        assert len(upgrades(gateway, handle.scenario_id)) == 1


def test_a_key_model_alias_routes_the_session_to_the_aliased_deployment(gateway: Gateway) -> None:
    """A key is created with ``aliases: {<alias>: <deployment name>}`` and upgrades with ``?model=<alias>``.
    The key alias is resolved like on HTTP routes and the deployment's upstream receives exactly one handshake.
    UNVERIFIED: read from the shared alias path in ``common_processing_pre_call_logic``; the access check runs
    on the alias name first."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        alias: Final = f"keyvoice-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(aliases={alias: model})
        outcome: Final = realtime(gateway, f"model={alias}", key=key)
        assert outcome.types[:1] == ("session.created",), outcome
        assert len(upgrades(gateway, handle.scenario_id)) == 1


def test_two_deployments_in_one_group_each_take_sessions(gateway: Gateway) -> None:
    """Two deployments share one public name, each on its own scripted scenario. Ten sessions opened on that
    name are spread by the router: each scenario's upstream sees at least one handshake and together they see
    exactly ten. UNVERIFIED: simple-shuffle makes a 10-0 split possible with probability 2/1024."""
    with gateway.scenario() as scenario:
        first: Final = scripted(scenario, (done_event(),) * 10)
        second: Final = scripted(scenario, (done_event(),) * 10)
        name: Final = f"dec1-rt-group-{uuid.uuid4().hex[:8]}"
        for handle in (first, second):
            scenario.model(
                name=name,
                model=f"openai/{REALTIME_MODEL}",
                api_key=handle.scenario_id,
                api_base=gateway.upstream_url.rstrip("/"),
            )
        for _ in range(10):
            realtime(gateway, f"model={name}", key=gateway.key)
        counts: Final = (len(upgrades(gateway, first.scenario_id)), len(upgrades(gateway, second.scenario_id)))
        assert sum(counts) == 10, counts
        assert min(counts) >= 1, counts


def test_a_session_on_a_deleted_deployment_is_refused(gateway: Gateway) -> None:
    """A deployment serves one session, is deleted through ``/model/delete``, and the same key upgrades again.
    The second upgrade never yields ``session.created`` (refused 403 at the access check or closed 1011 after a
    routing failure) and the upstream sees exactly the one earlier handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        assert realtime(gateway, f"model={model}", key=gateway.key).types[:1] == ("session.created",)
        identity: Final = str(
            gateway.rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (model,))[0][
                "model_id"
            ]
        )
        scenario.delete_model(identity)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key)
        assert "session.created" not in outcome.types, outcome
        assert len(upgrades(gateway, handle.scenario_id)) == 1


def test_live_sessions_reject_a_first_frame_that_is_not_json(gateway: Gateway) -> None:
    """``/v1/live/sessions`` accepts the upgrade, then reads the first frame. A frame ``not json`` gets one
    ``error`` event ``{"type": "invalid_request_error", "code": "invalid_session_start", "message": "First message
    must be a session.start JSON object with a non-empty session.model."}`` and close 1008
    ``Invalid session.start frame`` (proxy_server.py ``_reject_invalid_live_session_start``)."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = realtime(
            gateway, "", key=key, path="/v1/live/sessions", steps=(Send("not json"), Await("error"))
        )
        assert outcome.error_types() == ("invalid_request_error",), outcome
        assert outcome.error_messages() == (
            "First message must be a session.start JSON object with a non-empty session.model.",
        ), outcome


@pytest.mark.parametrize(
    "frame",
    (
        {"type": "session.update", "session": {"model": "gpt-realtime"}},
        {"type": "session.start", "session": {}},
        {"type": "session.start", "session": {"model": "   "}},
        {"type": "session.start", "session": "gpt-realtime"},
    ),
    ids=("wrong_type", "missing_model", "blank_model", "session_not_object"),
)
def test_live_sessions_reject_an_invalid_session_start_with_code_invalid_session_start(
    gateway: Gateway, frame: dict[str, object]
) -> None:
    """Each malformed first frame on ``/v1/live/sessions`` (wrong type, no ``session.model``, a blank model, a
    non-object session) is answered with one error event whose ``error.code`` is ``invalid_session_start`` and
    the socket closes 1008 with reason ``Invalid session.start frame`` (``parse_live_session_start`` returns
    None for all four)."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        outcome: Final = realtime(
            gateway, "", key=key, path="/v1/live/sessions", steps=(Send(frame), Await("error"), AwaitClose())
        )
        errors: Final = outcome.of_type("error")
        assert [object_value(error["error"]).get("code") for error in errors] == ["invalid_session_start"], outcome
        assert (outcome.close_code, outcome.close_reason) == (1008, "Invalid session.start frame"), outcome


def test_live_sessions_refuse_a_model_the_key_cannot_call_with_code_model_not_allowed(gateway: Gateway) -> None:
    """A key restricted to one model sends ``{"type": "session.start", "session": {"model": <other>}}`` as its
    first frame on ``/v1/live/sessions``. The re-authorization for the named model fails, the client gets an
    ``invalid_request_error`` event with ``code`` ``model_not_allowed`` and close 1008
    (``_live_session_denial_code``). UNVERIFIED: the access error type ends with ``_model_access_denied``."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        allowed: Final = openai_deployment(scenario, handle)
        other: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(models=[allowed])
        outcome: Final = realtime(
            gateway,
            "",
            key=key,
            path="/v1/live/sessions",
            steps=(Send({"type": "session.start", "session": {"model": other}}), Await("error"), AwaitClose()),
        )
        assert [object_value(error["error"]).get("code") for error in outcome.of_type("error")] == [
            "model_not_allowed"
        ], outcome
        assert outcome.close_code == 1008, outcome
        assert upgrades(gateway, handle.scenario_id) == ()


def test_live_sessions_refuse_a_non_openai_provider_after_session_start(gateway: Gateway) -> None:
    """A Gemini deployment is named in a valid ``session.start`` on ``/v1/live/sessions``. ``_alive_session``
    refuses non-OpenAI providers (``OpenAI Live sessions require the openai provider, got gemini``) and the
    session never opens: no ``session.started`` reaches the client and the socket is closed.
    UNVERIFIED: the exact close code (1011 from the route_request error path)."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/gemini-2.5-flash-native-audio-latest", api_key="dec1-rt-gemini")
        outcome: Final = realtime(
            gateway,
            "",
            key=gateway.key,
            path="/v1/live/sessions",
            steps=(Send({"type": "session.start", "session": {"model": model}}), AwaitClose()),
        )
        assert "session.started" not in outcome.types, outcome
        assert outcome.close_code is not None and outcome.close_code != 1000, outcome
