"""Which callbacks receive a request: the three config lists, per-key and per-team callbacks, and the opt-outs.

Three rigs. The "lists" rig sets `callbacks`, `success_callback` and `failure_callback` to different custom
logger instances so each list's meaning is observable. The "scoped" rig configures no global `generic_api`
callback, only its endpoint in the environment, so a payload at that endpoint can only come from a key's or team's
own `metadata.logging`. The last tests boot their own proxy for `/config/update` and for settings that change the
opt-outs.

The contract: `callbacks` receive success and failure events, `success_callback` only successes,
`failure_callback` only failures. A key's `metadata.logging` adds callbacks for that key's requests only and wins
over its team's. `no-log: true` in the request body skips logging callbacks but not spend tracking. Callers cannot
register or disable callbacks through the request body. `x-litellm-disable-callbacks` and a key's
`litellm_disabled_callbacks` are honoured only on a licensed proxy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, JsonValue, eventually, held, list_value, object_value
from integration._support.database import scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, wire_server
from integration.observability.dec1_callbacks_support import (
    CHAT,
    CHAT_REJECTED,
    PROBE_MODULE,
    RECORDER,
    SINK_SECRET,
    CallbackRig,
    Journal,
    ScriptedProvider,
    callback_rig,
    chat,
    marker,
    new_call_id,
)

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))
NEEDS_LICENSE: Final = "dynamic callback disabling is honoured only with LITELLM_LICENSE set"
LOG_EVENTS: Final = ("async_log_success_event", "async_log_failure_event")


def _generic(callback_type: str) -> list[JsonValue]:
    return [{"callback_name": "generic_api", "callback_type": callback_type, "callback_vars": {}}]


@pytest.fixture(scope="module")
def lists(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-lists"),
        {
            "callbacks": ["generic_api", RECORDER],
            "success_callback": [f"{PROBE_MODULE}.success_only"],
            "failure_callback": [f"{PROBE_MODULE}.failure_only"],
        },
    ) as created:
        yield created


@pytest.fixture(scope="module")
def scoped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(tmp_path_factory.mktemp("dec1-callbacks-scoped"), {"callbacks": [RECORDER]}) as created:
        yield created


def _log_events(rig: CallbackRig, call_id: str, logger: str) -> tuple[str, ...]:
    return tuple(name for name in rig.hook_names(call_id, logger) if name in LOG_EVENTS)


def _settled_log_events(rig: CallbackRig, call_id: str, logger: str, expected: tuple[str, ...]) -> None:
    held(lambda: _log_events(rig, call_id, logger), lambda found: found == expected, holding=3, seconds=30)


def test_callbacks_list_instance_gets_success_and_failure_events(lists: CallbackRig) -> None:
    """A logger listed under `callbacks` must get one async_log_success_event for a request that succeeds and one
    async_log_failure_event for a request the provider rejects."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(lists, ok).status_code == 200
    assert chat(lists, failed, model=CHAT_REJECTED).status_code == 400
    _settled_log_events(lists, ok, "recorder", ("async_log_success_event",))
    _settled_log_events(lists, failed, "recorder", ("async_log_failure_event",))


def test_success_callback_instance_gets_successes_and_never_failures(lists: CallbackRig) -> None:
    """A logger listed only under `success_callback` must get one async_log_success_event for a successful request
    and nothing at all for a failed one."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(lists, ok).status_code == 200
    assert chat(lists, failed, model=CHAT_REJECTED).status_code == 400
    lists.event(failed, "failure")
    _settled_log_events(lists, ok, "success_only", ("async_log_success_event",))
    _settled_log_events(lists, failed, "success_only", ())


def test_failure_callback_instance_gets_failures_and_never_successes(lists: CallbackRig) -> None:
    """A logger listed only under `failure_callback` must get one async_log_failure_event for a failed request and
    nothing at all for a successful one.

    BUG: a CustomLogger instance from `failure_callback` lands in `litellm.failure_callback`
    (`LoggingCallbackManager.add_litellm_failure_callback` only routes async *callables* to
    `_async_failure_callback`). `Logging.async_failure_handler` reads only `_async_failure_callback`, and the sync
    `failure_handler` calls `log_failure_event` only for sync requests (litellm_logging.py, `is_sync_request`), so
    on the proxy the instance gets no failure event at all. The success list does not have this gap:
    `_get_completed_success_callbacks` merges both lists and calls `async_log_success_event`."""
    pytest.skip("BUG: a CustomLogger listed under failure_callback never receives proxy failure events")
    ok, failed = new_call_id(), new_call_id()
    assert chat(lists, ok).status_code == 200
    assert chat(lists, failed, model=CHAT_REJECTED).status_code == 400
    lists.event(ok, "success")
    _settled_log_events(lists, failed, "failure_only", ("async_log_failure_event",))
    _settled_log_events(lists, ok, "failure_only", ())


def test_active_callbacks_endpoint_lists_every_configured_callback(lists: CallbackRig) -> None:
    """GET /active/callbacks on the lists rig must answer 200 and name the generic API logger and the custom
    recorder class among `all_litellm_callbacks`, and report a `num_callbacks` equal to the length of that list."""
    body: Final = lists.proxy.get("/active/callbacks")
    active: Final = [str(entry) for entry in list_value(body["all_litellm_callbacks"])]
    assert any("GenericAPILogger" in entry or "generic_api" in entry for entry in active), active
    assert any("Recorder" in entry for entry in active), active
    assert body["num_callbacks"] == len(active)


def test_no_log_request_skips_logging_callbacks(lists: CallbackRig) -> None:
    """The caller sends `"no-log": true` in the chat body. The caller gets a normal 200, but neither the
    generic_api endpoint nor the custom logger's async_log_success_event may receive anything for that call, even
    several seconds later."""
    call_id: Final = new_call_id()
    response: Final = chat(lists, call_id, **{"no-log": True})
    assert response.status_code == 200, response.text
    held(lambda: lists.events(call_id), lambda events: events == (), holding=4, seconds=10)
    assert _log_events(lists, call_id, "recorder") == ()


def test_no_log_request_is_still_written_to_spend_logs(lists: CallbackRig) -> None:
    """`"no-log": true` hides the request from logging callbacks but not from billing: a LiteLLM_SpendLogs row for
    the response id must still be written with the request's token counts.

    BUG: `Logging.should_run_callback` (litellm_logging.py) lets a no-log request reach only CustomLoggers whose
    class name contains `_PROXY_`, under the comment "proxy cost tracking callbacks should run". The spend
    tracker is `_ProxyDBLogger` (litellm/proxy/hooks/proxy_track_cost_callback.py), which does not match the
    case-sensitive check, so no-log requests are never billed or written to LiteLLM_SpendLogs."""
    pytest.skip("BUG: no-log requests skip _ProxyDBLogger, so no spend log row is written")
    response: Final = chat(lists, new_call_id(), **{"no-log": True})
    assert response.status_code == 200, response.text
    rows: Final = eventually(
        lambda: lists.proxy.rows(
            'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (str(response.json()["id"]),),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (11, 4)


def test_no_log_request_still_runs_the_proxy_hooks(lists: CallbackRig) -> None:
    """`"no-log": true` only silences log events. The custom callback's async_pre_call_hook and
    async_post_call_success_hook must still run for the request, because they enforce policy rather than log."""
    call_id: Final = new_call_id()
    assert chat(lists, call_id, **{"no-log": True}).status_code == 200
    names: Final = eventually(
        lambda: lists.hook_names(call_id), lambda found: "async_post_call_success_hook" in found, seconds=20
    )
    assert "async_pre_call_hook" in names


def test_request_body_cannot_disable_callbacks(lists: CallbackRig) -> None:
    """The caller sends `litellm_disabled_callbacks: ["generic_api"]` in the request body. The proxy strips that
    field from client bodies, so the generic_api payload for the call must still be delivered."""
    call_id: Final = new_call_id()
    assert chat(lists, call_id, litellm_disabled_callbacks=["generic_api"]).status_code == 200
    assert lists.event(call_id)["status"] == "success"


def test_request_body_cannot_register_callbacks(lists: CallbackRig) -> None:
    """The caller sends `callbacks` and `service_callback` lists in the request body. The request must answer 200
    and the proxy's active callback list must be exactly what it was before: a request can never add a
    process-wide callback."""
    before: Final = lists.proxy.get("/active/callbacks")["all_litellm_callbacks"]
    response: Final = chat(lists, new_call_id(), callbacks=["langfuse"], service_callback=["datadog"])
    assert response.status_code == 200, response.text
    assert lists.proxy.get("/active/callbacks")["all_litellm_callbacks"] == before


def test_disable_callbacks_header_is_ignored_without_a_license(lists: CallbackRig) -> None:
    """On a proxy without an enterprise license, a caller sending `x-litellm-disable-callbacks: generic_api` must
    not be able to turn the callback off: the payload for the call is still delivered."""
    if LICENSED:
        pytest.skip("this proxy is licensed, so the header is honoured; see the licensed test")
    call_id: Final = new_call_id()
    assert chat(lists, call_id, headers={"x-litellm-disable-callbacks": "generic_api"}).status_code == 200
    assert lists.event(call_id)["status"] == "success"


def test_disable_callbacks_header_skips_the_named_callback_with_a_license(lists: CallbackRig) -> None:
    """On a licensed proxy, `x-litellm-disable-callbacks: generic_api` on one request must stop the generic_api
    payload for that request only: nothing arrives for it, the custom recorder (not named in the header) still
    gets its success event, and the next request without the header is delivered as usual."""
    if not LICENSED:
        pytest.skip(NEEDS_LICENSE)
    disabled, normal = new_call_id(), new_call_id()
    assert chat(lists, disabled, headers={"x-litellm-disable-callbacks": "generic_api"}).status_code == 200
    assert chat(lists, normal).status_code == 200
    lists.event(normal)
    lists.hook(disabled, "async_log_success_event")
    held(lambda: lists.events(disabled), lambda events: events == (), holding=4, seconds=10)


def test_disable_callbacks_header_matches_names_case_insensitively(lists: CallbackRig) -> None:
    """On a licensed proxy the header value is a comma separated list compared in lower case with spaces trimmed:
    `x-litellm-disable-callbacks: Langfuse , GENERIC_API` must disable generic_api for that request."""
    if not LICENSED:
        pytest.skip(NEEDS_LICENSE)
    call_id: Final = new_call_id()
    assert chat(lists, call_id, headers={"x-litellm-disable-callbacks": "Langfuse , GENERIC_API"}).status_code == 200
    lists.hook(call_id, "async_log_success_event")
    held(lambda: lists.events(call_id), lambda events: events == (), holding=4, seconds=10)


def test_key_level_disabled_callbacks_skip_the_callback_for_that_key_only(lists: CallbackRig) -> None:
    """On a licensed proxy, a key whose metadata has `litellm_disabled_callbacks: ["generic_api"]` must produce no
    generic_api payload, while another key calling the same model still does.

    BUG: `key_and_team_logging_fields` (litellm/proxy/litellm_pre_call_utils.py) copies the key's list onto the
    request, but `initialize_standard_callback_dynamic_params` only reads names in `_supported_callback_params`
    (litellm/litellm_core_utils/initialize_dynamic_callback_params.py), which does not include
    `litellm_disabled_callbacks`. `EnterpriseCallbackControls.get_disabled_callbacks` reads it from those params,
    so it never sees the key's list and the callback still runs. The header form works."""
    pytest.skip("BUG: a key's litellm_disabled_callbacks metadata does not disable the named callback")
    if not LICENSED:
        pytest.skip(NEEDS_LICENSE)
    quiet_call, normal_call = new_call_id(), new_call_id()
    with lists.proxy.scenario() as scenario:
        quiet: Final = scenario.key(metadata={"litellm_disabled_callbacks": ["generic_api"]})
        normal: Final = scenario.key()
        assert chat(lists, quiet_call, key=quiet).status_code == 200
        assert chat(lists, normal_call, key=normal).status_code == 200
        lists.event(normal_call)
        held(lambda: lists.events(quiet_call), lambda events: events == (), holding=4, seconds=10)


def test_key_logging_metadata_adds_a_callback_for_that_key_only(scoped: CallbackRig) -> None:
    """No global generic_api callback is configured. A key created with `metadata.logging: [{callback_name:
    "generic_api", callback_type: "success", callback_vars: {}}]` makes a successful call and a payload for it must
    reach the generic_api endpoint, while the same call made with a key that has no logging metadata delivers
    nothing. UNVERIFIED: that "generic_api" resolves as a per-request dynamic callback was read from
    Logging._process_dynamic_callback_list, not observed."""
    own, other = new_call_id(), new_call_id()
    with scoped.proxy.scenario() as scenario:
        logging_key: Final = scenario.key(metadata={"logging": _generic("success")})
        plain_key: Final = scenario.key()
        assert chat(scoped, own, key=logging_key).status_code == 200
        assert chat(scoped, other, key=plain_key).status_code == 200
        assert scoped.settled_events(own, 1)[0]["status"] == "success"
        assert scoped.events(other) == ()


@pytest.mark.parametrize(
    ("callback_type", "expected"),
    [
        pytest.param("success", ("success",), id="success"),
        pytest.param("failure", ("failure",), id="failure"),
        pytest.param("success_and_failure", ("failure", "success"), id="success_and_failure"),
    ],
)
def test_key_logging_callback_type_selects_which_outcomes_are_delivered(
    scoped: CallbackRig, callback_type: str, expected: tuple[str, ...]
) -> None:
    """A key's logging entry has callback_type "success", "failure" or "success_and_failure". The key makes one
    successful and one provider-rejected call. Only the outcomes the type names may be delivered to the
    generic_api endpoint: one payload each, nothing for the other outcome. UNVERIFIED as above."""
    ok, failed = new_call_id(), new_call_id()
    with scoped.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"logging": _generic(callback_type)})
        assert chat(scoped, ok, key=key).status_code == 200
        assert chat(scoped, failed, key=key, model=CHAT_REJECTED).status_code == 400
        scoped.hook(ok, "async_log_success_event")
        scoped.hook(failed, "async_log_failure_event")
        delivered: Final = held(
            lambda: tuple(sorted(str(event["status"]) for event in (*scoped.events(ok), *scoped.events(failed)))),
            lambda statuses: statuses == expected,
            holding=3,
            seconds=30,
        )
        assert delivered == expected


def test_team_logging_metadata_applies_to_every_key_of_the_team(scoped: CallbackRig) -> None:
    """A team is created with `metadata.logging` naming generic_api. A key of that team, with no logging metadata
    of its own, makes a successful call and one payload for it must be delivered, carrying the team's id in
    `metadata.user_api_key_team_id`. UNVERIFIED as above."""
    call_id: Final = new_call_id()
    with scoped.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"logging": _generic("success_and_failure")})
        key: Final = scenario.key(team_id=team)
        assert chat(scoped, call_id, key=key).status_code == 200
        event: Final = scoped.settled_events(call_id, 1)[0]
        assert object_value(event["metadata"])["user_api_key_team_id"] == team


def test_key_logging_metadata_wins_over_team_logging_metadata(scoped: CallbackRig) -> None:
    """The team's `metadata.logging` names generic_api, and one of its keys has `metadata.logging: []`. Key
    settings win outright, and an empty list counts as configured, so that key's successful call must deliver
    nothing, while a sibling key without its own logging metadata still delivers one payload."""
    silenced, inherited = new_call_id(), new_call_id()
    with scoped.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"logging": _generic("success")})
        empty_key: Final = scenario.key(team_id=team, metadata={"logging": []})
        team_key: Final = scenario.key(team_id=team)
        assert chat(scoped, silenced, key=empty_key).status_code == 200
        assert chat(scoped, inherited, key=team_key).status_code == 200
        scoped.settled_events(inherited, 1)
        assert scoped.events(silenced) == ()


def test_deprecated_team_callback_settings_metadata_still_adds_callbacks(scoped: CallbackRig) -> None:
    """A team carries the older `metadata.callback_settings: {success_callback: ["generic_api"], failure_callback:
    [], callback_vars: {}}` shape. A successful call by one of its keys must still deliver one payload.
    UNVERIFIED as above; the shape is read from get_dynamic_logging_metadata."""
    call_id: Final = new_call_id()
    settings: Final[dict[str, JsonValue]] = {
        "success_callback": ["generic_api"],
        "failure_callback": [],
        "callback_vars": {},
    }
    with scoped.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"callback_settings": settings})
        key: Final = scenario.key(team_id=team)
        assert chat(scoped, call_id, key=key).status_code == 200
        assert scoped.settled_events(call_id, 1)[0]["status"] == "success"


def test_key_scoped_callback_does_not_remove_the_global_callbacks(scoped: CallbackRig) -> None:
    """A key with its own `metadata.logging` makes a call on a proxy whose global `callbacks` hold the custom
    recorder. The recorder must still get its async_log_success_event exactly once: per-key callbacks are added to
    the global ones, they do not replace them."""
    call_id: Final = new_call_id()
    with scoped.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"logging": _generic("success")})
        assert chat(scoped, call_id, key=key).status_code == 200
        _settled_log_events(scoped, call_id, "recorder", ("async_log_success_event",))


def test_invalid_key_logging_entry_is_ignored_and_the_request_succeeds(scoped: CallbackRig) -> None:
    """A key's `metadata.logging` holds an entry with an unknown callback variable. The entry is dropped with a
    warning when the request is processed, so the call still answers 200 and nothing is delivered to the
    generic_api endpoint. UNVERIFIED: /key/generate may instead reject the metadata; in that case the contract
    to pin is the 4xx on key creation."""
    call_id: Final = new_call_id()
    entry: Final[dict[str, JsonValue]] = {
        "callback_name": "generic_api",
        "callback_type": "success",
        "callback_vars": {"dec1_not_a_callback_var": "x"},
    }
    with scoped.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"logging": [entry]})
        assert chat(scoped, call_id, key=key).status_code == 200
        scoped.hook(call_id, "async_log_success_event")
        held(lambda: scoped.events(call_id), lambda events: events == (), holding=3, seconds=10)


def test_global_disable_no_log_param_makes_no_log_requests_log_again(tmp_path_factory: pytest.TempPathFactory) -> None:
    """With `litellm_settings.global_disable_no_log_param: true`, a caller's `"no-log": true` is ignored: the
    generic_api payload for the call is delivered like for any other request."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-no-log-off"),
        {"callbacks": ["generic_api", RECORDER], "global_disable_no_log_param": True},
    ) as rig:
        call_id: Final = new_call_id()
        assert chat(rig, call_id, **{"no-log": True}).status_code == 200
        assert rig.event(call_id)["status"] == "success"


def test_generic_api_requests_carry_the_configured_headers_and_a_json_array(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The generic_api callback is configured through GENERIC_LOGGER_ENDPOINT and GENERIC_LOGGER_HEADERS
    ("Authorization=Bearer <secret>"). Every POST it makes must carry that Authorization header and
    Content-Type application/json, and its body must be a JSON array of payload objects (the default
    `json_array` log format)."""
    with callback_rig(tmp_path_factory.mktemp("dec1-callbacks-generic-wire"), {"callbacks": ["generic_api"]}) as rig:
        call_id: Final = new_call_id()
        assert chat(rig, call_id).status_code == 200
        requests: Final = eventually(lambda: rig.sink.wire.drain(), lambda found: len(found) >= 1, seconds=20)
        for request in requests:
            assert request.method == "POST"
            assert request.headers["authorization"] == f"Bearer {SINK_SECRET}"
            assert request.headers["content-type"].startswith("application/json")
            assert isinstance(json.loads(request.body), list)
        assert any(call_id.encode() in request.body for request in requests)


def test_success_callback_list_alone_delivers_only_success_payloads(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`generic_api` is listed only under `success_callback`. A successful call must deliver one payload and a
    provider-rejected call must deliver none."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-success-list"),
        {"success_callback": ["generic_api"], "callbacks": [RECORDER]},
    ) as rig:
        ok, failed = new_call_id(), new_call_id()
        assert chat(rig, ok).status_code == 200
        assert chat(rig, failed, model=CHAT_REJECTED).status_code == 400
        rig.hook(failed, "async_log_failure_event")
        assert rig.settled_events(ok, 1)[0]["status"] == "success"
        assert rig.events(failed) == ()


def test_failure_callback_list_alone_delivers_only_failure_payloads(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`generic_api` is listed only under `failure_callback`. A provider-rejected call must deliver one payload
    with status "failure" and a successful call must deliver none."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-failure-list"),
        {"failure_callback": ["generic_api"], "callbacks": [RECORDER]},
    ) as rig:
        ok, failed = new_call_id(), new_call_id()
        assert chat(rig, ok).status_code == 200
        assert chat(rig, failed, model=CHAT_REJECTED).status_code == 400
        rig.hook(ok, "async_log_success_event")
        assert rig.settled_events(failed, 1)[0]["status"] == "failure"
        assert rig.events(ok) == ()


def test_same_callback_in_callbacks_and_success_callback_logs_once(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`generic_api` is listed under both `callbacks` and `success_callback`. One successful call must still
    deliver exactly one payload, not two. UNVERIFIED: de-duplication across the lists was not traced; if this
    fails with two payloads, that is the current behaviour to record."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-duplicate"),
        {"callbacks": ["generic_api"], "success_callback": ["generic_api"]},
    ) as rig:
        call_id: Final = new_call_id()
        assert chat(rig, call_id).status_code == 200
        rig.settled_events(call_id, 1)


def test_callback_added_through_config_update_starts_receiving_events(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy boots from the default config with no generic_api callback, on its own database. An admin POSTs
    /config/update with `litellm_settings.success_callback: ["generic_api"]` and the endpoint in
    `environment_variables`, which must answer {"message": "Config updated successfully"}. Without a restart,
    GET /active/callbacks must come to list the generic API logger, and a chat sent after that must deliver a
    payload to the endpoint; a chat sent before the update delivered nothing."""
    with (
        scratch_database() as database_url,
        wire_server(ScriptedProvider()) as provider,
        wire_server(lambda request: Reply()) as sink,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DATABASE_URL": database_url, "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"},
                remove_environment=("DATABASE_URL_READ_REPLICA", "GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"),
            ) as proxy,
            proxy.scenario() as scenario,
        ):
            journal: Final = Journal(sink, batched=True)
            model: Final = scenario.model(api_base=provider.url + "/v1")
            before, after = marker(), marker()
            proxy.chat(model, text=before)
            updated: Final = proxy.post(
                "/config/update",
                {
                    "litellm_settings": {"success_callback": ["generic_api"]},
                    "environment_variables": {"GENERIC_LOGGER_ENDPOINT": sink.url},
                },
            )
            assert updated == {"message": "Config updated successfully"}
            eventually(
                lambda: str(proxy.get("/active/callbacks")["all_litellm_callbacks"]).lower(),
                lambda active: "generic" in active,
                seconds=40,
            )
            proxy.chat(model, text=after)
            eventually(journal.bodies, lambda body: after.encode() in body, seconds=25)
            assert before.encode() not in journal.bodies()


def test_service_callback_prometheus_system_exposes_service_latency_metrics(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`litellm_settings.service_callback: ["prometheus_system"]` makes the proxy record health metrics for the
    services it depends on. After one chat, GET /metrics must answer 200 and contain at least one
    `litellm_postgres_` or `litellm_redis_` service series (PrometheusServicesLogger names them
    litellm_<service>_<type>). `prometheus_system` only registers the series in the process's Prometheus registry;
    /metrics is mounted by `PrometheusLogger._mount_metrics_endpoint`, which runs when `prometheus` is in the
    callbacks (litellm/proxy/common_utils/callback_utils.py), so the rig lists it too. It is a Starlette mount at
    `/metrics` (litellm/integrations/prometheus.py:5183), so `GET /metrics` answers a redirect with an empty body
    (run 43044106414 saw '' on every poll); the series are read from `/metrics/`, as the other metrics tests do."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-service"),
        {"callbacks": [RECORDER, "prometheus"], "service_callback": ["prometheus_system"]},
    ) as rig:
        assert chat(rig, new_call_id(), model=CHAT).status_code == 200
        text: Final = eventually(
            lambda: rig.proxy.request("GET", "/metrics/").text,
            lambda body: "litellm_postgres_" in body or "litellm_redis_" in body,
            seconds=30,
        )
        assert "_latency" in text
