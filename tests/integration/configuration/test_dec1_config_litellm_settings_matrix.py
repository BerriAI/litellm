"""Settings matrix for `litellm_settings`: one test per key, each asserting what a caller can observe.

`primary` is one proxy that sets the keys which compose (request timeout, a custom callback loaded from
a Python file beside the config, the local response cache, default user, key and team parameters, and
per-model-group header forwarding). `bounded` sets the keys that restrict key generation (upper bounds,
who may create personal keys, internal user budgets). Keys that stop the boot get their own proxy.

Most of these keys are written onto module-level litellm state by the Python loader and read from there
by request handlers. A Rust config loader must reach the same handlers with the same values, and must
keep loading operator-supplied Python (callbacks) relative to the config file.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import OwnedProxy
from tests.integration._support.wire import Reply, Request, Wire
from tests.integration.configuration.dec1_config_support import (
    active_settings,
    booted,
    chat,
    chat_reply,
    deployment,
    readiness_details,
    refused,
    scripted_upstream,
    sent,
)

_MODEL: Final = "dec1-litellm-model"
_FORWARDING: Final = "dec1-forwarding-model"
_FAILING: Final = "dec1-failing-model"
_DEFAULT_TEAM: Final = "dec1-default-team"
_EVENTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_CALLBACK_SOURCE: Final = """\
import json
import os

from litellm.integrations.custom_logger import CustomLogger


class Dec1SinkLogger(CustomLogger):
    def _write(self, event, kwargs):
        with open(os.environ["DEC1_CALLBACK_SINK"], "a") as sink:
            sink.write(json.dumps({"event": event, "messages": str(kwargs.get("messages"))}) + "\\n")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write("success", kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write("failure", kwargs)


handler_instance = Dec1SinkLogger()


def on_success(kwargs, completion_response, start_time, end_time):
    return None
"""


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    upstream: Wire
    sink: Path

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    def events(self) -> Sequence[dict[str, JsonValue]]:
        if not self.sink.exists():
            return []
        return _EVENTS.validate_python([json.loads(line) for line in self.sink.read_text().splitlines() if line])


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if JSON_OBJECT.validate_json(request.body).get("model") == "failing-upstream-model":
        return chat_reply("dec1 scripted failure", status=500)
    return chat_reply()


@pytest.fixture(scope="module")
def primary(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-litellm-primary")
    sink: Final = directory / "callback-events.jsonl"
    (directory / "dec1_callback.py").write_text(_CALLBACK_SOURCE)
    with scripted_upstream(_respond) as upstream, gateway_from_environment() as shared:
        base: Final = f"{upstream.url}/v1"
        config: Final = {
            "model_list": [
                deployment(_MODEL, base),
                deployment(_FORWARDING, base),
                deployment(_FAILING, base, model="openai/failing-upstream-model"),
            ],
            "litellm_settings": {
                "request_timeout": 42,
                "callbacks": ["dec1_callback.handler_instance"],
                "success_callback": ["dec1_callback.on_success"],
                "cache": True,
                "cache_params": {"type": "local"},
                "default_internal_user_params": {"max_budget": 7, "models": [_MODEL]},
                "default_key_generate_params": {"max_budget": 2, "models": [_MODEL]},
                "default_team_settings": [{"team_id": _DEFAULT_TEAM, "max_budget": 9}],
                "model_group_settings": {"forward_client_headers_to_llm_api": [_FORWARDING]},
            },
        }
        with booted(shared, directory, config, environment={"DEC1_CALLBACK_SINK": str(sink)}) as proxy:
            yield _Rig(proxy, upstream, sink)


@pytest.fixture(scope="module")
def bounded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-litellm-bounded")
    config: Final = {
        "litellm_settings": {
            "upperbound_key_generate_params": {"max_budget": 5, "duration": "1h", "rpm_limit": 10},
            "key_generation_settings": {"personal_key_generation": {"allowed_user_roles": ["proxy_admin"]}},
            "max_internal_user_budget": 3,
            "internal_user_budget_duration": "7d",
        }
    }
    with gateway_from_environment() as shared, booted(shared, directory, config) as proxy:
        yield proxy


def _fixed_chat(gateway: Gateway, model: str, content: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return gateway.post(
        "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}], **fields}
    )


def test_request_timeout_is_reported_by_the_active_settings_endpoint(primary: _Rig) -> None:
    """`request_timeout: 42`. `GET /active/callbacks` must report `litellm.request_timeout` as 42."""
    assert active_settings(primary.gateway)["litellm.request_timeout"] == 42


def test_callbacks_loads_a_custom_logger_from_a_file_beside_the_config(primary: _Rig) -> None:
    """`callbacks: ["dec1_callback.handler_instance"]` names an object in `dec1_callback.py` next to the
    config file. `GET /active/callbacks` must list an entry in `litellm.callbacks` whose text contains the
    class name `Dec1SinkLogger`, proving the module was found relative to the config and registered."""
    callbacks: Final = active_settings(primary.gateway)["litellm.callbacks"]
    assert isinstance(callbacks, list)
    assert any("Dec1SinkLogger" in string_value(entry) for entry in callbacks), callbacks


def test_a_configured_custom_logger_receives_a_success_event_for_a_completed_call(primary: _Rig) -> None:
    """With the custom logger configured, a successful chat completion must make the logger write one
    `success` event carrying that request's prompt text to its sink file. Polls the file, since callbacks
    run after the response is returned."""
    marker: Final = f"dec1 success marker {uuid.uuid4().hex}"
    _fixed_chat(primary.gateway, _MODEL, marker)
    events: Final = eventually(
        primary.events,
        lambda seen: any(event["event"] == "success" and marker in str(event["messages"]) for event in seen),
        seconds=30,
    )
    assert [event["event"] for event in events if marker in str(event["messages"])] == ["success"]


def test_a_configured_custom_logger_receives_a_failure_event_when_the_provider_errors(primary: _Rig) -> None:
    """A chat completion to a deployment whose provider answers 500 must return an error to the caller
    (status 500) and make the custom logger write a `failure` event carrying that request's prompt text."""
    marker: Final = f"dec1 failure marker {uuid.uuid4().hex}"
    response: Final = primary.gateway.request(
        "POST", "/v1/chat/completions", {"model": _FAILING, "messages": [{"role": "user", "content": marker}]}
    )
    assert response.status_code == 500, response.text
    eventually(
        primary.events,
        lambda seen: any(event["event"] == "failure" and marker in str(event["messages"]) for event in seen),
        seconds=30,
    )


def test_success_callback_function_is_listed_by_authenticated_readiness_details(primary: _Rig) -> None:
    """`success_callback: ["dec1_callback.on_success"]` names a plain function. The authenticated
    `GET /health/readiness/details` must list `on_success` in `success_callbacks`: callbacks are reported
    by function name, never by their configuration values."""
    callbacks: Final = readiness_details(primary.gateway)["success_callbacks"]
    assert isinstance(callbacks, list)
    assert "on_success" in callbacks, callbacks


def test_cache_with_local_type_is_reported_by_readiness_details(primary: _Rig) -> None:
    """`cache: true` with `cache_params.type: local`. `GET /health/readiness/details` must report
    `cache` as `local`."""
    assert readiness_details(primary.gateway)["cache"] == "local"


def test_cache_serves_an_identical_second_request_without_calling_the_provider(primary: _Rig) -> None:
    """With the response cache on, two byte-identical chat completions must return the same completion
    `id` while the provider receives exactly one request: the second answer comes from the cache."""
    content: Final = f"dec1 cached prompt {uuid.uuid4().hex}"
    sent(primary.upstream)
    first: Final = _fixed_chat(primary.gateway, _MODEL, content)
    second: Final = _fixed_chat(primary.gateway, _MODEL, content)
    assert second["id"] == first["id"]
    assert len(sent(primary.upstream)) == 1


def test_a_request_with_no_cache_bypasses_the_configured_cache(primary: _Rig) -> None:
    """With the response cache on, a repeat of a cached request that adds `cache: {"no-cache": true}` must
    call the provider again: the provider receives two requests in total.
    UNVERIFIED: that the proxy accepts the per-request `cache` control from any caller without extra
    settings; see `litellm/caching/caching_handler.py`."""
    content: Final = f"dec1 no-cache prompt {uuid.uuid4().hex}"
    sent(primary.upstream)
    _fixed_chat(primary.gateway, _MODEL, content)
    _fixed_chat(primary.gateway, _MODEL, content, cache={"no-cache": True})
    assert len(sent(primary.upstream)) == 2


def test_default_internal_user_params_fill_a_new_users_budget_and_models(primary: _Rig) -> None:
    """`default_internal_user_params: {max_budget: 7, models: [...]}`. `POST /user/new` with only a
    `user_id` must create a `LiteLLM_UserTable` row with `max_budget` 7 and the configured model list."""
    with primary.gateway.scenario() as scenario:
        user_id: Final = scenario.user()
        rows: Final = primary.gateway.rows(
            'SELECT max_budget, models FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)
        )
    assert len(rows) == 1
    assert rows[0]["max_budget"] == 7
    assert rows[0]["models"] == [_MODEL]


def test_default_internal_user_params_do_not_override_a_value_the_request_gives(primary: _Rig) -> None:
    """With default user parameters configured, `POST /user/new` that names its own `max_budget: 1` must
    store 1, not the default 7."""
    with primary.gateway.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=1)
        rows: Final = primary.gateway.rows('SELECT max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert rows[0]["max_budget"] == 1


def test_default_internal_user_params_are_not_applied_to_a_new_proxy_admin(primary: _Rig) -> None:
    """With default user parameters configured, `POST /user/new` with `user_role: proxy_admin` must not
    receive the default budget: the stored `max_budget` stays null."""
    with primary.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="proxy_admin")
        rows: Final = primary.gateway.rows('SELECT max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert rows[0]["max_budget"] is None


def test_default_key_generate_params_fill_a_new_keys_budget_and_models(primary: _Rig) -> None:
    """`default_key_generate_params: {max_budget: 2, models: [...]}`. `POST /key/generate` with an empty
    body must return a key whose `max_budget` is 2 and whose `models` is the configured list."""
    created: Final = primary.gateway.post("/key/generate", {})
    try:
        assert created["max_budget"] == 2
        assert created["models"] == [_MODEL]
    finally:
        primary.gateway.post("/key/delete", {"keys": [string_value(created["key"])]})


def test_default_key_generate_params_yield_to_a_value_in_the_request(primary: _Rig) -> None:
    """With default key parameters configured, `POST /key/generate` that names `max_budget: 1` must
    return a key with `max_budget` 1."""
    created: Final = primary.gateway.post("/key/generate", {"max_budget": 1})
    try:
        assert created["max_budget"] == 1
    finally:
        primary.gateway.post("/key/delete", {"keys": [string_value(created["key"])]})


def test_default_team_settings_first_entry_budget_is_given_to_a_new_team_without_one(primary: _Rig) -> None:
    """`default_team_settings` has one entry with `max_budget: 9`. `POST /team/new` without a
    `max_budget` must create a `LiteLLM_TeamTable` row with `max_budget` 9: the first entry's budget is
    the legacy default for every new team."""
    with primary.gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        rows: Final = primary.gateway.rows('SELECT max_budget FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert rows[0]["max_budget"] == 9


def test_model_group_settings_forwards_client_headers_only_for_the_listed_model_group(primary: _Rig) -> None:
    """`model_group_settings.forward_client_headers_to_llm_api` lists one model group. A chat completion
    to that group sent with `x-dec1-trace: abc123` must reach the provider with the header; the same call
    to a group not on the list must reach the provider without it."""
    sent(primary.upstream)
    listed: Final = chat(primary.gateway, _FORWARDING, headers={"x-dec1-trace": "abc123"})
    assert listed.status_code == 200, listed.text
    listed_headers: Final = [request.headers.get("x-dec1-trace") for request in sent(primary.upstream)]
    unlisted: Final = chat(primary.gateway, _MODEL, headers={"x-dec1-trace": "abc123"})
    assert unlisted.status_code == 200, unlisted.text
    unlisted_headers: Final = [request.headers.get("x-dec1-trace") for request in sent(primary.upstream)]
    assert (listed_headers, unlisted_headers) == (["abc123"], [None])


def test_upperbound_key_generate_params_reject_a_budget_over_the_limit(bounded: OwnedProxy) -> None:
    """`upperbound_key_generate_params.max_budget: 5`. `POST /key/generate` with `max_budget: 10` must be
    refused with 400 and a message containing `max_budget is over max limit set in config -
    user_value=10.0; max_value=5.0`."""
    response: Final = bounded.gateway.request("POST", "/key/generate", {"max_budget": 10})
    assert response.status_code == 400, response.text
    assert "max_budget is over max limit set in config - user_value=10.0; max_value=5.0" in response.text


def test_upperbound_key_generate_params_reject_a_duration_over_the_limit(bounded: OwnedProxy) -> None:
    """`upperbound_key_generate_params.duration: 1h`. `POST /key/generate` with `duration: 2h` must be
    refused with 400 and a message containing `duration is over max limit set in config - user_value=2h;
    max_value=1h`."""
    response: Final = bounded.gateway.request("POST", "/key/generate", {"duration": "2h"})
    assert response.status_code == 400, response.text
    assert "duration is over max limit set in config - user_value=2h; max_value=1h" in response.text


def test_upperbound_key_generate_params_reject_a_never_expiring_key(bounded: OwnedProxy) -> None:
    """With an upper bound on `duration`, a request for `duration: "-1"` (never expire) counts as infinite
    and must be refused with 400 naming `duration is over max limit set in config`."""
    response: Final = bounded.gateway.request("POST", "/key/generate", {"duration": "-1"})
    assert response.status_code == 400, response.text
    assert "duration is over max limit set in config - user_value=-1; max_value=1h" in response.text


def test_upperbound_key_generate_params_reject_a_rate_limit_over_the_limit(bounded: OwnedProxy) -> None:
    """`upperbound_key_generate_params.rpm_limit: 10`. `POST /key/generate` with `rpm_limit: 11` must be
    refused with 400 and a message containing `rpm_limit is over max limit set in config - user_value=11;
    max_value=10`."""
    response: Final = bounded.gateway.request("POST", "/key/generate", {"rpm_limit": 11})
    assert response.status_code == 400, response.text
    assert "rpm_limit is over max limit set in config - user_value=11; max_value=10" in response.text


def test_upperbound_key_generate_params_become_the_defaults_of_a_key_that_names_none(bounded: OwnedProxy) -> None:
    """With upper bounds configured, `POST /key/generate` with an empty body must return a key whose
    `max_budget` is 5 and `rpm_limit` is 10 and which has an expiry (from the 1h bound): omitted fields
    are filled with the bound, not left unlimited."""
    created: Final = bounded.gateway.post("/key/generate", {})
    try:
        assert created["max_budget"] == 5
        assert created["rpm_limit"] == 10
        assert created["expires"] is not None
    finally:
        bounded.gateway.post("/key/delete", {"keys": [string_value(created["key"])]})


def test_upperbound_key_generate_params_accept_values_at_the_limit(bounded: OwnedProxy) -> None:
    """A request exactly at the bounds (`max_budget: 5`, `duration: 1h`, `rpm_limit: 10`) must succeed:
    the bound is inclusive."""
    created: Final = bounded.gateway.post("/key/generate", {"max_budget": 5, "duration": "1h", "rpm_limit": 10})
    try:
        assert created["max_budget"] == 5
    finally:
        bounded.gateway.post("/key/delete", {"keys": [string_value(created["key"])]})


def test_key_generation_settings_stop_an_internal_user_from_creating_a_personal_key(bounded: OwnedProxy) -> None:
    """`key_generation_settings.personal_key_generation.allowed_user_roles: ["proxy_admin"]`. An admin
    creates an internal user and a key for them; that user's key then calls `POST /key/generate` and must
    be refused with 400 and a message containing `Personal key creation has been restricted by admin.`"""
    with bounded.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        user_key: Final = scenario.key(user_id=user_id)
        response: Final = bounded.gateway.request("POST", "/key/generate", {}, key=user_key)
    assert response.status_code == 400, response.text
    assert "Personal key creation has been restricted by admin." in response.text


def test_key_generation_settings_still_let_an_allowed_role_create_a_personal_key(bounded: OwnedProxy) -> None:
    """With personal key creation restricted to `proxy_admin`, the master key (a proxy admin) must still
    create a key with 200."""
    with bounded.gateway.scenario() as scenario:
        assert scenario.key().startswith("sk-")


def test_max_internal_user_budget_and_duration_apply_to_a_new_internal_user(bounded: OwnedProxy) -> None:
    """`max_internal_user_budget: 3` and `internal_user_budget_duration: 7d`. `POST /user/new` with
    `user_role: internal_user` and no budget must store `max_budget` 3 and `budget_duration` `7d`."""
    with bounded.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        rows: Final = bounded.gateway.rows(
            'SELECT max_budget, budget_duration FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)
        )
    assert rows[0]["max_budget"] == 3
    assert rows[0]["budget_duration"] == "7d"


def test_max_internal_user_budget_does_not_apply_to_other_roles(bounded: OwnedProxy) -> None:
    """With `max_internal_user_budget: 3`, `POST /user/new` with `user_role: internal_user_viewer` must
    not receive the budget: the stored `max_budget` stays null."""
    with bounded.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user_viewer")
        rows: Final = bounded.gateway.rows('SELECT max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert rows[0]["max_budget"] is None


def test_default_team_settings_entry_without_a_team_id_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """A `default_team_settings` entry has no `team_id`. The proxy must exit non-zero and log
    `team_id missing from default_team_settings at index=0`."""
    log: Final = refused(gateway, tmp_path, {"litellm_settings": {"default_team_settings": [{"max_budget": 1}]}})
    assert "team_id missing from default_team_settings at index=0" in log


def test_upperbound_key_generate_params_that_is_not_a_mapping_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`upperbound_key_generate_params: 5` (a number instead of a mapping). The proxy must exit non-zero
    and log `Invalid value set for upperbound_key_generate_params - value=5`."""
    log: Final = refused(gateway, tmp_path, {"litellm_settings": {"upperbound_key_generate_params": 5}})
    assert "Invalid value set for upperbound_key_generate_params - value=5" in log


def test_a_callback_naming_a_missing_module_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`callbacks: ["dec1_absent_callback.handler"]` names a module that does not exist. The proxy must
    exit non-zero and name the module in its log rather than start without the operator's callback.
    UNVERIFIED: the exact wording of the import error."""
    log: Final = refused(gateway, tmp_path, {"litellm_settings": {"callbacks": ["dec1_absent_callback.handler"]}})
    assert "dec1_absent_callback" in log


def test_cache_true_without_cache_params_uses_redis_from_the_environment(gateway: Gateway, tmp_path: Path) -> None:
    """`cache: true` with no `cache_params`. The cache type defaults to redis and the host and port come
    from `REDIS_HOST` and `REDIS_PORT`, which the harness sets for the owned proxy. `GET
    /health/readiness/details` must report `cache` as `redis`."""
    with booted(gateway, tmp_path, {"litellm_settings": {"cache": True}}) as owned:
        assert readiness_details(owned.gateway)["cache"] == "redis"


def test_cache_false_leaves_the_cache_off_even_with_cache_params(gateway: Gateway, tmp_path: Path) -> None:
    """`cache: false` next to a `cache_params` block. `GET /health/readiness/details` must report `cache`
    as null and two identical requests must both reach the provider."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1")],
            "litellm_settings": {"cache": False, "cache_params": {"type": "local"}},
        }
        with booted(gateway, tmp_path, config) as owned:
            content: Final = f"dec1 uncached prompt {uuid.uuid4().hex}"
            _fixed_chat(owned.gateway, _MODEL, content)
            _fixed_chat(owned.gateway, _MODEL, content)
            cache: Final = readiness_details(owned.gateway)["cache"]
            requests: Final = sent(upstream)
    assert cache is None
    assert len(requests) == 2


def test_cache_params_supported_call_types_limits_what_is_cached(gateway: Gateway, tmp_path: Path) -> None:
    """`cache_params.supported_call_types: ["aembedding"]` on a local cache. Chat completions are outside
    the list, so two identical chat requests must both reach the provider.
    UNVERIFIED: the call-type spelling the proxy's async chat path checks (`acompletion`); see
    `litellm/caching/caching.py`."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1")],
            "litellm_settings": {
                "cache": True,
                "cache_params": {"type": "local", "supported_call_types": ["aembedding"]},
            },
        }
        with booted(gateway, tmp_path, config) as owned:
            content: Final = f"dec1 call-type prompt {uuid.uuid4().hex}"
            _fixed_chat(owned.gateway, _MODEL, content)
            _fixed_chat(owned.gateway, _MODEL, content)
            requests: Final = sent(upstream)
    assert len(requests) == 2


def test_the_error_from_a_failing_provider_keeps_the_openai_error_shape(primary: _Rig) -> None:
    """Control for the failure-callback case: the caller of a deployment whose provider answers 500 must
    get a JSON body with a top-level `error` object holding a `message`."""
    response: Final = chat(primary.gateway, _FAILING)
    assert response.status_code == 500, response.text
    assert "message" in object_value(JSON_OBJECT.validate_json(response.content)["error"])
