"""Settings matrix for `router_settings`: one test per key, each asserting what a caller can observe.

Every key under `router_settings` that names a router argument is handed to the router when the config
is loaded. The router core is already Rust, but the config file is still parsed in Python; once the
loader moves too, these are the keys that must keep arriving: aliases, fallbacks, retries and the routing
strategy. The observable is always the sequence of requests the scripted provider receives.

Assumption: the harness adds `num_retries: 0` to any config whose `router_settings` does not set it, so a
test that wants retries sets the key itself and a test that wants a single attempt relies on the default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration._support.wire import Reply, Request
from tests.integration.configuration.dec1_config_support import (
    booted,
    chat,
    chat_reply,
    deployment,
    refused,
    scripted_upstream,
    sent,
    sent_json,
    served_model_names,
)

_PRIMARY: Final = "dec1-primary"
_BACKUP: Final = "dec1-backup"
_PRIMARY_UPSTREAM: Final = "primary-upstream-model"
_BACKUP_UPSTREAM: Final = "backup-upstream-model"


def _primary_fails(request: Request) -> Reply:
    """The provider: 500 for the primary deployment's upstream model, 200 for everything else."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if JSON_OBJECT.validate_json(request.body).get("model") == _PRIMARY_UPSTREAM:
        return chat_reply("dec1 primary is down", status=500)
    return chat_reply("served by the backup")


def _pair(base: str) -> list[dict[str, object]]:
    return [
        dict(deployment(_PRIMARY, base, model=f"openai/{_PRIMARY_UPSTREAM}")),
        dict(deployment(_BACKUP, base, model=f"openai/{_BACKUP_UPSTREAM}")),
    ]


def _models_sent(requests: tuple[Request, ...]) -> list[object]:
    return [sent_json(request)["model"] for request in requests]


def test_model_group_alias_serves_a_request_for_the_alias_from_the_real_group(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.model_group_alias: {dec1-alias: dec1-primary}`. A chat completion asking for
    `dec1-alias` must succeed and reach the provider with the real deployment's upstream model."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": _pair(f"{upstream.url}/v1"),
            "router_settings": {"model_group_alias": {"dec1-alias": _PRIMARY}},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, "dec1-alias")
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM]


def test_model_group_alias_is_listed_by_the_models_endpoint(gateway: Gateway, tmp_path: Path) -> None:
    """With a model group alias configured, `GET /v1/models` must list the alias name next to the real
    group names, so clients can discover it.
    UNVERIFIED: aliases given as a plain string are listed (an alias given as `{model, hidden: true}` is
    not); see the alias handling in `litellm/proxy/auth/model_checks.py`."""
    config: Final = {
        "model_list": _pair("http://127.0.0.1:9/v1"),
        "router_settings": {"model_group_alias": {"dec1-alias": _PRIMARY}},
    }
    with booted(gateway, tmp_path, config) as owned:
        names: Final = served_model_names(owned.gateway)
    assert {"dec1-alias", _PRIMARY, _BACKUP} <= names, sorted(names)


def test_fallbacks_send_a_failed_request_to_the_configured_backup_group(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.fallbacks: [{dec1-primary: [dec1-backup]}]`. The primary's provider answers 500.
    A chat completion for the primary must still return 200 with the backup's content, and the provider
    must have received the primary's upstream model first and the backup's second."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {
            "model_list": _pair(f"{upstream.url}/v1"),
            "router_settings": {"num_retries": 0, "fallbacks": [{_PRIMARY: [_BACKUP]}]},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    choice: Final = object_value(JSON_OBJECT.validate_json(response.content)["choices"][0])
    assert object_value(choice["message"])["content"] == "served by the backup"
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM, _BACKUP_UPSTREAM]


def test_without_fallbacks_a_failed_request_returns_the_provider_error(gateway: Gateway, tmp_path: Path) -> None:
    """Control: the same failing primary with no `fallbacks`. The caller must get a 500 and the provider
    must have received exactly one request, for the primary."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {"model_list": _pair(f"{upstream.url}/v1"), "router_settings": {"num_retries": 0}}
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY)
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM]


def test_default_fallbacks_apply_to_a_group_with_no_fallback_entry_of_its_own(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.default_fallbacks: [dec1-backup]` and no per-group `fallbacks`. A failing request
    for the primary must be retried on the backup group and return 200."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {
            "model_list": _pair(f"{upstream.url}/v1"),
            "router_settings": {"num_retries": 0, "default_fallbacks": [_BACKUP]},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM, _BACKUP_UPSTREAM]


def test_a_request_can_turn_configured_fallbacks_off(gateway: Gateway, tmp_path: Path) -> None:
    """With fallbacks configured, a chat completion that sends `disable_fallbacks: true` must get the
    primary's 500 and the backup must not be called.
    UNVERIFIED: that `disable_fallbacks` is accepted from the request body on the proxy; see
    `litellm/router_utils/fallback_event_handlers.py`."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {
            "model_list": _pair(f"{upstream.url}/v1"),
            "router_settings": {"num_retries": 0, "fallbacks": [{_PRIMARY: [_BACKUP]}]},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY, disable_fallbacks=True)
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM]


def test_num_retries_repeats_a_failing_provider_call_that_many_extra_times(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.num_retries: 2`. The provider always answers 500. One chat completion must return
    500 after the provider has received exactly three requests: the first attempt plus two retries."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {"model_list": _pair(f"{upstream.url}/v1"), "router_settings": {"num_retries": 2}}
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY)
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert _models_sent(requests) == [_PRIMARY_UPSTREAM] * 3


def test_num_retries_zero_makes_a_single_attempt(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.num_retries: 0`. The provider always answers 500. One chat completion must return
    500 after exactly one provider request."""
    with scripted_upstream(_primary_fails) as upstream:
        config: Final = {"model_list": _pair(f"{upstream.url}/v1"), "router_settings": {"num_retries": 0}}
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PRIMARY)
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert len(requests) == 1


def test_router_settings_are_reported_by_the_config_callbacks_endpoint(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings` sets `num_retries: 4` and `routing_strategy: least-busy`. `GET
    /get/config/callbacks` must return `router_settings` with those two values, which is what the admin
    UI shows as the live router configuration.
    UNVERIFIED: the key names inside the returned `router_settings` (taken to match the config keys)."""
    config: Final = {
        "model_list": _pair("http://127.0.0.1:9/v1"),
        "router_settings": {"num_retries": 4, "routing_strategy": "least-busy"},
    }
    with booted(gateway, tmp_path, config) as owned:
        settings: Final = object_value(owned.gateway.get("/get/config/callbacks")["router_settings"])
    assert settings["num_retries"] == 4
    assert settings["routing_strategy"] == "least-busy"


def test_an_unknown_routing_strategy_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.routing_strategy: dec1-not-a-strategy`. The proxy must not start serving with a
    silently substituted strategy: it must exit non-zero and name the bad value in its log.
    UNVERIFIED: that an unknown strategy fails router construction rather than falling back to the
    default; see the strategy validation in `litellm/router.py`."""
    log: Final = refused(
        gateway,
        tmp_path,
        {
            "model_list": _pair("http://127.0.0.1:9/v1"),
            "router_settings": {"routing_strategy": "dec1-not-a-strategy"},
        },
    )
    assert "dec1-not-a-strategy" in log
