"""Settings matrix for `general_settings`: one test per key, each asserting what a caller can observe.

Two shared proxies carry most of the matrix. `opened` sets every permissive key at once (custom key
header, user header, header forwarding, client-side credentials, wildcard listing, public readiness
details, alerting, drain endpoint, prompt storage). `strict` sets the restrictive keys
(`enforce_user_param`, `reject_clientside_metadata_tags`) and otherwise leaves defaults, so the same
request sent to both shows the key's effect and its default. Keys whose effect would disturb the shared
proxies (drain, `completion_model`, custom auth, pass-through routes, enterprise-gated keys) boot their own.

Each key is read by the Python proxy from the loaded `general_settings` mapping at request time; the Rust
config loader has to deliver the same values to the same decision points.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import OwnedProxy
from tests.integration._support.wire import Wire
from tests.integration.configuration.dec1_config_support import (
    active_settings,
    booted,
    chat,
    deployment,
    error_message,
    refused,
    scripted_upstream,
    sent,
    sent_json,
    served_model_names,
)

_MODEL: Final = "dec1-general-model"
_WILDCARD: Final = "dec1-wild/*"
_KEY_HEADER: Final = "X-Dec1-Key"
_USER_HEADER: Final = "X-Dec1-User"
_DRAIN_TOKEN: Final = "dec1-drain-token"
_NOT_PREMIUM: Final = "You must be a LiteLLM Enterprise user to use this feature."
_DETAILED_READINESS_KEYS: Final = frozenset(
    {
        "status",
        "db",
        "cache",
        "litellm_version",
        "success_callbacks",
        "use_aiohttp_transport",
        "log_level",
        "is_detailed_debug",
        "show_no_redis_warning",
        "show_env_credential_login_warning",
    }
)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    upstream: Wire
    second_upstream: Wire

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _models(upstream: Wire) -> list[dict[str, object]]:
    return [
        dict(deployment(_MODEL, f"{upstream.url}/v1")),
        dict(deployment(_WILDCARD, f"{upstream.url}/v1", model="openai/*")),
    ]


@pytest.fixture(scope="module")
def opened(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-general-opened")
    with scripted_upstream() as upstream, scripted_upstream() as second, gateway_from_environment() as shared:
        config: Final = {
            "model_list": _models(upstream),
            "general_settings": {
                "litellm_key_header_name": _KEY_HEADER,
                "user_header_name": _USER_HEADER,
                "forward_openai_org_id": True,
                "forward_client_headers_to_llm_api": True,
                "allow_client_side_credentials": True,
                "return_wildcard_routes": True,
                "allow_public_health_readiness_details": True,
                "alerting": ["slack"],
                "enable_drain_endpoint": True,
                "drain_endpoint_token": _DRAIN_TOKEN,
                "store_prompts_in_spend_logs": True,
            },
        }
        with booted(shared, directory, config, remove_environment=("DRAIN_ENDPOINT_TOKEN",)) as proxy:
            yield _Rig(proxy, upstream, second)


@pytest.fixture(scope="module")
def strict(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-general-strict")
    with scripted_upstream() as upstream, scripted_upstream() as second, gateway_from_environment() as shared:
        config: Final = {
            "model_list": _models(upstream),
            "general_settings": {"enforce_user_param": True, "reject_clientside_metadata_tags": True},
        }
        with booted(shared, directory, config, remove_environment=("DRAIN_ENDPOINT_TOKEN",)) as proxy:
            yield _Rig(proxy, upstream, second)


def _strict_chat_fields() -> dict[str, str]:
    return {"user": f"dec1-user-{uuid.uuid4().hex}"}


def test_litellm_key_header_name_accepts_the_key_from_the_configured_header(opened: _Rig) -> None:
    """`litellm_key_header_name: X-Dec1-Key`. A request with no `Authorization` header and the master key
    in `X-Dec1-Key: Bearer <key>` must be authenticated (200 on `GET /v1/models`)."""
    response: Final = opened.gateway.client.get("/v1/models", headers={_KEY_HEADER: f"Bearer {opened.gateway.key}"})
    assert response.status_code == 200, response.text


def test_litellm_key_header_name_takes_precedence_over_a_bad_authorization_header(opened: _Rig) -> None:
    """With the custom key header configured, a request carrying a wrong key in `Authorization` and the
    right key in `X-Dec1-Key` must be authenticated: when the custom header is present it is the key."""
    response: Final = opened.gateway.client.get(
        "/v1/models",
        headers={"Authorization": "Bearer sk-dec1-wrong", _KEY_HEADER: f"Bearer {opened.gateway.key}"},
    )
    assert response.status_code == 200, response.text


def test_without_litellm_key_header_name_a_key_in_that_header_is_not_a_credential(strict: _Rig) -> None:
    """Default: the proxy does not know `X-Dec1-Key`. A request with the master key only in that header
    must be rejected with 401."""
    response: Final = strict.gateway.client.get("/v1/models", headers={_KEY_HEADER: f"Bearer {strict.gateway.key}"})
    assert response.status_code == 401, response.text


def test_user_header_name_fills_the_user_field_sent_to_the_provider(opened: _Rig) -> None:
    """`user_header_name: X-Dec1-User`. A chat completion without a `user` in the body and with
    `X-Dec1-User: dec1-header-user` must reach the provider with `"user": "dec1-header-user"` in the JSON
    body."""
    sent(opened.upstream)
    response: Final = chat(opened.gateway, _MODEL, headers={_USER_HEADER: "dec1-header-user"})
    assert response.status_code == 200, response.text
    assert [sent_json(request).get("user") for request in sent(opened.upstream)] == ["dec1-header-user"]


def test_user_header_name_does_not_replace_a_user_given_in_the_body(opened: _Rig) -> None:
    """With the user header configured, a body that already names a `user` keeps it: the provider must
    receive the body's value, not the header's."""
    sent(opened.upstream)
    response: Final = chat(opened.gateway, _MODEL, headers={_USER_HEADER: "dec1-header-user"}, user="dec1-body-user")
    assert response.status_code == 200, response.text
    assert [sent_json(request).get("user") for request in sent(opened.upstream)] == ["dec1-body-user"]


def test_forward_openai_org_id_passes_the_callers_organization_header_upstream(opened: _Rig) -> None:
    """`forward_openai_org_id: true`. A chat completion sent with `OpenAI-Organization: org-dec1` must
    reach the provider with the same `openai-organization` header."""
    sent(opened.upstream)
    response: Final = chat(opened.gateway, _MODEL, headers={"OpenAI-Organization": "org-dec1"})
    assert response.status_code == 200, response.text
    assert [request.headers.get("openai-organization") for request in sent(opened.upstream)] == ["org-dec1"]


def test_without_forward_openai_org_id_the_organization_header_stops_at_the_proxy(strict: _Rig) -> None:
    """Default: the caller's `OpenAI-Organization` header must not appear on the provider request."""
    sent(strict.upstream)
    response: Final = chat(strict.gateway, _MODEL, headers={"OpenAI-Organization": "org-dec1"}, **_strict_chat_fields())
    assert response.status_code == 200, response.text
    assert [request.headers.get("openai-organization") for request in sent(strict.upstream)] == [None]


def test_forward_client_headers_to_llm_api_passes_x_headers_upstream(opened: _Rig) -> None:
    """`forward_client_headers_to_llm_api: true`. A chat completion sent with `x-dec1-trace: abc123` must
    reach the provider with that header and value; any `x-` header except `x-stainless-*` is forwarded."""
    sent(opened.upstream)
    response: Final = chat(opened.gateway, _MODEL, headers={"x-dec1-trace": "abc123"})
    assert response.status_code == 200, response.text
    assert [request.headers.get("x-dec1-trace") for request in sent(opened.upstream)] == ["abc123"]


def test_without_forward_client_headers_to_llm_api_x_headers_stop_at_the_proxy(strict: _Rig) -> None:
    """Default: a caller's `x-dec1-trace` header must not appear on the provider request."""
    sent(strict.upstream)
    response: Final = chat(strict.gateway, _MODEL, headers={"x-dec1-trace": "abc123"}, **_strict_chat_fields())
    assert response.status_code == 200, response.text
    assert [request.headers.get("x-dec1-trace") for request in sent(strict.upstream)] == [None]


def test_allow_client_side_credentials_lets_the_body_redirect_the_call(opened: _Rig) -> None:
    """`allow_client_side_credentials: true`. A chat completion whose body carries its own `api_base` and
    `api_key` must be accepted (200) and sent to that `api_base` with that key, not to the deployment's.
    UNVERIFIED: that the router honours the body's `api_base` for a named deployment once auth admits it;
    auth admission is read from `auth_utils.py`, the routing from `litellm/router.py`."""
    sent(opened.upstream)
    sent(opened.second_upstream)
    response: Final = chat(
        opened.gateway, _MODEL, api_base=f"{opened.second_upstream.url}/v1", api_key="sk-dec1-client-side"
    )
    assert response.status_code == 200, response.text
    assert sent(opened.upstream) == ()
    assert [request.headers["authorization"] for request in sent(opened.second_upstream)] == [
        "Bearer sk-dec1-client-side"
    ]


def test_without_allow_client_side_credentials_an_api_base_in_the_body_is_rejected(strict: _Rig) -> None:
    """Default: a body carrying `api_base` must be refused before any provider call, with a message that
    starts `Rejected Request: api_base is not allowed in request body.` and names
    `general_settings.allow_client_side_credentials` as the opt-in. Neither upstream may receive a request.
    UNVERIFIED: the status code; the check raises a plain ValueError inside auth, which the auth handler
    turns into a 401 `Authentication Error, ...`."""
    sent(strict.upstream)
    sent(strict.second_upstream)
    response: Final = chat(strict.gateway, _MODEL, api_base=f"{strict.second_upstream.url}/v1", **_strict_chat_fields())
    assert response.status_code == 401, response.text
    message: Final = error_message(response)
    assert "Rejected Request: api_base is not allowed in request body." in message
    assert "`general_settings.allow_client_side_credentials = true`" in message
    assert sent(strict.upstream) == () and sent(strict.second_upstream) == ()


def test_return_wildcard_routes_lists_wildcard_model_names(opened: _Rig) -> None:
    """`return_wildcard_routes: true`. `GET /v1/models` must list the wildcard deployment's name
    `dec1-wild/*` next to the concrete model."""
    names: Final = served_model_names(opened.gateway)
    assert {_MODEL, _WILDCARD} <= names, sorted(names)


def test_without_return_wildcard_routes_wildcard_names_are_hidden_unless_the_request_asks(strict: _Rig) -> None:
    """Default: `GET /v1/models` must not list `dec1-wild/*`, and the same call with
    `?return_wildcard_routes=true` must list it: the query parameter overrides the setting per request."""
    assert _WILDCARD not in served_model_names(strict.gateway)
    asked: Final = strict.gateway.get("/v1/models", {"return_wildcard_routes": "true"})["data"]
    assert isinstance(asked, list)
    assert _WILDCARD in {object_value(item)["id"] for item in asked}


def test_allow_public_health_readiness_details_exposes_the_detailed_payload_without_a_key(opened: _Rig) -> None:
    """`allow_public_health_readiness_details: true`. An unauthenticated `GET /health/readiness` must
    return 200 with exactly the detailed key set (status, db, cache, litellm_version, success_callbacks,
    use_aiohttp_transport, log_level, is_detailed_debug, show_no_redis_warning,
    show_env_credential_login_warning)."""
    response: Final = opened.gateway.client.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert frozenset(response.json()) == _DETAILED_READINESS_KEYS


def test_without_allow_public_health_readiness_details_the_public_probe_is_two_fields(strict: _Rig) -> None:
    """Default: an unauthenticated `GET /health/readiness` must return exactly
    `{"status": "healthy", "db": "connected"}`, with no version, cache or callback information."""
    response: Final = strict.gateway.client.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "healthy", "db": "connected"}


def test_alerting_is_reported_by_the_active_settings_endpoint(opened: _Rig) -> None:
    """`alerting: ["slack"]`. `GET /active/callbacks` must report `alerting` as the text `['slack']` and
    `num_alerting` as 1."""
    settings: Final = active_settings(opened.gateway)
    assert settings["alerting"] == "['slack']"
    assert settings["num_alerting"] == 1


def test_without_alerting_the_active_settings_endpoint_reports_none(strict: _Rig) -> None:
    """Default: `GET /active/callbacks` must report `alerting` as the text `None` and `num_alerting` as 0."""
    settings: Final = active_settings(strict.gateway)
    assert settings["alerting"] == "None"
    assert settings["num_alerting"] == 0


def test_enable_drain_endpoint_with_a_token_rejects_a_call_without_the_token(opened: _Rig) -> None:
    """`enable_drain_endpoint: true` with `drain_endpoint_token` set. `GET /health/drain` without
    `X-Drain-Token`, or with a wrong one, must return 401 `Invalid or missing X-Drain-Token` and must not
    start a shutdown: `/health/readiness` still answers 200 afterwards."""
    missing: Final = opened.gateway.client.get("/health/drain")
    wrong: Final = opened.gateway.client.get("/health/drain", headers={"X-Drain-Token": "dec1-wrong"})
    assert (missing.status_code, wrong.status_code) == (401, 401), (missing.text, wrong.text)
    assert wrong.json() == {"detail": "Invalid or missing X-Drain-Token"}
    assert opened.gateway.client.get("/health/readiness").status_code == 200


def test_without_enable_drain_endpoint_the_drain_route_is_a_404(strict: _Rig) -> None:
    """Default: `GET /health/drain` must return 404 `{"detail": "Not Found"}` whatever header is sent, and
    the proxy keeps serving."""
    response: Final = strict.gateway.client.get("/health/drain", headers={"X-Drain-Token": _DRAIN_TOKEN})
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Not Found"}
    assert strict.gateway.client.get("/health/readiness").status_code == 200


def test_store_prompts_in_spend_logs_writes_the_prompt_text_to_the_spend_row(opened: _Rig) -> None:
    """`store_prompts_in_spend_logs: true`. After a chat completion, the `LiteLLM_SpendLogs` row for that
    request must hold the prompt text in its `proxy_server_request` column, the stored request body. The
    `messages` column stays `{}` for chat: it is only filled for realtime calls
    (litellm/proxy/spend_tracking/spend_tracking_utils.py `_get_messages_for_spend_logs_payload`). The exact
    JSON shape of the stored body is not pinned, so the test only looks for the prompt text inside it."""
    marker: Final = f"dec1 stored prompt {uuid.uuid4().hex}"
    body: Final = opened.gateway.chat(_MODEL, text=marker)
    rows: Final = eventually(
        lambda: opened.gateway.rows(
            'SELECT proxy_server_request::text AS stored FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (string_value(body["id"]),),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert marker in string_value(rows[0]["stored"])


def test_enforce_user_param_rejects_an_llm_call_without_a_user(strict: _Rig) -> None:
    """`enforce_user_param: true`. A chat completion whose body has no `user` must be rejected with 401
    and the message `Authentication Error, 'user' param not passed in. 'enforce_user_param'=True`, and
    the provider must not be called."""
    sent(strict.upstream)
    response: Final = chat(strict.gateway, _MODEL)
    assert response.status_code == 401, response.text
    assert error_message(response) == "Authentication Error, 'user' param not passed in. 'enforce_user_param'=True"
    assert sent(strict.upstream) == ()


def test_enforce_user_param_accepts_an_llm_call_that_names_a_user(strict: _Rig) -> None:
    """With `enforce_user_param: true`, the same chat completion with a `user` field must succeed (200)
    and the provider must receive that `user`."""
    sent(strict.upstream)
    response: Final = chat(strict.gateway, _MODEL, user="dec1-named-user")
    assert response.status_code == 200, response.text
    assert [sent_json(request).get("user") for request in sent(strict.upstream)] == ["dec1-named-user"]


def test_enforce_user_param_does_not_apply_to_management_routes(strict: _Rig) -> None:
    """With `enforce_user_param: true`, a `POST /key/generate` (not an LLM route) without a `user` field
    must still succeed: the rule covers POSTs to LLM API routes only."""
    with strict.gateway.scenario() as scenario:
        assert scenario.key().startswith("sk-")


def test_reject_clientside_metadata_tags_rejects_a_body_that_sets_tags(strict: _Rig) -> None:
    """`reject_clientside_metadata_tags: true`. A chat completion whose body has `metadata.tags` must be
    rejected with 400, `error.param` equal to `metadata.tags`, and a message starting `Client-side
    'metadata.tags' not allowed in request. 'reject_clientside_metadata_tags'=True.`; the provider must
    not be called."""
    sent(strict.upstream)
    response: Final = chat(strict.gateway, _MODEL, metadata={"tags": ["dec1-tag"]}, **_strict_chat_fields())
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["param"] == "metadata.tags"
    assert string_value(error["message"]).startswith(
        "Client-side 'metadata.tags' not allowed in request. 'reject_clientside_metadata_tags'=True."
    )
    assert sent(strict.upstream) == ()


def test_without_reject_clientside_metadata_tags_a_body_may_set_tags(opened: _Rig) -> None:
    """Default: a chat completion whose body has `metadata.tags` must succeed with 200."""
    response: Final = chat(opened.gateway, _MODEL, metadata={"tags": ["dec1-tag"]})
    assert response.status_code == 200, response.text


def test_drain_with_the_right_token_takes_the_worker_out_of_rotation(gateway: Gateway, tmp_path: Path) -> None:
    """`enable_drain_endpoint: true` and a token. `GET /health/drain` with the right `X-Drain-Token` must
    return 200 with `status: drained`, and afterwards `/health/readiness` and `/health/liveliness` must
    both answer 503 `{"status": "shutting_down"}` so an orchestrator stops routing to the worker."""
    config: Final = {"general_settings": {"enable_drain_endpoint": True, "drain_endpoint_token": _DRAIN_TOKEN}}
    with booted(gateway, tmp_path, config) as owned:
        drained: Final = owned.gateway.client.get("/health/drain", headers={"X-Drain-Token": _DRAIN_TOKEN})
        readiness: Final = owned.gateway.client.get("/health/readiness")
        liveliness: Final = owned.gateway.client.get("/health/liveliness")
    assert drained.status_code == 200, drained.text
    assert drained.json()["status"] == "drained"
    assert (readiness.status_code, readiness.json()) == (503, {"status": "shutting_down"})
    assert (liveliness.status_code, liveliness.json()) == (503, {"status": "shutting_down"})


def test_drain_token_falls_back_to_the_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """`enable_drain_endpoint: true` with no `drain_endpoint_token` in the file, and `DRAIN_ENDPOINT_TOKEN`
    set in the environment. A drain call without the header must be refused with 401: the environment
    value is the expected token when the file names none."""
    config: Final = {"general_settings": {"enable_drain_endpoint": True}}
    with booted(gateway, tmp_path, config, environment={"DRAIN_ENDPOINT_TOKEN": "dec1-env-drain"}) as owned:
        response: Final = owned.gateway.client.get("/health/drain")
        still_ready: Final = owned.gateway.client.get("/health/readiness")
    assert response.status_code == 401, response.text
    assert still_ready.status_code == 200


def test_completion_model_overrides_the_model_named_in_the_request(gateway: Gateway, tmp_path: Path) -> None:
    """`completion_model: dec1-forced`. A chat completion that asks for `dec1-requested` must be served by
    the `dec1-forced` deployment: the provider must receive `"model": "forced-upstream-model"`, the
    upstream model of the forced deployment, not the requested one's."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [
                deployment("dec1-forced", f"{upstream.url}/v1", model="openai/forced-upstream-model"),
                deployment("dec1-requested", f"{upstream.url}/v1", model="openai/requested-upstream-model"),
            ],
            "general_settings": {"completion_model": "dec1-forced"},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, "dec1-requested")
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [sent_json(request)["model"] for request in requests] == ["forced-upstream-model"]


def test_allowed_ips_without_a_license_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`allowed_ips` is an enterprise key. With no valid license the proxy must exit non-zero and log
    `allowed_ips is an Enterprise Feature. Please add a valid LITELLM_LICENSE to your environment.`
    Assumes an empty `LITELLM_LICENSE` counts as no license."""
    log: Final = refused(
        gateway, tmp_path, {"general_settings": {"allowed_ips": ["127.0.0.1"]}}, environment={"LITELLM_LICENSE": ""}
    )
    assert "allowed_ips is an Enterprise Feature. Please add a valid LITELLM_LICENSE to your environment." in log


def test_enforced_params_without_a_license_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`enforced_params` is an enterprise key. With no valid license the proxy must exit non-zero and log
    `Trying to use `enforced_params`` followed by the enterprise notice. Assumes an empty
    `LITELLM_LICENSE` counts as no license."""
    log: Final = refused(
        gateway, tmp_path, {"general_settings": {"enforced_params": ["user"]}}, environment={"LITELLM_LICENSE": ""}
    )
    assert f"Trying to use `enforced_params`{_NOT_PREMIUM}" in log


def test_custom_auth_loads_a_python_function_from_a_file_next_to_the_config(gateway: Gateway, tmp_path: Path) -> None:
    """`custom_auth: dec1_custom_auth.user_api_key_auth` names a function in `dec1_custom_auth.py` sitting
    beside the config file. A request with the token that function accepts must get 200 on `GET /v1/models`;
    a request with any other token must get 401 whose message carries the function's own error text. A
    Rust front door has to keep calling operator-supplied Python here, resolved relative to the config."""
    (tmp_path / "dec1_custom_auth.py").write_text(
        "from fastapi import Request\n"
        "from litellm.proxy._types import UserAPIKeyAuth\n\n\n"
        "async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:\n"
        '    if api_key == "dec1-custom-token":\n'
        '        return UserAPIKeyAuth(api_key=api_key, user_id="dec1-custom-user")\n'
        '    raise Exception("dec1 custom auth rejected the token")\n'
    )
    config: Final = {"general_settings": {"custom_auth": "dec1_custom_auth.user_api_key_auth"}}
    with booted(gateway, tmp_path, config) as owned:
        accepted: Final = owned.gateway.request("GET", "/v1/models", key="dec1-custom-token")
        rejected: Final = owned.gateway.request("GET", "/v1/models", key="dec1-other-token")
    assert accepted.status_code == 200, accepted.text
    assert rejected.status_code == 401, rejected.text
    assert "dec1 custom auth rejected the token" in error_message(rejected)


def test_custom_auth_naming_a_missing_module_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`custom_auth` names a module that exists neither beside the config nor on the import path. The
    proxy must exit non-zero rather than start without the operator's auth, and the log must name the
    module.
    UNVERIFIED: the exact wording of the import error raised by `get_instance_fn`."""
    log: Final = refused(
        gateway, tmp_path, {"general_settings": {"custom_auth": "dec1_absent_module.user_api_key_auth"}}
    )
    assert "dec1_absent_module" in log


def test_pass_through_endpoints_forwards_a_configured_path_to_its_target_with_the_configured_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`pass_through_endpoints` maps `/dec1-forward` to a target URL and adds a header. An authenticated
    `POST /dec1-forward` must arrive at the target as `POST /dec1-target` carrying
    `x-dec1-injected: from-config` and the caller's JSON body, and the target's reply must come back.
    UNVERIFIED: that a loopback target is allowed by the SSRF settings in the test environment and that
    authenticated pass-through needs no license."""
    with scripted_upstream() as upstream:
        config: Final = {
            "general_settings": {
                "pass_through_endpoints": [
                    {
                        "path": "/dec1-forward",
                        "target": f"{upstream.url}/dec1-target",
                        "headers": {"x-dec1-injected": "from-config"},
                    }
                ]
            }
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = owned.gateway.request("POST", "/dec1-forward", {"dec1": "payload"})
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [(request.method, request.target) for request in requests] == [("POST", "/dec1-target")]
    assert requests[0].headers.get("x-dec1-injected") == "from-config"
    assert sent_json(requests[0]) == {"dec1": "payload"}


def test_a_pass_through_endpoint_requires_a_proxy_key_by_default(gateway: Gateway, tmp_path: Path) -> None:
    """A configured pass-through route with no `auth` field defaults to authenticated. A call without any
    key must be rejected with 401 and must not reach the target."""
    with scripted_upstream() as upstream:
        config: Final = {
            "general_settings": {
                "pass_through_endpoints": [{"path": "/dec1-forward", "target": f"{upstream.url}/dec1-target"}]
            }
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = owned.gateway.client.post("/dec1-forward", json={"dec1": "payload"})
            requests: Final = sent(upstream)
    assert response.status_code == 401, response.text
    assert requests == ()
