"""Callers cannot point the proxy at their own hosts or secrets unless an admin opted in (SSRF and key exfiltration).

Contract (litellm/proxy/auth/auth_utils.py `pre_db_read_auth_checks`, `is_request_body_safe`,
`_check_banned_params`, `_banned_param_error`, `_reject_secret_references`, `_is_param_allowed`):

* Before any key lookup, the request body is checked. `api_base`, `base_url`, `user_config`, provider identity
  selectors (AWS, Vertex, Azure), storage endpoints and observability hosts such as `langfuse_host` are refused,
  at the top level, inside `extra_body` and `litellm_embedding_config`, inside `metadata`, and inside dict
  fallback targets. The refusal is 401 `auth_error` "Authentication Error, Rejected Request: <param> is not
  allowed in request body. ..." and no provider is called. A `os.environ/...` value in a credential field is
  refused as a secret reference. The master key is checked the same way: being an admin does not open it.
* Opt ins are explicit: `general_settings.allow_client_side_credentials` proxy wide, or
  `configurable_clientside_auth_params` on every deployment of the model group, either the plain name
  `api_base` (any value) or `{"api_base": "<regex>"}` (matching values only). An opt in on one model does not
  open another.

A recording peer stands in for an attacker's host (`wire_server`); a refused request must leave it untouched.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Wire, wire_server
from integration.security.dec1_authsec_support import (
    CHAT,
    DEPLOYMENT_KEY,
    REJECTED_PARAM_TEMPLATE,
    assert_auth_error,
    assert_served,
    chat_body,
    marker,
    posts,
    provider_reply,
)
from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class Targets:
    gateway: Gateway
    scenario: Scenario
    provider: Wire
    attacker: Wire
    model: str


@contextmanager
def targets(gateway: Gateway, **parameters: JsonValue) -> Iterator[Targets]:
    """One chat model served by `provider`, plus a second peer playing a host the caller would like to reach."""
    with (
        wire_server(provider_reply) as provider,
        wire_server(provider_reply) as attacker,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key=DEPLOYMENT_KEY, **parameters)
        yield Targets(gateway, scenario, provider, attacker, model)


def assert_rejected(response_status: int, message: str, param: str) -> None:
    assert response_status == 401, message
    assert message.startswith(f"Authentication Error, {REJECTED_PARAM_TEMPLATE.format(param=param)}"), message


def assert_untouched(rig: Targets) -> None:
    assert posts(rig.provider) == () and posts(rig.attacker) == ()


@pytest.mark.parametrize(
    "param",
    [
        "base_url",
        "user_config",
        "aws_sts_endpoint",
        "azure_ad_token",
        "s3_endpoint_url",
        "langfuse_host",
    ],
)
def test_a_retargeting_param_at_the_top_level_is_refused(gateway: Gateway, param: str) -> None:
    """A virtual key sends a chat call whose body also carries one retargeting or identity field pointing at the
    attacker peer. The proxy must answer 401 `auth_error` whose message starts
    `Authentication Error, Rejected Request: <param> is not allowed in request body.`, and neither peer may
    receive a request. `user_config` gets a dict value, the others the attacker URL or a synthetic string."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        value: Final[JsonValue] = {"model_list": []} if param == "user_config" else f"{rig.attacker.url}/v1"
        refused: Final = gateway.request("POST", CHAT, chat_body(rig.model, **{param: value}), key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), param)
        assert_untouched(rig)


def test_a_vertex_target_param_on_a_model_group_that_never_reads_it_is_served(gateway: Gateway) -> None:
    """A virtual key sends a chat call to the OpenAI style deployment with `vertex_project` set to the attacker
    URL. The Vertex target params are only checked against model groups whose deployments read them
    (`VERTEX_TARGET_PARAMS`, `_model_group_reads_vertex_target`, litellm/proxy/auth/auth_utils.py:277-282, 717 and
    834-845), and an OpenAI deployment never does, so the call is served with 200 by the configured provider and
    the attacker peer receives nothing."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        served: Final = gateway.request(
            "POST", CHAT, chat_body(rig.model, vertex_project=f"{rig.attacker.url}/v1"), key=key
        )
        assert served.status_code == 200, served.text
        assert len(posts(rig.provider)) == 1
        assert posts(rig.attacker) == ()


def test_the_master_key_cannot_send_its_own_api_base_either(gateway: Gateway) -> None:
    """The master key sends a chat call with `api_base` set to the attacker peer. It must be refused exactly like
    a virtual key (401, `Rejected Request: api_base is not allowed in request body.`), because the check runs in
    `pre_db_read_auth_checks` before the caller is identified. A Rust port that checks the body only for virtual
    keys would let a stolen master key read deployment keys out of the proxy."""
    with targets(gateway) as rig:
        refused: Final = gateway.request("POST", CHAT, chat_body(rig.model, api_base=f"{rig.attacker.url}/v1"))
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert_untouched(rig)


def test_an_api_base_inside_extra_body_is_refused(gateway: Gateway) -> None:
    """A virtual key sends `extra_body: {"api_base": <attacker>}`. The body check descends into `extra_body`
    (`_NESTED_CONFIG_KEYS`), so the answer must be the same 401 naming `api_base`, and no peer is called."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        body: Final = chat_body(rig.model, extra_body={"api_base": f"{rig.attacker.url}/v1"})
        refused: Final = gateway.request("POST", CHAT, body, key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert_untouched(rig)


def test_a_langfuse_host_inside_metadata_is_refused(gateway: Gateway) -> None:
    """A virtual key sends `metadata: {"langfuse_host": <attacker>}`, which would send the proxy's Langfuse
    credentials to that host. The answer must be 401 naming `langfuse_host` (`_NESTED_METADATA_KEYS` with
    `_BANNED_METADATA_PARAMS`), and no peer is called."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        body: Final = chat_body(rig.model, metadata={"langfuse_host": rig.attacker.url})
        refused: Final = gateway.request("POST", CHAT, body, key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "langfuse_host")
        assert_untouched(rig)


def test_an_api_base_inside_a_fallback_target_is_refused(gateway: Gateway) -> None:
    """A virtual key sends `fallbacks: [{"model": <model>, "api_base": <attacker>}]`. Dict fallback targets are
    checked like the top level (`iter_request_fallback_targets`), so the answer must be 401 naming `api_base` and
    no peer is called."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        body: Final = chat_body(rig.model, fallbacks=[{"model": rig.model, "api_base": f"{rig.attacker.url}/v1"}])
        refused: Final = gateway.request("POST", CHAT, body, key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert_untouched(rig)


def test_an_api_base_on_embeddings_is_refused(gateway: Gateway) -> None:
    """A virtual key calls `POST /v1/embeddings` for a registered embedding model with `api_base` set to the
    attacker peer. The answer must be 401 naming `api_base`: the body check is route independent."""
    with (
        wire_server(provider_reply) as provider,
        wire_server(provider_reply) as attacker,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/text-embedding-3-small", api_base=f"{provider.url}/v1", api_key=DEPLOYMENT_KEY
        )
        key: Final = scenario.key(models=[model])
        refused: Final = gateway.request(
            "POST", "/v1/embeddings", {"model": model, "input": "x", "api_base": f"{attacker.url}/v1"}, key=key
        )
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert posts(provider) == () and posts(attacker) == ()


def test_an_api_base_on_the_messages_route_is_refused(gateway: Gateway) -> None:
    """A virtual key calls `POST /v1/messages` with `api_base` set to the attacker peer. The answer must be 401
    naming `api_base`; `/v1/messages` is one of the first routes moving to Rust and must keep the body check.
    UNVERIFIED: the Anthropic route returns the same OpenAI shaped auth error body."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        body: Final = {
            "model": rig.model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": marker()}],
            "api_base": rig.attacker.url,
        }
        refused: Final = gateway.request("POST", "/v1/messages", body, key=key)
        assert refused.status_code == 401, refused.text
        assert REJECTED_PARAM_TEMPLATE.format(param="api_base") in refused.text
        assert_untouched(rig)


def test_an_environment_reference_as_the_api_key_is_refused(gateway: Gateway) -> None:
    """A virtual key sends `api_key: "os.environ/LITELLM_MASTER_KEY"`, which would make the proxy read its own
    secret for a call the caller shapes. The answer must be 401 whose message contains
    `Rejected Request: api_key holds a secret reference`, and no peer may be called."""
    with targets(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        refused: Final = gateway.request(
            "POST", CHAT, chat_body(rig.model, api_key="os.environ/LITELLM_MASTER_KEY"), key=key
        )
        assert refused.status_code == 401, refused.text
        assert "Rejected Request: api_key holds a secret reference" in refused.text
        assert_untouched(rig)


def test_the_body_check_runs_before_the_key_is_looked_up(gateway: Gateway) -> None:
    """An unknown `sk-` key sends a chat call with `api_base`. The answer must be the `Rejected Request: api_base`
    401, not `Invalid API key provided.`: `pre_db_read_auth_checks` runs before the key read. Pinned because the
    Rust core reorders checks (#51374 "A few denials now win over others")."""
    with targets(gateway) as rig:
        refused: Final = gateway.request(
            "POST", CHAT, chat_body(rig.model, api_base=f"{rig.attacker.url}/v1"), key=f"sk-{marker()}"
        )
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")


def test_a_regex_opt_in_admits_a_matching_api_base_and_sends_the_call_there(gateway: Gateway) -> None:
    """The model is registered with `configurable_clientside_auth_params: [{"api_base": "<attacker url>.*"}]`
    (the attacker peer stands in for an approved host). A virtual key sends `api_base` and `api_key` naming that
    host. The call must be served with 200, the approved peer must receive exactly one chat request with the
    caller's `Authorization: Bearer <client key>`, and the configured provider none."""
    with (
        wire_server(provider_reply) as provider,
        wire_server(provider_reply) as approved,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            api_base=f"{provider.url}/v1",
            api_key=DEPLOYMENT_KEY,
            configurable_clientside_auth_params=[{"api_base": f"^{re.escape(approved.url)}.*"}],
        )
        key: Final = scenario.key(models=[model])
        served: Final = gateway.request(
            "POST", CHAT, chat_body(model, api_base=f"{approved.url}/v1", api_key="sk-dec1-client-side"), key=key
        )
        assert_served(served)
        calls: Final = posts(approved)
        assert [call.headers.get("authorization") for call in calls] == ["Bearer sk-dec1-client-side"]
        assert posts(provider) == ()


def test_a_regex_opt_in_refuses_an_api_base_that_does_not_match(gateway: Gateway) -> None:
    """Same model and opt in as above; the caller sends `api_base` naming a different host
    (`http://169.254.169.254/latest`, the cloud metadata address). The answer must be 401 naming `api_base` and no
    peer may be called."""
    with (
        wire_server(provider_reply) as provider,
        wire_server(provider_reply) as approved,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            api_base=f"{provider.url}/v1",
            api_key=DEPLOYMENT_KEY,
            configurable_clientside_auth_params=[{"api_base": f"^{re.escape(approved.url)}.*"}],
        )
        key: Final = scenario.key(models=[model])
        refused: Final = gateway.request(
            "POST", CHAT, chat_body(model, api_base="http://169.254.169.254/latest"), key=key
        )
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert posts(provider) == () and posts(approved) == ()


def test_an_opt_in_on_one_model_does_not_open_another(gateway: Gateway) -> None:
    """Model A is registered with `configurable_clientside_auth_params: ["api_base"]` (any value) and model B
    without it. A key allowed both sends `api_base` on a call to model B. The answer must be 401 naming
    `api_base`; the opt in is per model group (`_allow_model_level_clientside_configurable_parameters`)."""
    with targets(gateway) as rig:
        opened: Final = rig.scenario.model(
            api_base=f"{rig.provider.url}/v1", api_key=DEPLOYMENT_KEY, configurable_clientside_auth_params=["api_base"]
        )
        key: Final = rig.scenario.key(models=[rig.model, opened])
        refused: Final = gateway.request("POST", CHAT, chat_body(rig.model, api_base=f"{rig.attacker.url}/v1"), key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "api_base")
        assert_untouched(rig)


def test_a_named_api_base_opt_in_does_not_open_base_url(gateway: Gateway) -> None:
    """The model is registered with `configurable_clientside_auth_params: ["api_base"]`. The caller sends
    `base_url` instead. The answer must be 401 naming `base_url`: each banned param needs its own opt in."""
    with targets(gateway, configurable_clientside_auth_params=["api_base"]) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        refused: Final = gateway.request("POST", CHAT, chat_body(rig.model, base_url=f"{rig.attacker.url}/v1"), key=key)
        assert_rejected(refused.status_code, assert_auth_error(refused, 401), "base_url")
        assert_untouched(rig)
