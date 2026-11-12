from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
)
from tests.integration._support.openai_wire import chat_reply
from tests.integration._support.process import (
    graceful_stop_seconds,
    owned_proxy_process,
)
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.routing.test_per_model_rate_limit_fallbacks import (
    model_entry,
    proxy_config,
)

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def _reply(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert not any(
        field in body
        for field in (
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "router_settings_override",
        )
    ), body
    if request.target.startswith("/primary/"):
        messages: Final = json.dumps(body["messages"])
        for marker, status, code in (
            ("provider-failure", 500, "server_error"),
            ("context-window", 400, "context_length_exceeded"),
            ("content-policy", 400, "content_policy_violation"),
        ):
            if marker in messages:
                return Reply(
                    status=status,
                    body=json.dumps(
                        {
                            "error": {
                                "message": marker,
                                "type": ("invalid_request_error" if status == 400 else "server_error"),
                                "code": code,
                            }
                        }
                    ).encode(),
                )
    return chat_reply("chatcmpl-" + uuid.uuid4().hex, "gpt-4o-mini", "synthetic reply", stream=False)


@dataclass(frozen=True, slots=True)
class CustomerFallbackRig:
    gateway: Gateway
    wire: Wire
    primary: str
    first: str
    second: str
    global_backup: str

    @property
    def models(self) -> list[JsonValue]:
        return [self.primary, self.first, self.second, self.global_backup]

    def chain(self, target: str) -> list[JsonValue]:
        return [{self.primary: [target]}]

    def customer(self, scenario: Scenario, **fields: object) -> str:
        customer: Final = "integration-customer-fallback-" + uuid.uuid4().hex
        created: Final = self.gateway.post(
            "/customer/new",
            JSON_OBJECT.validate_python({"user_id": customer, **fields}),
        )
        budget_id: Final = created.get("budget_id")
        if isinstance(budget_id, str):
            scenario.cleanups.callback(scenario.delete_budget, budget_id)
        scenario.cleanups.callback(self.gateway.post, "/customer/delete", {"user_ids": [customer]})
        return customer

    def chat(self, key: str, customer: str | None, marker: str = "healthy", **extra: object) -> httpx.Response:
        return self.gateway.request(
            "POST",
            "/v1/chat/completions",
            JSON_OBJECT.validate_python(
                {
                    "model": self.primary,
                    "messages": [{"role": "user", "content": marker + " " + uuid.uuid4().hex}],
                    "max_tokens": 8,
                    **extra,
                }
            ),
            key=key,
            headers={
                "Connection": "close",
                **({"x-litellm-customer-id": customer} if customer is not None else {}),
            },
        )


@pytest.fixture(scope="module")
def customer_fallbacks(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CustomerFallbackRig]:
    directory: Final = tmp_path_factory.mktemp("customer-fallbacks")
    suffix: Final = uuid.uuid4().hex[:10]
    names: Final = tuple(
        "customer-fallback-" + label + "-" + suffix for label in ("primary", "first", "second", "global")
    )
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(_reply))
        config: Final = proxy_config(
            directory,
            model_list=tuple(
                model_entry(name, wire, api_base=f"{wire.url}/{label}/v1")
                for name, label in zip(names, ("primary", "first", "second", "global"), strict=True)
            ),
            fallbacks=({names[0]: [names[3]]},),
            general_settings={
                "disable_fallbacks_on_per_model_rate_limits": False,
                "allow_client_side_credentials": True,
            },
        )
        configuration: Final = JSON_OBJECT.validate_python(yaml.safe_load(config.read_text()))
        config.write_text(
            yaml.safe_dump(
                {
                    **configuration,
                    "router_settings": {
                        **object_value(configuration["router_settings"]),
                        "default_fallbacks": [names[2]],
                    },
                }
            )
        )
        owned: Final = stack.enter_context(owned_proxy_process(gateway, directory, {}, config=config, workers=2))
        yield CustomerFallbackRig(owned.gateway, wire, names[0], names[1], names[2], names[3])


def _assert_served(response: httpx.Response, group: str) -> None:
    assert (response.status_code, response.headers.get("x-litellm-model-group")) == (
        200,
        group,
    ), response.text


@pytest.mark.parametrize("failure", ("provider-failure", "context-window", "content-policy"))
def test_customer_chain_overrides_client_and_key_fallbacks_without_leaking_between_customers(
    customer_fallbacks: CustomerFallbackRig, failure: str
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        first: Final = rig.customer(scenario, fallbacks=rig.chain(rig.first))
        second: Final = rig.customer(scenario, fallbacks=rig.chain(rig.second))
        key: Final = scenario.key(
            models=rig.models,
            router_settings={"fallbacks": rig.chain(rig.global_backup)},
        )
        overrides: Final[dict[str, JsonValue]] = {
            "fallbacks": rig.chain(rig.global_backup),
            "context_window_fallbacks": rig.chain(rig.global_backup),
            "content_policy_fallbacks": rig.chain(rig.global_backup),
            "router_settings_override": {"fallbacks": rig.chain(rig.global_backup)},
        }
        for customer, expected in (
            (first, rig.first),
            (second, rig.second),
            ("unknown-" + uuid.uuid4().hex, rig.global_backup),
            (None, rig.global_backup),
            (first, rig.first),
        ):
            _assert_served(rig.chat(key, customer, failure, **overrides), expected)


def test_customer_model_allowance_and_fallback_are_shared_across_three_service_keys(
    customer_fallbacks: CustomerFallbackRig,
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        customer: Final = rig.customer(
            scenario,
            fallbacks=rig.chain(rig.first),
            tpm_limit=100000,
            rpm_limit=100,
            model_tpm_limit={rig.primary: 10000, rig.first: 10000},
            model_rpm_limit={rig.primary: 1, rig.first: 10},
        )
        keys: Final = tuple(scenario.key(models=rig.models) for _ in range(3))
        for key, expected in zip(keys, (rig.primary, rig.first, rig.first), strict=True):
            response = rig.chat(key, customer, fallbacks=rig.chain(rig.second))
            _assert_served(response, expected)
            if expected == rig.first:
                assert response.headers.get("x-litellm-fallback-reason") == "rate_limit:model_per_end_user", (
                    response.text
                )


def test_customer_fallback_cannot_bypass_target_or_overall_limits(
    customer_fallbacks: CustomerFallbackRig,
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=rig.models)
        target_limited: Final = rig.customer(
            scenario,
            fallbacks=rig.chain(rig.first),
            tpm_limit=10000,
            rpm_limit=100,
            model_tpm_limit={rig.primary: 1, rig.first: 1},
        )
        target_refused: Final = rig.chat(key, target_limited, fallbacks=rig.chain(rig.second))
        assert (
            target_refused.status_code,
            target_refused.headers.get("rate_limited_by"),
        ) == (
            429,
            "model_per_end_user",
        ), target_refused.text

        overall_limited: Final = rig.customer(scenario, fallbacks=rig.chain(rig.first), rpm_limit=1)
        _assert_served(rig.chat(key, overall_limited), rig.primary)
        overall_refused: Final = rig.chat(key, overall_limited, "provider-failure")
        assert (
            overall_refused.status_code,
            overall_refused.headers.get("rate_limited_by"),
        ) == (
            429,
            "end_user",
        ), overall_refused.text

        token_limited: Final = rig.customer(scenario, fallbacks=rig.chain(rig.first), tpm_limit=1)
        tokens_refused: Final = rig.chat(key, token_limited, "provider-failure")
        assert (
            tokens_refused.status_code,
            tokens_refused.headers.get("rate_limited_by"),
        ) == (
            429,
            "end_user",
        ), tokens_refused.text


@pytest.mark.parametrize("failure", ("healthy", "provider-failure"))
def test_customer_fallback_respects_key_and_customer_model_access(
    customer_fallbacks: CustomerFallbackRig, failure: str
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        cases: Final[tuple[tuple[list[JsonValue], list[JsonValue]], ...]] = (
            ([rig.primary], rig.models),
            (rig.models, [rig.primary]),
        )
        for models, key_models in cases:
            customer = rig.customer(
                scenario,
                fallbacks=rig.chain(rig.first),
                models=models,
                **({"model_tpm_limit": {rig.primary: 1}} if failure == "healthy" else {}),
            )
            key = scenario.key(models=key_models)
            rig.wire.drain()
            response = rig.chat(key, customer, failure)
            assert response.status_code == (429 if failure == "healthy" else 500), response.text
            assert not tuple(request for request in rig.wire.drain() if request.target.startswith("/first/"))


@pytest.mark.parametrize("failure", ("provider-failure", "context-window", "content-policy"))
def test_customer_fallback_update_preserves_disables_and_clears_the_policy(
    customer_fallbacks: CustomerFallbackRig, failure: str
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=rig.models)
        customer: Final = rig.customer(scenario, fallbacks=rig.chain(rig.first))
        renamed: Final = rig.gateway.post("/customer/update", {"user_id": customer, "alias": "renamed"})
        assert renamed["fallbacks"] == rig.chain(rig.first)
        _assert_served(rig.chat(key, customer, failure), rig.first)

        disabled: Final = rig.gateway.post("/customer/update", {"user_id": customer, "fallbacks": []})
        assert disabled["fallbacks"] == []
        assert rig.gateway.get("/customer/info", {"end_user_id": customer})["fallbacks"] == []
        refused: Final = rig.chat(
            key,
            customer,
            failure,
            fallbacks=rig.chain(rig.second),
            context_window_fallbacks=rig.chain(rig.second),
            content_policy_fallbacks=rig.chain(rig.second),
            router_settings_override={"fallbacks": rig.chain(rig.second)},
        )
        assert refused.status_code == (500 if failure == "provider-failure" else 400), refused.text

        cleared: Final = rig.gateway.post("/customer/update", {"user_id": customer, "fallbacks": None})
        assert cleared["fallbacks"] is None
        assert rig.gateway.get("/customer/info", {"end_user_id": customer})["fallbacks"] is None
        _assert_served(rig.chat(key, customer, failure), rig.global_backup)


def test_configured_customer_rejects_a_client_router_without_fallback_target_guards(
    customer_fallbacks: CustomerFallbackRig,
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        customer: Final = rig.customer(scenario, fallbacks=rig.chain(rig.first))
        response: Final = rig.chat(rig.gateway.key, customer, user_config={})
        assert response.status_code == 400, response.text


def test_customer_policy_without_the_requested_primary_does_not_use_the_global_default(
    customer_fallbacks: CustomerFallbackRig,
) -> None:
    rig: Final = customer_fallbacks
    with rig.gateway.scenario() as scenario:
        customer: Final = rig.customer(scenario, fallbacks=[{rig.first: [rig.second]}])
        key: Final = scenario.key(models=rig.models)
        rig.wire.drain()
        response: Final = rig.chat(key, customer, "provider-failure")
        assert response.status_code == 500, response.text
        assert not tuple(
            request
            for request in rig.wire.drain()
            if request.method == "POST" and not request.target.startswith("/primary/")
        )
