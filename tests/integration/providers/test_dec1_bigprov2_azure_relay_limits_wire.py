"""Pre-call checks on the Azure pass-through relay to a router model group.

`/azure/openai/deployments/<model group>/...` and `/azure_ai/<model group>/...` are not forwarded to the
operator's `AZURE_API_BASE`: when a path segment names a router model group, `_relay_azure_router_model`
(`litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py:2575`) calls
`llm_router.allm_passthrough_route` for that group directly. That skips `pass_through_request`, which is where
every other provider route runs the proxy's common pre-call path (the v3 parallel request limiter and the
dynamic rate limiter hooks). Whether a key's limits still apply on this relay was left open by the round 1
pass-through suite (`index/test_dec1_passthru_.md`, "Not covered"), and the Rust front door will rebuild this
dispatch, so these tests pin the intended behaviour: auth level checks (budget) refuse before the relay, and a
key's rpm limit holds on the relay like on chat. UNVERIFIED for the rpm case: if it fails, the relay bypasses
the limiter and that is the bug to file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_azure_support import chat_reply
from integration.providers.dec1_bigprov2_support import error_of, marker

_API_VERSION: Final = "2024-10-21"


def _respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    return chat_reply("relayed")


@contextmanager
def _group(gateway: Gateway) -> Iterator[tuple[Wire, str]]:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/dec1-relay-deployment",
            api_base=wire.url,
            api_key="dec1-relay-key",
            api_version=_API_VERSION,
        )
        yield wire, model


def _relay(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        f"/azure/openai/deployments/{model}/chat/completions",
        {"messages": [{"role": "user", "content": marker()}]},
        key=key,
        params={"api-version": _API_VERSION},
    )


def _posts(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if request.method == "POST")


def test_a_zero_budget_key_is_refused_before_the_azure_relay(gateway: Gateway) -> None:
    """A key with `max_budget: 0` calls the relay for a router group. The answer is 422 with `error.type`
    `budget_exceeded`, decided at auth before the relay runs, and the deployment's peer receives nothing."""
    with _group(gateway) as (wire, model):
        key: Final = gateway.post("/key/generate", {"max_budget": 0})["key"]
        try:
            response: Final = _relay(gateway, model, str(key))
            assert response.status_code == 422, response.text
            assert error_of(response)["type"] == "budget_exceeded", response.text
            assert _posts(wire) == 0
        finally:
            gateway.post("/key/delete", {"keys": [key]})


def test_a_keys_rpm_limit_holds_on_the_azure_relay(gateway: Gateway) -> None:
    """A key with `rpm_limit: 1` calls the relay twice. The first is answered 200 with the deployment's
    `relayed` content and the second must be 429 `throttling_error`, with the peer seeing exactly one request.
    UNVERIFIED: the relay calls the router directly and skips `pass_through_request`'s pre-call hook."""
    with _group(gateway) as (wire, model), gateway.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        first: Final = _relay(gateway, model, key)
        assert first.status_code == 200, first.text
        assert "relayed" in first.text
        second: Final = _relay(gateway, model, key)
        assert second.status_code == 429, second.text
        assert error_of(second)["type"] == "throttling_error", second.text
        assert _posts(wire) == 1


def test_one_rpm_window_is_shared_between_chat_and_the_azure_relay(gateway: Gateway) -> None:
    """A key with `rpm_limit: 1` makes one `/v1/chat/completions` call to the group and then one relay call.
    The relay call must be 429, because both routes draw on the key's one request window. UNVERIFIED, same
    cause as above."""
    with _group(gateway) as (wire, model), gateway.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        chat: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker()}]},
            key=key,
        )
        assert chat.status_code == 200, chat.text
        relayed: Final = _relay(gateway, model, key)
        assert relayed.status_code == 429, relayed.text
        assert _posts(wire) == 1


def test_a_key_limited_to_another_group_is_refused_on_the_relay(gateway: Gateway) -> None:
    """A key with `models: ["dec1-only-this-model"]` calls the relay for the group. It is refused 403
    `key_model_access_denied` (the group is read from the deployments segment by `_router_model_from_azure_route`)
    and the peer receives nothing, even though the relay itself bypasses the pre-call path."""
    with _group(gateway) as (wire, model), gateway.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-only-this-model"])
        response: Final = _relay(gateway, model, key)
        assert response.status_code == 403, response.text
        assert error_of(response)["type"] == "key_model_access_denied", response.text
        assert _posts(wire) == 0
