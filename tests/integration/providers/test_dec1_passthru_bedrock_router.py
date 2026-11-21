"""Bedrock pass-through to a router model group: `/bedrock/model/<group>/<action>` under keys, limits and spend.

When the model segment of a `/bedrock` path names a router model group, the call does not go to AWS directly: it
goes through the proxy's common request path with `route_type = allm_passthrough_route`
(`handle_bedrock_passthrough_router_model`, `common_request_processing.py:4431`). The router picks a deployment,
replaces the group name in the path with that deployment's Bedrock model id, signs with the deployment's AWS
credentials and sends the caller's native Converse or Invoke body. Because it runs the common path, the
group-level machinery applies: key and team model allowlists, per-model rate limits, load balancing, budgets, and
a spend row with `call_type = allm_passthrough_route` and `model_group = <group>` priced from the native usage at
the deployment's price.

The existing `authorization/test_bedrock_passthrough_model_access.py` pins the key model allowlist for the four
actions and `providers/test_bedrock_passthrough_aws_errors.py` pins key budget and key rpm refusals on Converse;
this file covers the rest. Every test uses the shared `gateway` and registers its own group, so the deployment's
`api_base` is the owned peer.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import wire_server
from integration.authorization.dec1_limits_support import hashed, limit_message
from integration.providers.dec1_passthru_support import (
    BEDROCK_CONVERSE_BODY,
    BEDROCK_INPUT_TOKENS,
    BEDROCK_MODEL_ID,
    BEDROCK_OUTPUT_TOKENS,
    BEDROCK_STREAM_BYTES,
    ROUTER_PASS_THROUGH_CALL_TYPE,
    SPEND_SECONDS,
    assert_key_absent,
    bedrock_deployment,
    bedrock_peer,
    key_spend,
    number,
    openai_error,
    path_of,
    posts,
    refused_until,
    spend_row,
)

_INPUT_RATE: Final = 0.001
_OUTPUT_RATE: Final = 0.002
_CALL_COST: Final = BEDROCK_INPUT_TOKENS * _INPUT_RATE + BEDROCK_OUTPUT_TOKENS * _OUTPUT_RATE


def _converse(gateway: Gateway, group: str, key: str, action: str = "converse", **headers: str):  # noqa: ANN202
    return gateway.request("POST", f"/bedrock/model/{group}/{action}", BEDROCK_CONVERSE_BODY, key=key, headers=headers)


def test_a_converse_call_writes_a_router_pass_through_row_priced_at_the_deployment_price(gateway: Gateway) -> None:
    """A group with one Bedrock deployment priced 0.001 in and 0.002 out is called with
    `/bedrock/model/<group>/converse`; the reply reports 11 input and 4 output tokens. One row must appear for the
    key with `call_type` `allm_passthrough_route`, `model_group` `<group>`, `custom_llm_provider` `bedrock`,
    tokens 11 and 4, `status` `success` and spend 11 * 0.001 + 4 * 0.002 = 0.019."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(
            **bedrock_deployment(wire, input_cost_per_token=_INPUT_RATE, output_cost_per_token=_OUTPUT_RATE)
        )
        key: Final = scenario.key()
        assert _converse(gateway, group, key).status_code == 200
        row: Final = spend_row(gateway, key)
        assert {
            name: row[name]
            for name in (
                "call_type",
                "model_group",
                "custom_llm_provider",
                "prompt_tokens",
                "completion_tokens",
                "status",
            )
        } == {
            "call_type": ROUTER_PASS_THROUGH_CALL_TYPE,
            "model_group": group,
            "custom_llm_provider": "bedrock",
            "prompt_tokens": BEDROCK_INPUT_TOKENS,
            "completion_tokens": BEDROCK_OUTPUT_TOKENS,
            "status": "success",
        }, row
        assert number(row["spend"]) == pytest.approx(_CALL_COST)


def test_a_converse_stream_is_relayed_byte_for_byte_and_billed_from_its_metadata_frame(gateway: Gateway) -> None:
    """`/bedrock/model/<group>/converse-stream` must answer with exactly the AWS event-stream bytes the deployment
    sent, and the key's row must record the 11 and 4 tokens carried by the final `metadata` frame at the
    deployment price."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(
            **bedrock_deployment(wire, input_cost_per_token=_INPUT_RATE, output_cost_per_token=_OUTPUT_RATE)
        )
        key: Final = scenario.key()
        response: Final = _converse(gateway, group, key, "converse-stream")
        assert response.status_code == 200, response.text
        assert response.content == BEDROCK_STREAM_BYTES
        row: Final = spend_row(gateway, key)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (BEDROCK_INPUT_TOKENS, BEDROCK_OUTPUT_TOKENS), row
        assert number(row["spend"]) == pytest.approx(_CALL_COST)


def test_the_group_name_in_the_path_is_replaced_by_the_deployment_model_id_and_signed(gateway: Gateway) -> None:
    """A call to `/bedrock/model/<group>/converse` must reach the peer as `POST /model/<deployment model
    id>/converse`, signed with SigV4 for the deployment's access key (`authorization` starting
    `AWS4-HMAC-SHA256 Credential=AKIADEC1PASSTHRU/`), with the caller's virtual key nowhere in the request."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        key: Final = scenario.key()
        assert _converse(gateway, group, key).status_code == 200
        received: Final = posts(wire)
        assert [path_of(request) for request in received] == [f"/model/{BEDROCK_MODEL_ID}/converse"], received
        assert received[0].headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIADEC1PASSTHRU/")
        assert_key_absent(received[0], key)


def test_ordinary_caller_headers_stay_at_the_proxy_and_x_pass_headers_are_forwarded(gateway: Gateway) -> None:
    """The caller sends `x-dec1-trace: t1` and `x-pass-x-dec1-route: blue`. Bedrock must receive `x-dec1-route:
    blue` and no `x-dec1-trace`: the router pass-through forwards only `x-pass-*` headers, minus credentials
    (`_bedrock_pass_through_headers`). UNVERIFIED: that the router hop sends `request_headers` on to the deployment
    and not only into the signature."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        key: Final = scenario.key()
        response: Final = gateway.request(
            "POST",
            f"/bedrock/model/{group}/converse",
            BEDROCK_CONVERSE_BODY,
            key=key,
            headers={"x-dec1-trace": "t1", "x-pass-x-dec1-route": "blue"},
        )
        assert response.status_code == 200, response.text
        upstream: Final = posts(wire)[0]
        assert upstream.headers.get("x-dec1-route") == "blue", upstream.headers
        assert "x-dec1-trace" not in upstream.headers, upstream.headers


def test_a_per_model_rpm_limit_holds_a_router_group_reached_through_bedrock_pass_through(gateway: Gateway) -> None:
    """A key has `model_rpm_limit: {<group>: 1}`. Two `/bedrock/model/<group>/converse` calls: the second must be
    refused with 429 for `model_per_key: <key hash>:<group>`, `Limit type: requests`, and the deployment must have
    received one call. Unlike operator-configured pass-through, this route resolves to a real group."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        key: Final = scenario.key(model_rpm_limit={group: 1})
        assert _converse(gateway, group, key).status_code == 200
        refused: Final = _converse(gateway, group, key)
        error: Final = openai_error(refused, 429)
        assert str(error["message"]).startswith(
            limit_message("model_per_key", f"{hashed(key)}:{group}", "requests", 1)
        ), refused.text
        assert len(posts(wire)) == 1


def test_a_team_without_the_group_refuses_its_key_on_bedrock_pass_through(gateway: Gateway) -> None:
    """A team lists only another group and its key has no model list. A `/bedrock/model/<group>/converse` call must
    be refused with 403 `team_model_access_denied` and reach no deployment."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        team: Final = scenario.team(models=[f"dec1-other-{uuid.uuid4().hex[:8]}"])
        key: Final = scenario.key(team_id=team)
        response: Final = _converse(gateway, group, key)
        assert openai_error(response, 403)["type"] == "team_model_access_denied", response.text
        assert posts(wire) == ()


def test_a_raw_bedrock_model_id_is_checked_against_a_restricted_keys_models(gateway: Gateway) -> None:
    """A key with `models: [<group>]` calls `/bedrock/model/<raw Bedrock model id>/converse`, an id that is no
    router group. It must be refused with 403 `key_model_access_denied` naming the raw id
    (`_model_from_bedrock_route`), so a key cannot bypass its allowlist by naming the provider model directly. The
    refusal happens before any AWS call, so no network is used."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        key: Final = scenario.key(models=[group])
        response: Final = gateway.request(
            "POST", f"/bedrock/model/{BEDROCK_MODEL_ID}/converse", BEDROCK_CONVERSE_BODY, key=key
        )
        error: Final = openai_error(response, 403)
        assert error["type"] == "key_model_access_denied", response.text
        assert BEDROCK_MODEL_ID in str(error["message"]), response.text
        assert posts(wire) == ()


def test_a_bedrock_path_that_names_no_model_is_a_400(gateway: Gateway) -> None:
    """`POST /bedrock/not-a-model-path` has neither a `model` segment nor a known action. It must be refused with
    400 whose body contains `'model' keyword not found and unable to extract model from endpoint. Expected format:
    /model/{modelId}/{action}. Got: not-a-model-path` (`extract_model_from_bedrock_endpoint`). UNVERIFIED: the exact
    envelope the proxy wraps the `{"error": ...}` detail in."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = gateway.request("POST", "/bedrock/not-a-model-path", BEDROCK_CONVERSE_BODY, key=key)
        assert response.status_code == 400, response.text
        assert (
            "'model' keyword not found and unable to extract model from endpoint. Expected format: "
            "/model/{modelId}/{action}. Got: not-a-model-path"
        ) in response.text


def test_bedrock_pass_through_load_balances_across_the_groups_deployments(gateway: Gateway) -> None:
    """A group has two deployments that call two different peers. Twelve `/bedrock/model/<group>/converse` calls
    must all be served and both peers must receive at least one: the router, not the path, picks the deployment.
    The chance that a fair pick sends all twelve to one peer is 1 in 2048."""
    with (
        wire_server(bedrock_peer) as first,
        wire_server(bedrock_peer) as second,
        gateway.scenario() as scenario,
    ):
        group: Final = f"dec1-passthru-lb-{uuid.uuid4().hex[:10]}"
        scenario.model(name=group, **bedrock_deployment(first))
        scenario.model(name=group, **bedrock_deployment(second))
        key: Final = scenario.key()
        statuses: Final = [_converse(gateway, group, key).status_code for _ in range(12)]
        assert statuses == [200] * 12
        counts: Final = (len(posts(first)), len(posts(second)))
        assert sum(counts) == 12 and min(counts) >= 1, counts


def test_a_key_budget_closes_after_bedrock_router_pass_through_spend(gateway: Gateway) -> None:
    """A key has `max_budget: 0.01`; one Converse call costs 0.019 at the deployment price. Once that spend settles
    on the key, the next call must be refused with 422 `budget_exceeded` and reach no deployment. The router hop
    carries the key's metadata (`get_passthrough_router_request_metadata`), otherwise spend would never be charged
    to the key and the budget would never close."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(
            **bedrock_deployment(wire, input_cost_per_token=_INPUT_RATE, output_cost_per_token=_OUTPUT_RATE)
        )
        key: Final = scenario.key(max_budget=0.01)
        assert _converse(gateway, group, key).status_code == 200
        eventually(lambda: key_spend(gateway, key), lambda spent: spent > 0, seconds=SPEND_SECONDS)
        refused: Final = refused_until(lambda: _converse(gateway, group, key), 422)
        assert openai_error(refused, 422)["type"] == "budget_exceeded", refused.text
        wire.drain()
        assert _converse(gateway, group, key).status_code == 422
        assert posts(wire) == ()


def test_the_row_carries_the_keys_team(gateway: Gateway) -> None:
    """A key of a team calls `/bedrock/model/<group>/converse`. The router pass-through row must carry that
    `team_id`, so team budgets and reports count Bedrock pass-through traffic."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert _converse(gateway, group, key).status_code == 200
        assert spend_row(gateway, key)["team_id"] == team


def test_a_deployment_error_is_relayed_with_its_status_and_logged_as_a_failure(gateway: Gateway) -> None:
    """The deployment answers 500 to a Converse call whose body carries the failure marker. The caller must receive
    a 5xx and the key's row must have `status` `failure`. UNVERIFIED: whether the router hop maps Bedrock's 500 to
    500 or 503 for pass-through, so only the 5xx class is pinned."""
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        group: Final = scenario.model(**bedrock_deployment(wire))
        key: Final = scenario.key()
        body: Final = {"messages": [{"role": "user", "content": [{"text": "dec1-passthru-fail"}]}]}
        response: Final = gateway.request("POST", f"/bedrock/model/{group}/converse", body, key=key)
        assert 500 <= response.status_code < 600, response.text
        assert spend_row(gateway, key)["status"] == "failure"
