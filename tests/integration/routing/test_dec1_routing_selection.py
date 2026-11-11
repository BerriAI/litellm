"""How one request is matched to one deployment: names, aliases, wildcards, ids, weights, order, filters.

One owned proxy runs with the default `simple-shuffle` strategy, `num_retries: 0`, cooldowns disabled, and
`model_group_alias`, `enable_pre_call_checks`, `enable_tag_filtering`, `default_litellm_params` and
`set_verbose` all set. Every healthy scripted upstream model answers with the text `served <upstream model>`,
so the content of a 200 names the deployment that served it.

Simple shuffle picks at random, weighted by `weight`, else by `rpm`, else by `tpm`, each normalised by the
largest value in the group, so a deployment whose weight is zero or missing next to a weighted sibling is
never picked. Only deployments with the lowest `order` are candidates. Pre-call checks drop deployments whose
`model_info.max_input_tokens` is smaller than the prompt, and tag filtering keeps the deployments whose
`litellm_params.tags` match the request's tags.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    INVALID_MODEL,
    Rig,
    chat,
    content_of,
    deployment,
    error_message,
    error_of,
    routing_rig,
)
from pydantic import JsonValue

from tests.integration._support.client import list_value, object_value

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

DEFAULT_TEMPERATURE: Final = 0.25
SMALL_WINDOW_TOKENS: Final = 20
LONG_PROMPT: Final = "routing contract " * 400
TAGS_REFUSED: Final = "Not allowed to access model due to tags configuration"

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "routing_strategy": "simple-shuffle",
    "num_retries": 0,
    "disable_cooldowns": True,
    "model_group_alias": {
        "dec1-sel-alias": "sel-real",
        "dec1-sel-hidden-alias": {"model": "sel-real", "hidden": True},
    },
    "enable_pre_call_checks": True,
    "enable_tag_filtering": True,
    "default_litellm_params": {"temperature": DEFAULT_TEMPERATURE},
    "set_verbose": True,
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("sel-weight", "ok-sel-weighted", api_base, weight=1),
        deployment("sel-weight", "ok-sel-unweighted", api_base),
        deployment("sel-split", "ok-sel-heavy", api_base, weight=9),
        deployment("sel-split", "ok-sel-light", api_base, weight=1),
        deployment("sel-rpm", "ok-sel-rpm", api_base, rpm=1000),
        deployment("sel-rpm", "ok-sel-norpm", api_base),
        deployment("sel-tpm", "ok-sel-tpm", api_base, tpm=100000),
        deployment("sel-tpm", "ok-sel-notpm", api_base),
        deployment("sel-weight-over-rpm", "ok-sel-weight-wins", api_base, weight=1, rpm=100),
        deployment("sel-weight-over-rpm", "ok-sel-rpm-loses", api_base, rpm=10000),
        deployment("sel-order", "ok-sel-order-one", api_base, order=1),
        deployment("sel-order", "ok-sel-order-two", api_base, order=2),
        deployment("sel-order-fail", "status-500-sel-order-one", api_base, order=1),
        deployment("sel-order-fail", "ok-sel-order-two-backup", api_base, order=2),
        deployment("sel-real", "ok-sel-real", api_base),
        {
            "model_name": "dec1sel/*",
            "litellm_params": {"model": "openai/*", "api_key": "dec1-routing-provider-key", "api_base": api_base},
            "model_info": {"id": "dec1-sel-wildcard"},
        },
        deployment("openai/dec1-sel-prefixed", "ok-sel-prefixed", api_base),
        deployment("sel-id", "ok-sel-id-first", api_base),
        deployment("sel-id", "ok-sel-id-second", api_base),
        deployment("sel-defaults", "ok-sel-defaults", api_base),
        deployment("sel-window", "ok-sel-small", api_base, model_info={"max_input_tokens": SMALL_WINDOW_TOKENS}),
        deployment("sel-window", "ok-sel-large", api_base, model_info={"max_input_tokens": 100000}),
        deployment("sel-window-tiny", "ok-sel-tiny", api_base, model_info={"max_input_tokens": SMALL_WINDOW_TOKENS}),
        deployment("sel-tags", "ok-sel-paid", api_base, tags=["paid"]),
        deployment("sel-tags", "ok-sel-free", api_base, tags=["free"]),
        deployment("sel-tags", "ok-sel-default", api_base, tags=["default"]),
        deployment("sel-tags-strict", "ok-sel-only-paid", api_base, tags=["paid"]),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-selection"), _models, ROUTER_SETTINGS) as value:
        yield value


def _served(rig: Rig, model: str, **extra: JsonValue) -> str:
    response: Final = chat(rig.gateway, model, **extra)
    assert response.status_code == 200, response.text
    return content_of(response).removeprefix("served ")


def _served_many(rig: Rig, model: str, requests: int, **extra: JsonValue) -> Counter[str]:
    return Counter(_served(rig, model, **extra) for _ in range(requests))


def test_a_deployment_with_a_weight_takes_all_traffic_from_a_sibling_without_one(rig: Rig) -> None:
    """One deployment has `weight: 1` and its sibling has no weight.

    Twelve requests must all be served by the weighted deployment. A missing weight counts as zero.
    """
    assert _served_many(rig, "sel-weight", 12) == {"ok-sel-weighted": 12}


def test_weights_of_nine_and_one_send_most_traffic_to_the_heavier_deployment(rig: Rig) -> None:
    """The deployments have `weight: 9` and `weight: 1`.

    Of sixty requests the heavier deployment must serve more than twice as many as the lighter one (the
    expected split is 54 to 6), and nothing else serves the group.
    """
    served: Final = _served_many(rig, "sel-split", 60)
    assert set(served) <= {"ok-sel-heavy", "ok-sel-light"}
    assert served["ok-sel-heavy"] > 2 * served["ok-sel-light"], served


def test_rpm_is_the_selection_weight_when_no_deployment_sets_weight(rig: Rig) -> None:
    """One deployment has `rpm: 1000`, its sibling has no `rpm`, and neither sets `weight`.

    Twelve requests must all be served by the deployment that declares an rpm.
    """
    assert _served_many(rig, "sel-rpm", 12) == {"ok-sel-rpm": 12}


def test_tpm_is_the_selection_weight_when_no_deployment_sets_weight_or_rpm(rig: Rig) -> None:
    """One deployment has `tpm: 100000`, its sibling has no `tpm`, and neither sets `weight` or `rpm`.

    Twelve requests must all be served by the deployment that declares a tpm.
    """
    assert _served_many(rig, "sel-tpm", 12) == {"ok-sel-tpm": 12}


def test_weight_decides_selection_before_rpm(rig: Rig) -> None:
    """One deployment has `weight: 1` and `rpm: 100`; its sibling has no weight and `rpm: 10000`.

    Twelve requests must all be served by the weighted deployment, although the sibling has a hundred times
    its rpm: rpm is only read as a weight when no deployment in the group has a positive `weight`.
    """
    assert _served_many(rig, "sel-weight-over-rpm", 12) == {"ok-sel-weight-wins": 12}


def test_only_the_lowest_order_deployment_serves_while_it_is_healthy(rig: Rig) -> None:
    """The deployments set `litellm_params.order` to 1 and 2 and both are healthy.

    Twelve requests must all be served by the `order: 1` deployment.
    """
    assert _served_many(rig, "sel-order", 12) == {"ok-sel-order-one": 12}


def test_a_failing_order_one_deployment_hands_the_request_to_order_two_as_a_fallback(rig: Rig) -> None:
    """The `order: 1` deployment answers 500 and the `order: 2` deployment is healthy; no fallbacks are set.

    The caller gets 200 from the `order: 2` deployment, the failing upstream model received exactly one
    request, and the response carries `x-litellm-attempted-fallbacks: 1`. Moving to the next order level is
    counted as a fallback.
    """
    response: Final = chat(rig.gateway, "sel-order-fail")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-sel-order-two-backup"
    assert rig.upstream.count("status-500-sel-order-one") == 1
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_a_model_group_alias_routes_to_the_group_it_names(rig: Rig) -> None:
    """`model_group_alias` maps `dec1-sel-alias` to the group `sel-real`.

    A request for `dec1-sel-alias` returns 200 from `sel-real`'s deployment, with `x-litellm-model-id:
    dec1-ok-sel-real`, and the upstream receives that deployment's own model name `ok-sel-real`.
    """
    response: Final = chat(rig.gateway, "dec1-sel-alias")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-sel-real"
    assert response.headers["x-litellm-model-id"] == "dec1-ok-sel-real"


def test_a_hidden_alias_is_routable_but_absent_from_the_model_list(rig: Rig) -> None:
    """`model_group_alias` has a plain alias and one written as `{"model": "sel-real", "hidden": true}`.

    A request for the hidden alias returns 200 from `sel-real`'s deployment. `GET /v1/models` lists the plain
    alias `dec1-sel-alias` and does not list `dec1-sel-hidden-alias`. UNVERIFIED: that `/v1/models` lists
    plain aliases for the master key; the `hidden` flag is documented on `RouterModelGroupAliasItem` in
    `litellm/types/router.py`.
    """
    assert _served(rig, "dec1-sel-hidden-alias") == "ok-sel-real"
    listed: Final = {object_value(entry)["id"] for entry in list_value(rig.gateway.get("/v1/models")["data"])}
    assert "dec1-sel-alias" in listed
    assert "dec1-sel-hidden-alias" not in listed


def test_a_wildcard_group_sends_the_requested_suffix_as_the_upstream_model(rig: Rig) -> None:
    """A deployment is named `dec1sel/*` with `litellm_params.model: openai/*`.

    A request for `dec1sel/ok-sel-wild-anything` returns 200, and the upstream receives exactly one request
    whose `model` is `ok-sel-wild-anything`: the part the wildcard matched replaces the `*` of the deployment.
    """
    assert _served(rig, "dec1sel/ok-sel-wild-anything") == "ok-sel-wild-anything"
    assert rig.upstream.count("ok-sel-wild-anything") == 1


def test_a_group_whose_name_carries_a_provider_prefix_is_matched_by_its_full_name(rig: Rig) -> None:
    """A group is named `openai/dec1-sel-prefixed` and its deployment calls the upstream model
    `ok-sel-prefixed`.

    A request for `openai/dec1-sel-prefixed` returns 200 from that deployment, and the upstream receives the
    deployment's model name, not the group name with the prefix stripped. A group name is matched as a whole
    before any provider prefix is interpreted.
    """
    assert _served(rig, "openai/dec1-sel-prefixed") == "ok-sel-prefixed"
    assert rig.upstream.count("dec1-sel-prefixed") == 0


def test_a_request_whose_model_is_a_deployment_id_always_reaches_that_deployment(rig: Rig) -> None:
    """Group `sel-id` has two healthy deployments; one has the id `dec1-ok-sel-id-second`.

    Ten requests with `model` set to that id must all be served by that deployment, never by its sibling, and
    each response carries `x-litellm-model-id: dec1-ok-sel-id-second`.
    """
    responses: Final = tuple(chat(rig.gateway, "dec1-ok-sel-id-second") for _ in range(10))
    assert {response.status_code for response in responses} == {200}
    assert {content_of(response) for response in responses} == {"served ok-sel-id-second"}
    assert {response.headers["x-litellm-model-id"] for response in responses} == {"dec1-ok-sel-id-second"}


def test_a_success_names_the_serving_deployment_and_its_api_base_in_response_headers(rig: Rig) -> None:
    """A request to the single-deployment group `sel-real` succeeds.

    The response carries `x-litellm-model-id: dec1-ok-sel-real` and an `x-litellm-model-api-base` equal to the
    deployment's configured `api_base` (compared without a trailing slash).
    """
    response: Final = chat(rig.gateway, "sel-real")
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == "dec1-ok-sel-real"
    assert response.headers["x-litellm-model-api-base"].rstrip("/") == rig.upstream.api_base


def test_default_litellm_params_fill_a_parameter_the_caller_did_not_send(rig: Rig) -> None:
    """`default_litellm_params: {"temperature": 0.25}`; the caller sends no temperature.

    The request the upstream receives must carry `temperature: 0.25`.
    """
    assert _served(rig, "sel-defaults") == "ok-sel-defaults"
    assert rig.upstream.calls("ok-sel-defaults")[-1].body["temperature"] == DEFAULT_TEMPERATURE


def test_a_parameter_the_caller_sends_wins_over_default_litellm_params(rig: Rig) -> None:
    """`default_litellm_params: {"temperature": 0.25}`; the caller sends `temperature: 0.9`.

    The request the upstream receives must carry `temperature: 0.9`.
    """
    assert _served(rig, "sel-defaults", temperature=0.9) == "ok-sel-defaults"
    assert rig.upstream.calls("ok-sel-defaults")[-1].body["temperature"] == 0.9


def test_pre_call_checks_skip_a_deployment_whose_context_window_is_smaller_than_the_prompt(rig: Rig) -> None:
    """Pre-call checks are on; the group has a deployment with `model_info.max_input_tokens: 20` and one with
    100000.

    Eight requests with a prompt of roughly 800 words must all be served by the large deployment, and the
    small deployment's upstream model receives nothing.
    """
    messages: Final[JsonValue] = [{"role": "user", "content": LONG_PROMPT}]
    assert _served_many(rig, "sel-window", 8, messages=messages) == {"ok-sel-large": 8}
    assert rig.upstream.count("ok-sel-small") == 0


def test_pre_call_checks_return_400_without_an_upstream_call_when_no_window_fits(rig: Rig) -> None:
    """Pre-call checks are on; the group's only deployment has `model_info.max_input_tokens: 20`.

    A prompt of roughly 800 words returns HTTP 400 with `code` `"400"`. The message contains
    `Context Window exceeded for given call` and `Max Input Tokens=20`, and the upstream receives nothing.
    """
    response: Final = chat(rig.gateway, "sel-window-tiny", messages=[{"role": "user", "content": LONG_PROMPT}])
    assert response.status_code == 400, response.text
    assert error_of(response)["code"] == "400"
    assert "Context Window exceeded for given call" in error_message(response), response.text
    assert f"Max Input Tokens={SMALL_WINDOW_TOKENS}" in error_message(response), response.text
    assert rig.upstream.count("ok-sel-tiny") == 0


def test_a_request_tag_in_metadata_routes_to_the_deployment_with_that_tag(rig: Rig) -> None:
    """Tag filtering is on; the group has deployments tagged `paid`, `free` and `default`.

    Eight requests that send `metadata: {"tags": ["paid"]}` must all be served by the `paid` deployment.
    """
    assert _served_many(rig, "sel-tags", 8, metadata={"tags": ["paid"]}) == {"ok-sel-paid": 8}


def test_the_x_litellm_tags_header_routes_to_the_deployment_with_that_tag(rig: Rig) -> None:
    """Tag filtering is on; the group has deployments tagged `paid`, `free` and `default`.

    Eight requests that send the header `x-litellm-tags: free` must all be served by the `free` deployment.
    """
    responses: Final = tuple(chat(rig.gateway, "sel-tags", headers={"x-litellm-tags": "free"}) for _ in range(8))
    assert {response.status_code for response in responses} == {200}
    assert {content_of(response) for response in responses} == {"served ok-sel-free"}


def test_a_request_without_tags_is_served_by_the_default_tagged_deployment(rig: Rig) -> None:
    """Tag filtering is on; the group has deployments tagged `paid`, `free` and `default`.

    Eight requests without any tag must all be served by the deployment tagged `default`.
    """
    assert _served_many(rig, "sel-tags", 8) == {"ok-sel-default": 8}


def test_a_request_tag_no_deployment_carries_is_refused_with_401(rig: Rig) -> None:
    """Tag filtering is on; the group's only deployment is tagged `paid` and the caller sends the tag `free`.

    The caller gets HTTP 401 with a message containing
    `Not allowed to access model due to tags configuration`, and the upstream receives nothing.
    UNVERIFIED: that this case does not fail open; see `_resolve_or_fail_open` in
    `litellm/router_strategy/tag_based_routing.py`.
    """
    response: Final = chat(rig.gateway, "sel-tags-strict", metadata={"tags": ["free"]})
    assert response.status_code == 401, response.text
    assert TAGS_REFUSED in error_message(response), response.text
    assert rig.upstream.count("ok-sel-only-paid") == 0


def test_a_model_name_no_group_alias_wildcard_or_id_matches_is_refused_with_400(rig: Rig) -> None:
    """The caller asks for `dec1-sel-no-such-model`, which nothing in the model list matches.

    The caller gets HTTP 400 with `type` `invalid_request_error` and a message that names the model and says
    `There are no healthy deployments for this model`, never the proxy's own `Invalid model name` refusal.
    Pinned current behaviour: this proxy has a wildcard group (`dec1sel/*`), and once any wildcard pattern is
    configured the proxy hands every unknown name to the router (litellm/proxy/route_llm_request.py:1076)
    instead of answering `Invalid model name passed in model=...` itself, so the router's no-deployment 400
    is what the caller sees.
    """
    response: Final = chat(rig.gateway, "dec1-sel-no-such-model")
    assert response.status_code == 400, response.text
    assert error_of(response)["type"] == "invalid_request_error", response.text
    assert error_message(response).startswith(
        "litellm.BadRequestError: You passed in model=dec1-sel-no-such-model. "
        "There are no healthy deployments for this model"
    ), response.text
    assert INVALID_MODEL not in response.text


def test_set_verbose_in_router_settings_is_accepted_and_the_proxy_keeps_routing(rig: Rig) -> None:
    """The proxy booted with `router_settings.set_verbose: true`.

    The setting only raises the router's log level, so the observable contract is that the proxy starts with
    it and serves a plain request with 200. A config loader that rejects the key as unknown would fail every
    test in this file at boot.
    """
    assert _served(rig, "sel-real") == "ok-sel-real"
    assert rig.gateway.request("GET", "/health/readiness").status_code == 200
