"""simple-shuffle across providers: how `weight`, `rpm`, `tpm` and saved router weights split a model group's traffic.

The default strategy runs natively in the Rust router core (`simple_shuffle` in
`litellm/rust/crates/router-core/src/select.rs:130`). It picks at random, weighted by the request's saved
router weights for the group (a key's or team's `router_settings.weights`) when there are any, else by every
deployment's `weight`, else `rpm`, else `tpm`. Each set is normalised by its largest value, and a set whose largest
value is zero is skipped. With no weight of any kind the pick is uniform. A missing value counts as zero, so a
deployment without a weight next to a weighted sibling never serves.

Every group here mixes deployments on OpenAI, Azure OpenAI, Bedrock, Vertex AI and Anthropic, so a rewrite that
weighs deployments differently per provider, or reads the weight from a different place, shows up as a shifted
share. The core draws from the thread RNG and offers no seed to the proxy, so the shares are checked over 200
requests with bounds of about five standard deviations around the expected share.

Saved router weights are validated when a key or team is written (`validate_router_settings_weights` in
`litellm/proxy/management_endpoints/router_weights.py`) and read on every request
(`hierarchical_router_settings_updates` in `litellm/proxy/common_request_processing.py:1070`): the key's
`router_settings` win over the team's as a whole, they are not merged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    dep,
    ident,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(600), pytest.mark.shares_worker_state]

SAMPLES: Final = 200

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "routing_strategy": "simple-shuffle",
    "num_retries": 0,
    "disable_cooldowns": True,
}


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("azure", "shuf-weight", "sw-azure", base, weight=3),
        dep("openai", "shuf-weight", "sw-openai", base, weight=1),
        dep("openai", "shuf-three", "s3-openai", base, weight=1),
        dep("bedrock", "shuf-three", "s3-bedrock", base, weight=1),
        dep("vertex", "shuf-three", "s3-vertex", base, weight=2),
        dep("anthropic", "shuf-rpm", "sr-anthropic", base, rpm=300),
        dep("bedrock", "shuf-rpm", "sr-bedrock", base, rpm=100),
        dep("azure", "shuf-tpm", "st-azure", base, tpm=90000),
        dep("vertex", "shuf-tpm", "st-vertex", base, tpm=30000),
        dep("openai", "shuf-uniform", "su-openai", base),
        dep("bedrock", "shuf-uniform", "su-bedrock", base),
        dep("anthropic", "shuf-uniform", "su-anthropic", base),
        dep("azure", "shuf-zero", "sz-azure", base, weight=0),
        dep("vertex", "shuf-zero", "sz-vertex", base, weight=1),
        dep("openai", "shuf-float", "sf-openai", base, weight=0.5),
        dep("anthropic", "shuf-float", "sf-anthropic", base, weight=1.5),
        dep("bedrock", "shuf-allzero", "sa-bedrock", base, weight=0, rpm=10),
        dep("vertex", "shuf-allzero", "sa-vertex", base, weight=0, rpm=30),
        dep("openai", "shuf-rpm-over-tpm", "srt-openai", base, rpm=1, tpm=1),
        dep("azure", "shuf-rpm-over-tpm", "srt-azure", base, tpm=1000000),
        dep("openai", "shuf-saved", "ss-openai", base, weight=100),
        dep("bedrock", "shuf-saved", "ss-bedrock", base),
        dep("anthropic", "shuf-saved", "ss-anthropic", base),
        dep("vertex", "shuf-other", "so-vertex", base, weight=1),
        dep("azure", "shuf-other", "so-azure", base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-shuffle"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_weights_of_three_and_one_split_azure_and_openai_about_three_to_one(rig: Rig) -> None:
    """An Azure deployment has `weight: 3` and an OpenAI deployment of the same group `weight: 1`.

    Over 200 requests Azure must serve between 120 and 180 (expected 150) and OpenAI the rest, and both must
    serve at least once. A rewrite that ignored `weight` would give about 100 each.
    """
    counts: Final = served_many(rig.gateway, "shuf-weight", SAMPLES)
    assert sum(counts.values()) == SAMPLES
    assert 120 <= counts["sw-azure"] <= 180, counts
    assert counts["sw-openai"] >= 20, counts


def test_three_providers_weighted_one_one_two_get_a_quarter_a_quarter_and_a_half(rig: Rig) -> None:
    """OpenAI and Bedrock have `weight: 1` and Vertex `weight: 2` in one group.

    Over 200 requests OpenAI and Bedrock must each serve between 20 and 80 (expected 50) and Vertex between 65 and
    135 (expected 100). Weights are relative to each other, normalised by the largest, so no provider is favoured.
    """
    counts: Final = served_many(rig.gateway, "shuf-three", SAMPLES)
    assert 20 <= counts["s3-openai"] <= 80, counts
    assert 20 <= counts["s3-bedrock"] <= 80, counts
    assert 65 <= counts["s3-vertex"] <= 135, counts


def test_rpm_weights_the_shuffle_between_anthropic_and_bedrock_when_no_weight_is_set(rig: Rig) -> None:
    """No deployment sets `weight`; Anthropic has `rpm: 300` and Bedrock `rpm: 100`.

    Over 200 requests Anthropic must serve between 120 and 180 (expected 150), Bedrock the rest. Under
    simple-shuffle `rpm` is only a weight, not a per-minute cap, so all 200 succeed within the same minute.
    """
    counts: Final = served_many(rig.gateway, "shuf-rpm", SAMPLES)
    assert sum(counts.values()) == SAMPLES
    assert 120 <= counts["sr-anthropic"] <= 180, counts
    assert counts["sr-bedrock"] >= 20, counts


def test_tpm_weights_the_shuffle_between_azure_and_vertex_when_no_weight_or_rpm_is_set(rig: Rig) -> None:
    """No deployment sets `weight` or `rpm`; Azure has `tpm: 90000` and Vertex `tpm: 30000`.

    Over 200 requests Azure must serve between 120 and 180 (expected 150), Vertex the rest.
    """
    counts: Final = served_many(rig.gateway, "shuf-tpm", SAMPLES)
    assert 120 <= counts["st-azure"] <= 180, counts
    assert counts["st-vertex"] >= 20, counts


def test_a_group_without_any_weight_rpm_or_tpm_is_shuffled_uniformly_across_providers(rig: Rig) -> None:
    """OpenAI, Bedrock and Anthropic deployments set no `weight`, `rpm` or `tpm`.

    Over 150 requests each must serve between 21 and 79 (expected 50): the core falls back to a uniform
    `random.choice` (`select.rs:178`).
    """
    counts: Final = served_many(rig.gateway, "shuf-uniform", 150)
    assert set(counts) == {"su-openai", "su-bedrock", "su-anthropic"}, counts
    assert all(21 <= count <= 79 for count in counts.values()), counts


def test_an_explicit_weight_of_zero_never_serves_next_to_a_weighted_sibling(rig: Rig) -> None:
    """Azure has `weight: 0` and Vertex `weight: 1`.

    Forty requests must all be served by Vertex. A zero weight is a valid way to drain a deployment without
    removing it.
    """
    assert served_many(rig.gateway, "shuf-zero", 40) == {"sz-vertex": 40}


def test_fractional_weights_split_traffic_like_their_ratio(rig: Rig) -> None:
    """OpenAI has `weight: 0.5` and Anthropic `weight: 1.5`.

    Over 200 requests Anthropic must serve between 120 and 180 (expected 150). Weights are floats, not
    truncated to integers (a truncation would make OpenAI's weight zero and give Anthropic every request).
    """
    counts: Final = served_many(rig.gateway, "shuf-float", SAMPLES)
    assert 120 <= counts["sf-anthropic"] <= 180, counts
    assert counts["sf-openai"] >= 20, counts


def test_weights_that_are_all_zero_are_skipped_and_rpm_decides(rig: Rig) -> None:
    """Both deployments have `weight: 0`; Bedrock has `rpm: 10` and Vertex `rpm: 30`.

    A metric whose largest value is zero is skipped (`select.rs:158`), so `rpm` weighs the pick: over 200 requests
    Vertex must serve between 120 and 180 (expected 150) and Bedrock at least 20.
    """
    counts: Final = served_many(rig.gateway, "shuf-allzero", SAMPLES)
    assert 120 <= counts["sa-vertex"] <= 180, counts
    assert counts["sa-bedrock"] >= 20, counts


def test_rpm_is_read_before_tpm_even_when_tpm_is_far_larger(rig: Rig) -> None:
    """OpenAI has `rpm: 1, tpm: 1`; Azure has only `tpm: 1000000`.

    Because one deployment sets `rpm`, the `rpm` set weighs the pick and Azure's missing `rpm` counts as zero, so
    thirty requests must all reach OpenAI, even though Azure's `tpm` is a million times larger. `rpm: 1` is not
    a per-minute cap under simple-shuffle.
    """
    assert served_many(rig.gateway, "shuf-rpm-over-tpm", 30) == {"srt-openai": 30}


def test_a_keys_saved_router_weights_replace_the_config_weights_for_its_requests(rig: Rig) -> None:
    """The config gives OpenAI `weight: 100`; Bedrock and Anthropic set none. A key is created with
    `router_settings.weights` naming only the Bedrock deployment with weight 1.

    Thirty requests with that key must all reach Bedrock (deployments the saved weights leave out count as zero),
    while thirty requests with the master key must all reach OpenAI.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"weights": {"shuf-saved": {ident("ss-bedrock"): 1}}})
        assert served_many(rig.gateway, "shuf-saved", 30, key=key) == {"ss-bedrock": 30}
    assert served_many(rig.gateway, "shuf-saved", 30) == {"ss-openai": 30}


def test_saved_router_weights_split_traffic_by_their_own_ratio(rig: Rig) -> None:
    """A key's `router_settings.weights` give Bedrock 1 and Anthropic 3 in the group whose config weights OpenAI 100.

    Over 200 requests with that key OpenAI must never serve, Anthropic must serve between 120 and 180 (expected 150)
    and Bedrock the rest.
    """
    weights: Final[dict[str, JsonValue]] = {"shuf-saved": {ident("ss-bedrock"): 1, ident("ss-anthropic"): 3}}
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"weights": weights})
        counts: Final = served_many(rig.gateway, "shuf-saved", SAMPLES, key=key)
    assert counts["ss-openai"] == 0, counts
    assert 120 <= counts["ss-anthropic"] <= 180, counts
    assert counts["ss-bedrock"] >= 20, counts


def test_saved_router_weights_for_one_group_leave_other_groups_on_their_config_weights(rig: Rig) -> None:
    """A key's saved weights name only the group `shuf-saved`. The same key calls `shuf-other`, whose config gives
    Vertex `weight: 1` and Azure none.

    Twenty requests to `shuf-other` must all reach Vertex: saved weights are looked up by model group.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"weights": {"shuf-saved": {ident("ss-bedrock"): 1}}})
        assert served_many(rig.gateway, "shuf-other", 20, key=key) == {"so-vertex": 20}


def test_an_empty_saved_weights_group_falls_back_to_the_config_weights(rig: Rig) -> None:
    """A key's `router_settings.weights` maps `shuf-saved` to an empty object.

    An empty group is valid and skipped (`select.rs:141`), so twenty requests must all reach OpenAI, the
    deployment the config weights.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"weights": {"shuf-saved": {}}})
        assert served_many(rig.gateway, "shuf-saved", 20, key=key) == {"ss-openai": 20}


def test_a_teams_saved_router_weights_apply_to_its_keys(rig: Rig) -> None:
    """A team is created with `router_settings.weights` naming only the Anthropic deployment of `shuf-saved`; a key
    of that team sets no router settings.

    Twenty requests with the key must all reach Anthropic.
    """
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"weights": {"shuf-saved": {ident("ss-anthropic"): 1}}})
        key: Final = scenario.key(team_id=team)
        assert served_many(rig.gateway, "shuf-saved", 20, key=key) == {"ss-anthropic": 20}


def test_a_keys_router_settings_shadow_the_teams_weights_entirely(rig: Rig) -> None:
    """The team saves weights naming only Anthropic; its key saves `router_settings: {"num_retries": 0}` and no
    weights.

    Key settings win over team settings as a whole (`_get_hierarchical_router_settings` in
    `litellm/proxy/proxy_server.py:8603` returns the key's dict without merging), so twenty requests with the key
    must all reach OpenAI on the config weights, not Anthropic.
    """
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"weights": {"shuf-saved": {ident("ss-anthropic"): 1}}})
        key: Final = scenario.key(team_id=team, router_settings={"num_retries": 0})
        assert served_many(rig.gateway, "shuf-saved", 20, key=key) == {"ss-openai": 20}


def test_a_key_whose_saved_weights_win_over_its_teams_weights(rig: Rig) -> None:
    """The team saves weights naming only Anthropic; its key saves weights naming only Bedrock.

    Twenty requests with the key must all reach Bedrock.
    """
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"weights": {"shuf-saved": {ident("ss-anthropic"): 1}}})
        key: Final = scenario.key(team_id=team, router_settings={"weights": {"shuf-saved": {ident("ss-bedrock"): 1}}})
        assert served_many(rig.gateway, "shuf-saved", 20, key=key) == {"ss-bedrock": 20}


@pytest.mark.parametrize(
    ("weights", "status", "detail"),
    [
        pytest.param(
            {"shuf-saved": {"dec1-routestrat-no-such-deployment": 1}},
            400,
            "Unknown deployment ID in router weights: dec1-routestrat-no-such-deployment",
            id="unknown-id",
        ),
        pytest.param(
            {"shuf-other": {ident("ss-bedrock"): 1}},
            400,
            f"Deployment {ident('ss-bedrock')} does not belong to model group shuf-other",
            id="wrong-group",
        ),
        pytest.param(
            {"shuf-saved": {ident("ss-bedrock"): 0, ident("ss-anthropic"): 0}},
            422,
            "Each nonempty weights group must contain at least one positive weight",
            id="all-zero",
        ),
        pytest.param(
            {"shuf-saved": {ident("ss-bedrock"): -1}},
            422,
            "Input should be greater than or equal to 0",
            id="negative",
        ),
        pytest.param(
            {"shuf-saved": {ident("ss-bedrock"): "1"}},
            422,
            "Input should be a valid number",
            id="string-weight",
        ),
    ],
)
def test_key_generate_refuses_saved_router_weights_that_cannot_route(
    rig: Rig, weights: dict[str, JsonValue], status: int, detail: str
) -> None:
    """`POST /key/generate` with `router_settings.weights` that name an unknown deployment id, a deployment under a
    group it does not belong to, only zero weights, a negative weight, or a weight written as a string.

    An unknown id or a wrong group answers HTTP 400 with the detail from
    `litellm/proxy/management_endpoints/router_weights.py`. The shape checks run earlier, in the request model's
    `RouterWeights` type (`litellm/types/router_weights.py:7-17`: strict float, `ge=0`, one positive weight per
    group), so those answer FastAPI's 422 with the pydantic message. Without these checks a key could be saved
    with weights that make every request fail at routing.
    """
    response: Final = rig.gateway.request("POST", "/key/generate", {"router_settings": {"weights": weights}})
    assert response.status_code == status, response.text
    assert detail in response.text, response.text


def test_team_new_refuses_saved_router_weights_naming_an_unknown_deployment(rig: Rig) -> None:
    """`POST /team/new` with `router_settings.weights` naming a deployment id no model has.

    The answer is HTTP 400 containing `Unknown deployment ID in router weights: dec1-routestrat-ghost`, the same
    validation keys get (`team_endpoints.py` imports `validate_router_settings_weights`).
    """
    response: Final = rig.gateway.request(
        "POST",
        "/team/new",
        {"router_settings": {"weights": {"shuf-saved": {"dec1-routestrat-ghost": 1}}}},
    )
    assert response.status_code == 400, response.text
    assert "Unknown deployment ID in router weights: dec1-routestrat-ghost" in response.text, response.text


def test_the_served_deployment_is_the_one_named_in_x_litellm_model_id_for_every_provider(rig: Rig) -> None:
    """Sixty requests to the five-way mixed groups `shuf-three` and `shuf-weight`.

    For every response the `x-litellm-model-id` header must equal the id of the deployment whose label the peer
    put in the answer text (`served` asserts it), so the header reveals the shuffle's choice for OpenAI, Azure,
    Bedrock and Vertex deployments alike.
    """
    labels: Final = {served(rig.gateway, group) for group in ("shuf-three", "shuf-weight") for _ in range(30)}
    assert labels <= {"s3-openai", "s3-bedrock", "s3-vertex", "sw-azure", "sw-openai"}
    assert len(labels) >= 4, labels
