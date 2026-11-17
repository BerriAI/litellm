"""Tag-based routing across providers: match-any versus match-all, the `default` tag, per-group and per-key switches.

Tag filtering runs after the cooldown and pre-call filters and before the order filter and the strategy
(`select.rs:587`), answered by `get_deployments_for_tag` in `litellm/router_strategy/tag_based_routing.py:560`.
It is on when the router's `enable_tag_filtering` is true, when any deployment of the group sets
`model_info.enable_tag_filtering` (which overrides the router-wide value for that group), or when the caller's key
or team saved `router_settings.enable_tag_filtering: true`. A request body's own `enable_tag_filtering` is dropped
(`route_llm_request.py:797`), so a caller cannot switch tag policy off or on by itself.

With `tag_filtering_match_any` true (the default) a deployment matches when it shares any tag with the request;
with false it must carry every request tag. A request with no tags goes to the deployments tagged `default`, or
to the whole group when none is. A request whose tags match nothing falls to the `default` deployments when the
tag is unknown to the group, and is refused with 401 `Not allowed to access model due to tags configuration` when
the group knows the tag but no candidate carries all of it, or when nothing is tagged `default`.

Deployments are on Azure OpenAI, Bedrock, Vertex AI, OpenAI and Anthropic. The basic single-tag cases and the
`&`/`!` operators and prefixes are pinned in `test_dec1_routing_selection.py` and `test_tag_routing_prefix.py`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_routestrat_support import (
    TAGS_REFUSED,
    Rig,
    chat,
    dep,
    error_message,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]


def _tags(*tags: str) -> dict[str, JsonValue]:
    return {"metadata": {"tags": list(tags)}}


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("azure", "tag-tiers", "tt-azure", base, tags=["premium"]),
        dep("bedrock", "tag-tiers", "tt-bedrock", base, tags=["standard"]),
        dep("vertex", "tag-tiers", "tt-vertex", base, tags=["default"]),
        dep("bedrock", "tag-shared", "ts-bedrock", base, tags=["batch"], weight=3),
        dep("openai", "tag-shared", "ts-openai", base, tags=["batch"], weight=1),
        dep("anthropic", "tag-shared", "ts-anthropic", base, tags=["interactive"], weight=100),
        dep("openai", "tag-both", "tb-openai", base, tags=["premium", "eu"]),
        dep("anthropic", "tag-both", "tb-anthropic", base, tags=["premium"]),
        dep("vertex", "tag-untagged", "tu-vertex", base),
        dep("openai", "tag-untagged", "tu-openai", base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
        "enable_tag_filtering": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-tags"), _models, settings) as value:
        yield value


def test_one_matching_tag_is_enough_under_the_default_match_any(rig: Rig) -> None:
    """The request carries tags `premium` and `eu`; Azure is tagged `premium`, Bedrock `standard`, Vertex `default`.

    Twenty requests must all be served by Azure: sharing one tag is a match, and a matched deployment is preferred
    over the `default` one.
    """
    assert served_many(rig.gateway, "tag-tiers", 20, **_tags("premium", "eu")) == {"tt-azure": 20}


def test_an_untagged_request_goes_to_the_default_tagged_provider(rig: Rig) -> None:
    """A request without tags on the group where Vertex is tagged `default`.

    Twenty requests must all be served by Vertex, never by the Azure or Bedrock deployments with other tags.
    """
    assert served_many(rig.gateway, "tag-tiers", 20) == {"tt-vertex": 20}


def test_a_tag_unknown_to_the_group_falls_back_to_the_default_deployment(rig: Rig) -> None:
    """The request carries the tag `gold`, which no deployment of the group carries; Vertex is tagged `default`.

    Twenty requests must all be served by Vertex with 200: an unknown tag is treated like no tag when the group has
    a `default` deployment (`_tag_known_to_group` is false).
    """
    assert served_many(rig.gateway, "tag-tiers", 20, **_tags("gold")) == {"tt-vertex": 20}


def test_deployments_sharing_a_tag_are_shuffled_by_weight_among_themselves(rig: Rig) -> None:
    """Bedrock (`weight: 3`) and OpenAI (`weight: 1`) are tagged `batch`; Anthropic (`weight: 100`) is tagged
    `interactive`. Requests carry `batch`.

    Over 200 requests Anthropic must serve none, Bedrock between 120 and 180 (expected 150) and OpenAI the rest:
    tag filtering narrows the pool first and the shuffle weighs only what is left.
    """
    counts: Final = served_many(rig.gateway, "tag-shared", 200, **_tags("batch"))
    assert counts["ts-anthropic"] == 0, counts
    assert 120 <= counts["ts-bedrock"] <= 180, counts
    assert counts["ts-openai"] >= 20, counts


def test_any_of_two_tagged_providers_can_serve_a_request_naming_their_common_tag(rig: Rig) -> None:
    """OpenAI is tagged `premium` and `eu`, Anthropic only `premium`. Requests carry `premium`.

    Over forty requests both providers must serve at least five: under match-any both match, and the shuffle
    picks uniformly between two unweighted deployments.
    """
    counts: Final = served_many(rig.gateway, "tag-both", 40, **_tags("premium"))
    assert counts["tb-openai"] >= 5, counts
    assert counts["tb-anthropic"] >= 5, counts


def test_a_tagged_request_to_an_untagged_group_is_refused_with_401(rig: Rig) -> None:
    """Neither deployment of `tag-untagged` (Vertex, OpenAI) has tags or a `default` tag; the request carries `premium`.

    The caller gets HTTP 401 with `Not allowed to access model due to tags configuration` in the message and the
    peer receives nothing. UNVERIFIED: that no fail-open applies here; `_resolve_or_fail_open` only opens for a
    group with `model_info.allow_fail_open`.
    """
    response: Final = chat(rig.gateway, "tag-untagged", **_tags("premium"))
    assert response.status_code == 401, response.text
    assert TAGS_REFUSED in error_message(response), response.text
    assert rig.peer.count("tu-vertex") == 0
    assert rig.peer.count("tu-openai") == 0


def test_an_untagged_request_to_an_untagged_group_uses_every_deployment(rig: Rig) -> None:
    """A request without tags on `tag-untagged`, where nothing is tagged.

    Over forty requests both Vertex and OpenAI must serve at least five: with no `default` tag the whole group is
    the pool.
    """
    counts: Final = served_many(rig.gateway, "tag-untagged", 40)
    assert counts["tu-vertex"] >= 5, counts
    assert counts["tu-openai"] >= 5, counts


def test_the_x_litellm_tags_header_routes_across_providers_like_metadata_tags(rig: Rig) -> None:
    """The request sends `x-litellm-tags: standard` and no body tags.

    Twenty requests must all be served by the Bedrock deployment tagged `standard`.
    """
    responses: Final = [chat(rig.gateway, "tag-tiers", headers={"x-litellm-tags": "standard"}) for _ in range(20)]
    assert [response.status_code for response in responses] == [200] * 20
    assert Counter(response.headers["x-litellm-model-id"] for response in responses) == {
        "dec1-routestrat-tt-bedrock": 20
    }


def _match_all_models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("openai", "tag-all", "ta-openai", base, tags=["premium", "eu"]),
        dep("anthropic", "tag-all", "ta-anthropic", base, tags=["premium"]),
        dep("bedrock", "tag-all-default", "tad-bedrock", base, tags=["premium"]),
        dep("vertex", "tag-all-default", "tad-vertex", base, tags=["default"]),
    )


@pytest.fixture(scope="module")
def match_all(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
        "enable_tag_filtering": True,
        "tag_filtering_match_any": False,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-tags-all"), _match_all_models, settings) as value:
        yield value


def test_with_match_any_off_only_a_deployment_carrying_every_request_tag_serves(match_all: Rig) -> None:
    """`tag_filtering_match_any: false`; OpenAI is tagged `premium` and `eu`, Anthropic only `premium`. Requests
    carry both tags.

    Twenty requests must all be served by OpenAI: the request tags must be a subset of the deployment's tags.
    """
    assert served_many(match_all.gateway, "tag-all", 20, **_tags("premium", "eu")) == {"ta-openai": 20}


def test_with_match_any_off_a_single_request_tag_matches_both_deployments(match_all: Rig) -> None:
    """`tag_filtering_match_any: false`; requests carry only `premium`, which both deployments have.

    Over forty requests both OpenAI and Anthropic must serve at least five.
    """
    counts: Final = served_many(match_all.gateway, "tag-all", 40, **_tags("premium"))
    assert counts["ta-openai"] >= 5, counts
    assert counts["ta-anthropic"] >= 5, counts


def test_with_match_any_off_a_known_tag_without_a_full_match_is_refused_even_with_a_default(match_all: Rig) -> None:
    """`tag_filtering_match_any: false`; Bedrock is tagged `premium` and Vertex `default`. The request carries
    `premium` and `eu`.

    No deployment carries both tags and `premium` is known to the group, so the request must be refused with HTTP
    401 and `Not allowed to access model due to tags configuration` instead of falling to the `default` Vertex
    deployment, and the peer receives nothing for either.
    """
    response: Final = chat(match_all.gateway, "tag-all-default", **_tags("premium", "eu"))
    assert response.status_code == 401, response.text
    assert TAGS_REFUSED in error_message(response), response.text
    assert match_all.peer.count("tad-bedrock") == 0
    assert match_all.peer.count("tad-vertex") == 0


def _switch_models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("azure", "tag-off", "toff-azure", base, tags=["premium"]),
        dep("bedrock", "tag-off", "toff-bedrock", base, tags=["standard"]),
        dep("vertex", "tag-group-on", "ton-vertex", base, tags=["premium"], model_info={"enable_tag_filtering": True}),
        dep("anthropic", "tag-group-on", "ton-anthropic", base, tags=["standard"]),
    )


def test_tag_switches_when_the_router_leaves_tag_filtering_off(tmp_path: Path) -> None:
    """A proxy with `enable_tag_filtering` unset. `tag-off` has an Azure deployment tagged `premium` and a Bedrock one
    tagged `standard`; `tag-group-on` sets `model_info.enable_tag_filtering: true` on its Vertex deployment.

    Requests tagged `premium`: to `tag-off` they reach both providers over forty requests (at least five each),
    because tags are ignored; the same with `enable_tag_filtering: true` in the request body still reach both,
    because the proxy drops that body field; to `tag-group-on` all twenty reach Vertex, because one deployment's
    `model_info` switches filtering on for its whole group; and with a key whose saved `router_settings` set
    `enable_tag_filtering: true`, twenty requests to `tag-off` all reach Azure.
    """
    settings: Final[dict[str, JsonValue]] = {"routing_strategy": "simple-shuffle", "num_retries": 0}
    with strat_rig(tmp_path, _switch_models, settings) as rig:
        ignored: Final = served_many(rig.gateway, "tag-off", 40, **_tags("premium"))
        assert ignored["toff-azure"] >= 5 and ignored["toff-bedrock"] >= 5, ignored
        body_switch: Final = served_many(rig.gateway, "tag-off", 40, enable_tag_filtering=True, **_tags("premium"))
        assert body_switch["toff-azure"] >= 5 and body_switch["toff-bedrock"] >= 5, body_switch
        assert served_many(rig.gateway, "tag-group-on", 20, **_tags("premium")) == {"ton-vertex": 20}
        with rig.gateway.scenario() as scenario:
            key: Final = scenario.key(router_settings={"enable_tag_filtering": True})
            assert served_many(rig.gateway, "tag-off", 20, key=key, **_tags("premium")) == {"toff-azure": 20}
