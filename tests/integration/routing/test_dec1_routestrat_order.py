"""`order` across providers: only the lowest order level serves, and a failure walks up the levels.

Selection keeps only the deployments whose `litellm_params.order` equals the lowest order present
(`select.rs:688`), so a deployment with no `order` next to ordered siblings is dropped from every pick: it never
serves, not even after the ordered levels fail (the known ruling these tests pin). When a pick at one level fails,
the call loop walks the group's distinct orders in ascending order (`call.rs:799`), then the configured
`fallbacks`. A content policy error never walks the order levels, and neither does a context window error the
group has a `context_window_fallbacks` entry for (`call.rs:789`). When every level failed, the caller gets the
last level's error.

Within one level the router's strategy (simple-shuffle here) still decides, weights included. Deployments are on
Azure OpenAI, Bedrock, OpenAI, Vertex AI and Anthropic, each failing with its own provider's error shape. Retries
are off and cooldowns disabled, so every walk is visible as one call per level.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.wire import Reply
from tests.integration.routing.dec1_reliab_support import (
    BEDROCK_INTERNAL_TEXT,
    azure_rate_limit,
    bedrock_error,
    google_error,
    openai_error,
)
from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    chat,
    dep,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

SAMPLES: Final = 200


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("azure", "ord-three", "o3-azure", base, order=1),
        dep("bedrock", "ord-three", "o3-bedrock", base, order=2),
        dep("openai", "ord-three", "o3-openai", base, order=3),
        dep("vertex", "ord-gaps", "og-vertex", base, order=10),
        dep("anthropic", "ord-gaps", "og-anthropic", base, order=5),
        dep("openai", "ord-gaps", "og-openai", base, order=1),
        dep("vertex", "ord-unordered", "ou-vertex", base, order=1),
        dep("anthropic", "ord-unordered", "ou-anthropic", base),
        dep("openai", "ord-lone-level", "oll-openai", base, order=1),
        dep("anthropic", "ord-lone-level", "oll-anthropic", base),
        dep("openai", "ord-last-error", "ole-openai", base, order=1),
        dep("azure", "ord-last-error", "ole-azure", base, order=2),
        dep("bedrock", "ord-last-error", "ole-bedrock", base),
        dep("bedrock", "ord-level-weights", "olw-bedrock", base, order=1, weight=3),
        dep("vertex", "ord-level-weights", "olw-vertex", base, order=1, weight=1),
        dep("openai", "ord-level-weights", "olw-openai", base, order=2, weight=100),
        dep("azure", "ord-then-fallback", "otf-azure", base, order=1),
        dep("vertex", "ord-then-fallback", "otf-vertex", base, order=2),
        dep("anthropic", "ord-backup", "ob-anthropic", base),
        dep("openai", "ord-policy", "op-openai", base, order=1),
        dep("bedrock", "ord-policy", "op-bedrock", base, order=2),
        dep("azure", "ord-recover", "or-azure", base, order=1),
        dep("openai", "ord-recover", "or-openai", base, order=2),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
        "fallbacks": [{"ord-then-fallback": ["ord-backup"]}],
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-order"), _models, settings) as value:
        yield value


def _openai_500() -> Reply:
    return openai_error(500, "scripted order failure", kind="server_error")


def test_a_healthy_first_level_takes_every_request_across_three_levels(rig: Rig) -> None:
    """Azure has `order: 1`, Bedrock `order: 2` and OpenAI `order: 3`, all healthy.

    Twenty requests must all be served by Azure; Bedrock and OpenAI receive nothing.
    """
    assert served_many(rig.gateway, "ord-three", 20) == {"o3-azure": 20}
    assert rig.peer.count("o3-bedrock") == 0
    assert rig.peer.count("o3-openai") == 0


def test_failures_walk_three_levels_in_order_across_providers(rig: Rig) -> None:
    """Azure (`order: 1`) answers its next request with an Azure 429 and Bedrock (`order: 2`) with a Bedrock 500
    `InternalServerException`; OpenAI (`order: 3`) is healthy.

    One request must return 200 served by OpenAI, with `x-litellm-model-id` naming it, after the peer saw exactly
    one call to Azure, then one to Bedrock, then one to OpenAI, in that order.
    """
    rig.peer.plan("o3-azure", azure_rate_limit())
    rig.peer.plan("o3-bedrock", bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT))
    before: Final = len(rig.peer.order("o3-azure", "o3-bedrock", "o3-openai"))
    assert served(rig.gateway, "ord-three") == "o3-openai"
    assert rig.peer.order("o3-azure", "o3-bedrock", "o3-openai")[before:] == ("o3-azure", "o3-bedrock", "o3-openai")


def test_order_values_need_not_be_consecutive_and_are_walked_ascending(rig: Rig) -> None:
    """The group lists Vertex with `order: 10`, Anthropic with `order: 5` and OpenAI with `order: 1`.

    While healthy, ten requests go to OpenAI. When OpenAI answers one request with a 500, that request must
    reach Anthropic (order 5) next and return 200 from it; Vertex (order 10) is not called. List order and gaps
    between values do not matter, only the ascending numeric order.
    """
    assert served_many(rig.gateway, "ord-gaps", 10) == {"og-openai": 10}
    rig.peer.plan("og-openai", _openai_500())
    assert served(rig.gateway, "ord-gaps") == "og-anthropic"
    assert rig.peer.count("og-vertex") == 0


def test_a_deployment_without_order_never_serves_next_to_an_ordered_sibling(rig: Rig) -> None:
    """Vertex has `order: 1`; Anthropic in the same group sets no `order`.

    Thirty requests must all be served by Vertex and Anthropic must receive nothing: the order filter keeps only
    deployments whose order equals the lowest one, so an unordered deployment is dropped.
    """
    assert served_many(rig.gateway, "ord-unordered", 30) == {"ou-vertex": 30}
    assert rig.peer.count("ou-anthropic") == 0


def test_a_failing_single_order_level_does_not_reach_the_unordered_sibling(rig: Rig) -> None:
    """OpenAI has `order: 1` and answers its next request with a 500; Anthropic in the same group has no order.

    The request must fail with HTTP 500 and Anthropic must receive nothing: with only one distinct order there is
    no level to walk to, and the unordered deployment is never a candidate.
    """
    rig.peer.plan("oll-openai", _openai_500())
    response: Final = chat(rig.gateway, "ord-lone-level")
    assert response.status_code == 500, response.text
    assert rig.peer.count("oll-anthropic") == 0


def test_when_every_level_fails_the_caller_gets_the_last_levels_error(rig: Rig) -> None:
    """OpenAI (`order: 1`) answers with a 500 and Azure (`order: 2`) with an Azure 429; Bedrock in the group has no
    order.

    The caller must get HTTP 429, the error of the last level walked, and Bedrock must receive nothing. This
    differs from configured fallbacks, where the caller gets the first group's error (see the routing index).
    """
    rig.peer.plan("ole-openai", _openai_500())
    rig.peer.plan("ole-azure", azure_rate_limit())
    response: Final = chat(rig.gateway, "ord-last-error")
    assert response.status_code == 429, response.text
    assert rig.peer.order("ole-openai", "ole-azure")[-2:] == ("ole-openai", "ole-azure")
    assert rig.peer.count("ole-bedrock") == 0


def test_weights_split_traffic_inside_the_lowest_order_level(rig: Rig) -> None:
    """At `order: 1` Bedrock has `weight: 3` and Vertex `weight: 1`; OpenAI at `order: 2` has `weight: 100`.

    Over 200 requests OpenAI must serve none, Bedrock between 120 and 180 (expected 150) and Vertex the rest:
    the order filter runs before the shuffle, and the shuffle only sees one level.
    """
    counts: Final = served_many(rig.gateway, "ord-level-weights", SAMPLES)
    assert counts["olw-openai"] == 0, counts
    assert 120 <= counts["olw-bedrock"] <= 180, counts
    assert counts["olw-vertex"] >= 20, counts


def test_order_levels_are_walked_before_the_configured_fallback_group(rig: Rig) -> None:
    """`ord-then-fallback` has Azure at `order: 1` and Vertex at `order: 2`, and falls back to `ord-backup`
    (Anthropic). Azure answers with an Azure 429 and Vertex with a Google 503 `UNAVAILABLE`.

    The request must return 200 served by Anthropic, after the peer saw Azure, then Vertex, then Anthropic.
    """
    rig.peer.plan("otf-azure", azure_rate_limit())
    rig.peer.plan("otf-vertex", google_error(503, "UNAVAILABLE", "The service is currently unavailable."))
    before: Final = len(rig.peer.order("otf-azure", "otf-vertex", "ob-anthropic"))
    assert served(rig.gateway, "ord-then-fallback") == "ob-anthropic"
    assert rig.peer.order("otf-azure", "otf-vertex", "ob-anthropic")[before:] == (
        "otf-azure",
        "otf-vertex",
        "ob-anthropic",
    )


def test_a_content_policy_refusal_does_not_walk_to_the_next_order_level(rig: Rig) -> None:
    """OpenAI (`order: 1`) answers with OpenAI's 400 `content_policy_violation`; Bedrock is at `order: 2`.

    The caller must get HTTP 400 and Bedrock must receive nothing: a content policy error skips the order walk
    (`call.rs:789`), because another deployment of the same model would refuse the same content.
    """
    rig.peer.plan(
        "op-openai",
        openai_error(400, "Your request was rejected by the safety system.", code="content_policy_violation"),
    )
    response: Final = chat(rig.gateway, "ord-policy")
    assert response.status_code == 400, response.text
    assert rig.peer.count("op-bedrock") == 0


def test_the_first_level_serves_again_once_it_recovers(rig: Rig) -> None:
    """Azure (`order: 1`) fails one request with an Azure 429; OpenAI is at `order: 2`. Cooldowns are disabled.

    The failing request must be served by OpenAI, and the next request by Azure again: walking a level is per
    request and leaves no state behind when cooldowns are off.
    """
    rig.peer.plan("or-azure", azure_rate_limit())
    assert served(rig.gateway, "ord-recover") == "or-openai"
    assert served(rig.gateway, "ord-recover") == "or-azure"
