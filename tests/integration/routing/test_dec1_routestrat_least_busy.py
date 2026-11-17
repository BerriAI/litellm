"""`routing_strategy: least-busy` across providers: a request goes to the deployment with the fewest requests in flight.

The Rust core asks the Python host for the pick (`SelectStage::Strategy`), answered by
`LeastBusyLoggingHandler._select` in `litellm/router_strategy/least_busy.py:356`. It counts, per deployment, the
attempts started and not yet settled plus the picks reserved and not yet started, and takes the index of the
smallest count with `min` (`_least_busy_index`, line 192), so ties go to the deployment listed first. `weight`,
`rpm` and `tpm` play no part. A count goes up when an attempt starts (`log_pre_api_call`) and down when it
settles, on success or failure.

Each deployment here sits on a different provider (Bedrock, OpenAI, Vertex AI), and the peer can hold a label's
answer until the test releases it, so the test controls what is in flight. Retries are off and cooldowns are
disabled, so only the strategy decides.

Settlement runs in the success callback, which may finish just after the caller has its response, so tests that
send requests one after another allow one stale count.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration.routing.dec1_reliab_support import openai_error
from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    chat,
    content_of,
    dep,
    reported_strategy,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

GROUP: Final = "busy-three"
LABELS: Final = ("lb-bedrock", "lb-openai", "lb-vertex")


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("bedrock", GROUP, "lb-bedrock", base),
        dep("openai", GROUP, "lb-openai", base, weight=100),
        dep("vertex", GROUP, "lb-vertex", base, weight=100),
        dep("openai", "busy-fail", "lbf-openai", base),
        dep("anthropic", "busy-fail", "lbf-anthropic", base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "least-busy",
        "num_retries": 0,
        "disable_cooldowns": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-least-busy"), _models, settings) as value:
        yield value


def _reached(rig: Rig, before: int) -> tuple[str, ...]:
    return rig.peer.order(*LABELS)[before:]


def _stack_in_flight(
    rig: Rig, pool: ThreadPoolExecutor, requests: int
) -> tuple[tuple[str, ...], list[Future[httpx.Response]]]:
    """Send `requests` held requests one at a time, each once the previous one reached the peer."""
    before: Final = len(rig.peer.order(*LABELS))
    futures: Final[list[Future[httpx.Response]]] = []  # mutable-ok: one future per held request
    for sent in range(1, requests + 1):
        futures.append(pool.submit(chat, rig.gateway, GROUP))
        eventually(lambda: len(_reached(rig, before)), lambda seen, sent=sent: seen == sent, seconds=20)
    return _reached(rig, before), futures


def test_least_busy_is_reported_by_router_settings(rig: Rig) -> None:
    """The proxy is booted with `routing_strategy: least-busy`.

    `GET /router/settings` must report `current_values.routing_strategy` as `least-busy`.
    """
    assert reported_strategy(rig.gateway) == "least-busy"


def test_with_nothing_in_flight_the_first_listed_deployment_serves_and_weights_are_ignored(rig: Rig) -> None:
    """The group lists Bedrock first with no weight, then OpenAI and Vertex with `weight: 100`.

    Twenty requests sent one after another must be served mostly by Bedrock (at least 14) and never by Vertex,
    because a tie of zero in-flight requests goes to the first index and weights are not read. One stale
    in-flight count from a late settlement can push a single request to OpenAI.
    """
    counts: Final = served_many(rig.gateway, GROUP, 20)
    assert counts["lb-bedrock"] >= 14, counts
    assert counts["lb-vertex"] == 0, counts


def test_three_overlapping_requests_land_on_three_different_providers(rig: Rig) -> None:
    """Every deployment of the group holds its answer. The test sends a request, waits until the peer has it, and
    sends the next, three times.

    The three requests must reach Bedrock, OpenAI and Vertex in that order: each new request goes to the first
    deployment with nothing in flight. All three return 200 once released.
    """
    for label in LABELS:
        rig.peer.hold(label)
    with ThreadPoolExecutor(3) as pool:
        try:
            reached, futures = _stack_in_flight(rig, pool, 3)
        finally:
            for label in LABELS:
                rig.peer.release(label)
        assert reached == LABELS
        assert [future.result().status_code for future in futures] == [200, 200, 200]


def test_a_fourth_overlapping_request_goes_back_to_the_first_listed_deployment(rig: Rig) -> None:
    """All three deployments hold, and three requests are in flight, one on each.

    A fourth request sent while they are held must reach Bedrock, the first index among three counts of one.
    """
    for label in LABELS:
        rig.peer.hold(label)
    with ThreadPoolExecutor(4) as pool:
        try:
            reached, futures = _stack_in_flight(rig, pool, 4)
        finally:
            for label in LABELS:
                rig.peer.release(label)
        assert reached == (*LABELS, "lb-bedrock")
        assert all(future.result().status_code == 200 for future in futures)


def test_a_released_request_frees_its_deployment_for_the_next_one(rig: Rig) -> None:
    """Bedrock holds one request in flight while OpenAI and Vertex are idle; the test then releases Bedrock.

    While Bedrock is busy a new request must reach OpenAI. After the held request returns 200, sequential requests
    must reach Bedrock again within 20 seconds, which shows the success settled Bedrock's count.
    """
    rig.peer.hold("lb-bedrock")
    with ThreadPoolExecutor(1) as pool:
        before: Final = len(rig.peer.order(*LABELS))
        held: Final = pool.submit(chat, rig.gateway, GROUP)
        eventually(lambda: _reached(rig, before), lambda seen: seen == ("lb-bedrock",), seconds=20)
        try:
            assert served(rig.gateway, GROUP) == "lb-openai"
        finally:
            rig.peer.release("lb-bedrock")
        assert held.result().status_code == 200
    eventually(lambda: served(rig.gateway, GROUP), lambda label: label == "lb-bedrock", seconds=20)


def test_a_failed_request_also_frees_its_deployment(rig: Rig) -> None:
    """In the group `busy-fail` OpenAI is listed first and answers its next request with an OpenAI 500; Anthropic
    is listed second.

    The first request returns 500 from OpenAI. Sequential requests after it must reach OpenAI again within 20
    seconds: a failure settles the in-flight count like a success. A rewrite that only decremented on success
    would leave OpenAI looking busy and send every later request to Anthropic.
    """
    rig.peer.plan("lbf-openai", openai_error(500, "scripted least-busy failure", kind="server_error"))
    failed: Final = chat(rig.gateway, "busy-fail")
    assert failed.status_code == 500, failed.text
    assert rig.peer.count("lbf-openai") == 1
    eventually(lambda: served(rig.gateway, "busy-fail"), lambda label: label == "lbf-openai", seconds=20)


def test_the_answer_text_and_model_id_header_agree_under_least_busy(rig: Rig) -> None:
    """One request to the group with nothing held.

    The response text names the peer label that answered and `x-litellm-model-id` names the same deployment id
    (`dec1-routestrat-<label>`), for whichever provider served it.
    """
    response: Final = chat(rig.gateway, GROUP)
    assert response.status_code == 200, response.text
    label: Final = content_of(response).removeprefix("served ")
    assert label in LABELS
    assert response.headers["x-litellm-model-id"] == f"dec1-routestrat-{label}"
