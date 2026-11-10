"""`max_fallbacks`, `fallback_on_unsupported_params: false` and the `"*"` fallback entry.

One owned proxy runs with `num_retries: 0`, cooldowns disabled, `max_fallbacks: 1` and
`fallback_on_unsupported_params: false`. Every group has one deployment.

`max_fallbacks` limits how deep fallbacks nest, not how many targets one list may hold: the limit is checked
once before a group's list is walked, and every target in that list is then tried. With a limit of 1 a
fallback group's own fallback list is never followed. `fallback_on_unsupported_params: false` makes a plain
provider 400 return to the caller instead of falling back, while context window errors keep falling back. A
`{"*": [...]}` entry in `fallbacks` serves every group that has no entry of its own, the same as
`default_fallbacks`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    MESSAGE_PREFIX,
    Rig,
    chat,
    content_of,
    deployment,
    error_message,
    routing_rig,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "max_fallbacks": 1,
    "fallback_on_unsupported_params": False,
    "fallbacks": [
        {"limit-chain-a": ["limit-chain-b"]},
        {"limit-chain-b": ["limit-chain-c"]},
        {"limit-list": ["limit-list-dead-one", "limit-list-dead-two", "limit-list-ok"]},
        {"limit-400": ["limit-400-backup"]},
        {"limit-ctx": ["limit-ctx-backup"]},
        {"limit-500": ["limit-500-backup"]},
        {"*": ["limit-star"]},
    ],
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("limit-chain-a", "status-500-limita", api_base),
        deployment("limit-chain-b", "status-503-limitb", api_base),
        deployment("limit-chain-c", "ok-limitc", api_base),
        deployment("limit-list", "status-500-limitlist", api_base),
        deployment("limit-list-dead-one", "status-503-limitdeadone", api_base),
        deployment("limit-list-dead-two", "status-503-limitdeadtwo", api_base),
        deployment("limit-list-ok", "ok-limitlist", api_base),
        deployment("limit-400", "status-400-limit", api_base),
        deployment("limit-400-backup", "ok-limit400backup", api_base),
        deployment("limit-ctx", "context-limit", api_base),
        deployment("limit-ctx-backup", "ok-limitctxbackup", api_base),
        deployment("limit-500", "status-500-limit", api_base),
        deployment("limit-500-backup", "ok-limit500backup", api_base),
        deployment("limit-nochain", "status-500-limitnochain", api_base),
        deployment("limit-star", "ok-limitstar", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-fallback-limits"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_max_fallbacks_one_does_not_follow_a_fallback_groups_own_fallback_list(rig: Rig) -> None:
    """`max_fallbacks: 1`; `limit-chain-a` (500) falls back to `limit-chain-b` (503), which has its own entry
    pointing at the healthy `limit-chain-c`.

    The caller gets HTTP 500 with the message of `limit-chain-a`. The upstream saw a then b, once each, and the
    model behind `limit-chain-c` received nothing. With the default limit of 5 the same chain is served by c.
    """
    response: Final = chat(rig.gateway, "limit-chain-a")
    assert response.status_code == 500, response.text
    assert error_message(response).startswith(MESSAGE_PREFIX[500] + "scripted 500 from status-500-limita")
    assert rig.upstream.order("status-500-limita", "status-503-limitb") == ("status-500-limita", "status-503-limitb")
    assert rig.upstream.count("ok-limitc") == 0


def test_max_fallbacks_one_still_tries_every_target_of_one_fallback_list(rig: Rig) -> None:
    """`max_fallbacks: 1`; `limit-list` (500) has three fallback targets: two that answer 503 and a healthy one.

    The caller gets 200 with the third target's content, and the upstream saw the primary and all three
    targets once each, in list order. The limit bounds nesting depth, not list length.
    """
    response: Final = chat(rig.gateway, "limit-list")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-limitlist"
    models: Final = ("status-500-limitlist", "status-503-limitdeadone", "status-503-limitdeadtwo", "ok-limitlist")
    assert rig.upstream.order(*models) == models


def test_fallback_on_unsupported_params_false_returns_a_provider_400_without_falling_back(rig: Rig) -> None:
    """`fallback_on_unsupported_params: false`; `limit-400` answers 400 and has a fallback entry.

    The caller gets HTTP 400 with a message starting `litellm.BadRequestError: OpenAIException - `, and the
    fallback's upstream model receives nothing. With the default of true the same request is served by the
    fallback. UNVERIFIED: that the proxy marks an upstream 400 as naming its provider, which
    `is_bad_request_without_fallback` in `litellm-rust/crates/router-core/src/call.rs` requires.
    """
    response: Final = chat(rig.gateway, "limit-400")
    assert response.status_code == 400, response.text
    assert error_message(response).startswith(MESSAGE_PREFIX[400]), response.text
    assert rig.upstream.count("status-400-limit") == 1
    assert rig.upstream.count("ok-limit400backup") == 0


def test_fallback_on_unsupported_params_false_still_falls_back_on_a_context_window_error(rig: Rig) -> None:
    """`fallback_on_unsupported_params: false`; `limit-ctx` answers the 400 context length body and has a
    regular fallback entry.

    The caller gets 200 with the fallback's content: a context window error is a bad request that keeps its
    fallback.
    """
    response: Final = chat(rig.gateway, "limit-ctx")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-limitctxbackup"


def test_fallback_on_unsupported_params_false_still_falls_back_on_a_500(rig: Rig) -> None:
    """`fallback_on_unsupported_params: false`; `limit-500` answers 500 and has a fallback entry.

    The caller gets 200 with the fallback's content and `x-litellm-attempted-fallbacks: 1`.
    """
    response: Final = chat(rig.gateway, "limit-500")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-limit500backup"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_a_star_entry_in_fallbacks_serves_a_group_without_its_own_entry(rig: Rig) -> None:
    """`fallbacks` holds `{"*": ["limit-star"]}`; `limit-nochain` answers 500 and has no entry of its own.

    The caller gets 200 with `limit-star`'s content.
    """
    response: Final = chat(rig.gateway, "limit-nochain")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-limitstar"
