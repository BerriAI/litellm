"""The response headers that reveal the router's choice, and the pre-call checks that narrow it, across providers.

Every answer the proxy relays from a deployment names it: `x-litellm-model-id` is the deployment's
`model_info.id`, `x-litellm-model-api-base` its API base without query string, and `x-litellm-model-name` the
deployment's `litellm_params.model` (`get_custom_headers` in `litellm/proxy/common_request_processing.py:2857`,
fed by the `deployment` metadata the router writes in `litellm/router_rust/invoke/shaping.py:486`). Clients and
dashboards use these to see which provider served, so the values must survive the Rust port for every provider.

With `enable_pre_call_checks: true` the core drops, before the strategy runs, every deployment whose context
window (`model_info.max_input_tokens`) the prompt exceeds (`litellm/rust/crates/router-core/src/precall.rs`), and
answers 400 `Context Window exceeded` naming each deployment's window when none fits. A deployment that declares no
window and whose model the cost map does not know is never dropped. With
`optional_pre_call_checks: ["enforce_model_rate_limits"]` a deployment over its `rpm` this minute raises a 429 for
that attempt, which a retry moves to a sibling.

One proxy serves the header and context window tests; the rate limit test boots its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_reliab_support import (
    ANTHROPIC_MODEL,
    BEDROCK_MODEL,
    GEMINI_MODEL,
    OPENAI_MODEL,
    openai_error,
)
from tests.integration.routing.dec1_routestrat_support import (
    LONG_PROMPT,
    Rig,
    chat,
    content_of,
    dep,
    early_in_a_minute,
    error_message,
    ident,
    served,
    served_many,
    strat_rig,
    stream_chat,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

LONG: Final[list[JsonValue]] = [{"role": "user", "content": LONG_PROMPT}]
MODEL_NAMES: Final = {
    "openai": f"openai/{OPENAI_MODEL}",
    "azure": "azure/hp-azure",
    "bedrock": f"bedrock/converse/{BEDROCK_MODEL}",
    "vertex": f"vertex_ai/{GEMINI_MODEL}",
    "anthropic": f"anthropic/{ANTHROPIC_MODEL}",
}


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *(dep(provider, f"hdr-{provider}", f"hp-{provider}", base) for provider in MODEL_NAMES),
        dep("openai", "hdr-failover", "hf-openai", base, weight=1),
        dep("bedrock", "hdr-failover", "hf-bedrock", base),
        dep("azure", "hdr-error", "he-azure", base),
        dep("bedrock", "pc-window", "pw-bedrock", base, model_info={"max_input_tokens": 50}),
        dep("vertex", "pc-window", "pw-vertex", base, model_info={"max_input_tokens": 100000}),
        dep("anthropic", "pc-none-fit", "pnf-anthropic", base, model_info={"max_input_tokens": 50}),
        dep("azure", "pc-none-fit", "pnf-azure", base, model_info={"max_input_tokens": 60}),
        dep("azure", "pc-unknown-window", "puw-azure", base),
        dep("openai", "pc-window-weighted", "pww-openai", base, weight=100, model_info={"max_input_tokens": 50}),
        dep("anthropic", "pc-window-weighted", "pww-anthropic", base, model_info={"max_input_tokens": 100000}),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 1,
        "disable_cooldowns": True,
        "enable_pre_call_checks": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-headers"), _models, settings) as value:
        yield value


@pytest.mark.parametrize("provider", list(MODEL_NAMES))
def test_a_success_names_the_serving_deployment_for_every_provider(rig: Rig, provider: str) -> None:
    """One chat request to a single-deployment group on OpenAI, Azure, Bedrock, Vertex or Anthropic.

    The 200 must carry `x-litellm-model-id: dec1-routestrat-hp-<provider>`, an `x-litellm-model-api-base` that starts
    with the deployment's own `<peer>/hp-<provider>` base, and `x-litellm-model-name` equal to the deployment's
    `litellm_params.model` (for example `bedrock/converse/amazon.nova-pro-v1:0`). UNVERIFIED: for Bedrock, Vertex
    and Anthropic the api base header is the provider URL the handler built, which is only checked by prefix.
    """
    response: Final = chat(rig.gateway, f"hdr-{provider}")
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served hp-{provider}"
    assert response.headers["x-litellm-model-id"] == ident(f"hp-{provider}")
    assert response.headers["x-litellm-model-api-base"].startswith(f"{rig.peer.url}/hp-{provider}")
    assert response.headers["x-litellm-model-name"] == MODEL_NAMES[provider]


def test_the_api_base_header_drops_the_query_string(rig: Rig) -> None:
    """One request to the Azure group, whose provider URL carries `?api-version=2024-10-21`.

    `x-litellm-model-api-base` must not contain `?` or `api-version`: the header is cut at the query string so it
    never leaks query parameters.
    """
    response: Final = chat(rig.gateway, "hdr-azure")
    assert response.status_code == 200, response.text
    assert "?" not in response.headers["x-litellm-model-api-base"]
    assert "api-version" not in response.headers["x-litellm-model-api-base"]


@pytest.mark.parametrize("provider", ["openai", "bedrock", "anthropic"])
def test_a_streamed_answer_names_the_serving_deployment_in_its_headers(rig: Rig, provider: str) -> None:
    """A streamed chat request to the OpenAI, Bedrock or Anthropic group.

    The stream's response headers must carry `x-litellm-model-id: dec1-routestrat-hp-<provider>` before the body is
    read, and the streamed text contains `served hp-<provider>`.
    """
    headers, body = stream_chat(rig.gateway, f"hdr-{provider}")
    assert headers["x-litellm-model-id"] == ident(f"hp-{provider}")
    assert f"hp-{provider}" in body


@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "vertex"])
def test_messages_answers_name_the_serving_deployment(rig: Rig, provider: str) -> None:
    """A `POST /v1/messages` request to the Anthropic, Bedrock or Vertex group.

    The 200 must carry `x-litellm-model-id: dec1-routestrat-hp-<provider>`, so the Anthropic-native endpoint reveals
    the routing choice like chat completions do.
    """
    response: Final = rig.gateway.request(
        "POST",
        "/v1/messages",
        {"model": f"hdr-{provider}", "max_tokens": 16, "messages": [{"role": "user", "content": "which deployment"}]},
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == ident(f"hp-{provider}")


def test_after_a_retry_the_header_names_the_deployment_that_served_not_the_one_that_failed(rig: Rig) -> None:
    """`hdr-failover` has OpenAI (`weight: 1`, so it is picked first) and Bedrock; `num_retries: 1`. OpenAI answers
    its next request with a 500.

    The request must return 200 with text `served hf-bedrock`, `x-litellm-model-id: dec1-routestrat-hf-bedrock` and
    an `x-litellm-model-name` of the Bedrock model, after the peer saw OpenAI then Bedrock.
    """
    rig.peer.plan("hf-openai", openai_error(500, "scripted header failover", kind="server_error"))
    before: Final = len(rig.peer.order("hf-openai", "hf-bedrock"))
    response: Final = chat(rig.gateway, "hdr-failover")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served hf-bedrock"
    assert response.headers["x-litellm-model-id"] == ident("hf-bedrock")
    assert response.headers["x-litellm-model-name"] == f"bedrock/converse/{BEDROCK_MODEL}"
    assert rig.peer.order("hf-openai", "hf-bedrock")[before:] == ("hf-openai", "hf-bedrock")


def test_an_upstream_error_still_names_the_deployment_that_failed(rig: Rig) -> None:
    """`hdr-error` has one Azure deployment, which answers its next two requests with a 400 (so the retry fails
    too).

    The caller gets HTTP 400 and the error response carries `x-litellm-model-id: dec1-routestrat-he-azure`, read
    from the logging object (`maybe_get_model_id`). UNVERIFIED: that the failure path finds the model id for a
    deployment-level error; see `common_request_processing.py:5258`.
    """
    failure: Final = openai_error(400, "scripted header error")
    rig.peer.plan("he-azure", failure, failure)
    response: Final = chat(rig.gateway, "hdr-error")
    assert response.status_code == 400, response.text
    assert response.headers.get("x-litellm-model-id") == ident("he-azure"), response.headers


def test_a_long_prompt_skips_the_provider_whose_window_is_too_small(rig: Rig) -> None:
    """`pc-window` has Bedrock with `max_input_tokens: 50` and Vertex with 100000; the prompt is about 1200 tokens.

    Ten requests must all be served by Vertex and Bedrock must receive none of them, while short prompts still
    reach both providers (at least three each over thirty).
    """
    before: Final = rig.peer.count("pw-bedrock")
    assert served_many(rig.gateway, "pc-window", 10, messages=LONG) == {"pw-vertex": 10}
    assert rig.peer.count("pw-bedrock") == before
    counts: Final = served_many(rig.gateway, "pc-window", 30)
    assert counts["pw-bedrock"] >= 3 and counts["pw-vertex"] >= 3, counts


def test_the_context_window_filter_runs_before_weights(rig: Rig) -> None:
    """`pc-window-weighted` has OpenAI with `weight: 100` and a window of 50 tokens, and Anthropic with no weight
    and a window of 100000. The prompt is about 1200 tokens.

    Ten requests must all reach Anthropic: the pre-call checks remove OpenAI before the shuffle reads weights, so a
    weight cannot push a request onto a deployment that cannot hold it.
    """
    assert served_many(rig.gateway, "pc-window-weighted", 10, messages=LONG) == {"pww-anthropic": 10}


def test_when_no_window_fits_the_400_names_every_deployments_window(rig: Rig) -> None:
    """`pc-none-fit` has Anthropic with a window of 50 tokens and Azure with 60; the prompt is about 1200 tokens.

    The caller gets HTTP 400 whose message contains `Context Window exceeded`, `Max Input Tokens=50` and
    `Max Input Tokens=60` (one `Model=..., Max Input Tokens=..., Got=...` entry per deployment, `precall.rs:104`),
    and neither provider receives the request.
    """
    response: Final = chat(rig.gateway, "pc-none-fit", messages=LONG)
    assert response.status_code == 400, response.text
    message: Final = error_message(response)
    assert "Context Window exceeded" in message, message
    assert "Max Input Tokens=50" in message, message
    assert "Max Input Tokens=60" in message, message
    assert f"Model=anthropic/{ANTHROPIC_MODEL}" in message, message
    assert rig.peer.count("pnf-anthropic") == 0
    assert rig.peer.count("pnf-azure") == 0


def test_a_deployment_with_no_known_window_is_kept_for_any_prompt(rig: Rig) -> None:
    """`pc-unknown-window` has one Azure deployment `azure/puw-azure`, which sets no `max_input_tokens` and whose
    model is not in the cost map. The prompt is about 1200 tokens.

    The request must be served with 200: an unknown window is never treated as too small.
    """
    assert served(rig.gateway, "pc-unknown-window", messages=LONG) == "puw-azure"


def test_enforce_model_rate_limits_moves_traffic_to_the_sibling_once_a_deployment_used_its_rpm(tmp_path: Path) -> None:
    """A proxy with `optional_pre_call_checks: ["enforce_model_rate_limits"]` and `num_retries: 2`. In `pc-rpm`
    Bedrock has `rpm: 2` and `weight: 1`, so the shuffle always picks it first, and Vertex has no limit.

    Starting early in a minute, eight sequential requests must all return 200. Bedrock serves exactly two of them;
    on the others its attempt is refused by the check (`Model rate limit exceeded. RPM limit=2`) without an
    upstream call, and the retry is served by Vertex.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("bedrock", "pc-rpm", "pr-bedrock", base, rpm=2, weight=1),
            dep("vertex", "pc-rpm", "pr-vertex", base),
        )

    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 2,
        "disable_cooldowns": True,
        "optional_pre_call_checks": ["enforce_model_rate_limits"],
    }
    with strat_rig(tmp_path, models, settings) as own:
        early_in_a_minute()
        counts: Final = served_many(own.gateway, "pc-rpm", 8)
        assert counts == {"pr-bedrock": 2, "pr-vertex": 6}, counts
        assert own.peer.count("pr-bedrock") == 2
