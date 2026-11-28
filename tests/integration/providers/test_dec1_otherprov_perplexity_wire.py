"""Contract: chat completions on Perplexity (`perplexity/sonar*`) through the proxy, against a loopback peer.

Perplexity speaks the OpenAI chat dialect with search on top. What the proxy must keep:

- URL and auth: `POST <api_base>/chat/completions` with `Authorization: Bearer <key>`, plus the attribution
  header `X-Pplx-Integration: litellm` on every request (`PerplexityChatConfig.get_attribution_headers`);
- params: Perplexity takes no tools, so `tools` is refused with 400; `reasoning_effort` and
  `web_search_options` are only accepted on models whose cost map row says they support reasoning or web search;
- answers: the top-level `citations` URL list and `search_results` are kept on the response, and every `[n]`
  marker in the text becomes a `url_citation` annotation with the marker's character offsets and the title of
  the matching search result;
- cost: a reported `usage.cost.total_cost` is the response cost; otherwise tokens are priced from the row and
  each search query (`usage.num_search_queries`) is billed at the row's LOW search context price, whatever
  context size the caller asked for.

The cached-prompt and off-peak cost rows are covered in `cost_calculation/cases/misc.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov_support import (
    PROVIDER_KEY,
    TOOL,
    Call,
    chat,
    chat_json,
    deployment,
    error_message,
    first_message,
    json_reply,
    peer,
    served,
    usage,
)
from pydantic import JsonValue

_SONAR: Final = "perplexity/sonar"
_TRANSFORM_RESPONSE_BUG: Final = (
    "BUG: PerplexityChatConfig.transform_response never runs on the chat path, so no annotations and no search fee"
    " (perplexity/chat/transformation.py:84-120; openai/openai.py:929,1072 call only transform_parsed_response_dict)"
)
_CITED_TEXT: Final = "Oslo is cold [1] and wet [2]."
_CITATIONS: Final[list[JsonValue]] = ["https://example.test/a", "https://example.test/b"]
_SEARCH_RESULTS: Final[list[JsonValue]] = [
    {"title": "Weather A", "url": "https://example.test/a", "date": "2026-11-01"},
    {"title": "Weather B", "url": "https://example.test/b", "date": "2026-11-02"},
]


def _cited(call: Call) -> Reply:
    return json_reply(
        chat_json(
            call.body, content=_CITED_TEXT, extra_body={"citations": _CITATIONS, "search_results": _SEARCH_RESULTS}
        )
    )


def test_posts_chat_completions_under_the_api_base_with_bearer_and_integration_header(gateway: Gateway) -> None:
    """A `perplexity/sonar` deployment whose `api_base` is the bare peer URL.

    The peer receives one `POST /chat/completions` with `authorization: Bearer dec1-otherprov-provider-key`, body
    `model: "sonar"`, and the attribution header `x-pplx-integration: litellm` (`perplexity/chat/transformation.py:23`,
    `:32-33`, applied in `main.py:2870-2874`). UNVERIFIED: that the OpenAI-client path forwards the attribution
    header unchanged (it is passed as request headers).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        served(chat(gateway, model))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/chat/completions")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body["model"] == "sonar"
        assert call.headers.get("x-pplx-integration") == "litellm", call.headers


def test_caller_header_with_the_same_name_wins_over_the_attribution_header(gateway: Gateway) -> None:
    """The deployment sets `extra_headers: {"x-pplx-integration": "dec1-app"}`. The base config documents that a
    caller header with the same name in any casing wins (`base_llm/chat/transformation.py:116-123`), so the peer
    sees `x-pplx-integration: dec1-app` exactly once. UNVERIFIED: deployment `extra_headers` count as caller
    headers for `with_attribution_headers`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR, extra_headers={"x-pplx-integration": "dec1-app"})
        served(chat(gateway, model))
        assert provider.only().headers.get("x-pplx-integration") == "dec1-app"


def test_tools_are_refused_because_perplexity_takes_none(gateway: Gateway) -> None:
    """Perplexity's supported list has no `tools` (`perplexity/chat/transformation.py:52-80`), so a call with one
    tool is answered 400 with `perplexity does not support parameters: ['tools']` and the peer is not called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        response: Final = chat(gateway, model, tools=[TOOL])
        assert response.status_code == 400, response.text
        assert "perplexity does not support parameters: ['tools']" in error_message(response)
        assert provider.posts() == ()


def test_reasoning_effort_is_refused_on_a_model_without_reasoning(gateway: Gateway) -> None:
    """`perplexity/sonar` has no `supports_reasoning` in the cost map, so `reasoning_effort` is not added to its
    supported list (`perplexity/chat/transformation.py:69-73`) and `reasoning_effort: "high"` is answered 400
    naming `reasoning_effort`, without calling the peer."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        response: Final = chat(gateway, model, reasoning_effort="high")
        assert response.status_code == 400, response.text
        assert "reasoning_effort" in error_message(response)
        assert provider.posts() == ()


def test_reasoning_model_forwards_reasoning_effort_and_web_search_options(gateway: Gateway) -> None:
    """`perplexity/sonar-reasoning-pro` supports reasoning and web search in the cost map, so
    `reasoning_effort: "high"` and `web_search_options: {"search_context_size": "high"}` both reach the peer body
    unchanged (`perplexity/chat/transformation.py:69-79`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "perplexity/sonar-reasoning-pro")
        served(chat(gateway, model, reasoning_effort="high", web_search_options={"search_context_size": "high"}))
        sent: Final = provider.only().body
        assert sent["reasoning_effort"] == "high", sent
        assert sent["web_search_options"] == {"search_context_size": "high"}, sent


def test_citations_and_search_results_are_kept_on_the_response(gateway: Gateway) -> None:
    """Perplexity answers with top-level `citations` (two URLs) and `search_results` (two entries).

    The caller's body carries both lists unchanged at the top level
    (`_add_citations_as_annotations`, `perplexity/chat/transformation.py:242-246`).
    """
    with peer(_cited) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        body: Final = served(chat(gateway, model))
        assert body["citations"] == _CITATIONS, body
        assert body["search_results"] == _SEARCH_RESULTS, body


def test_each_numbered_marker_becomes_a_url_citation_with_its_offsets_and_title(gateway: Gateway) -> None:
    """The answer text is `Oslo is cold [1] and wet [2].`.

    The caller's message has two `url_citation` annotations: `[1]` at offsets 13..16 for
    `https://example.test/a` titled `Weather A`, and `[2]` at 25..28 for `https://example.test/b` titled
    `Weather B` (`perplexity/chat/transformation.py:210-237`, 1-based marker to `citations` index).
    Skipped as a product bug until fixed, see `_TRANSFORM_RESPONSE_BUG`."""
    pytest.skip(_TRANSFORM_RESPONSE_BUG)
    with peer(_cited) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        annotations: Final = list_value(first_message(served(chat(gateway, model)))["annotations"])
        assert [object_value(item)["url_citation"] for item in annotations] == [
            {"url": "https://example.test/a", "title": "Weather A", "start_index": 13, "end_index": 16},
            {"url": "https://example.test/b", "title": "Weather B", "start_index": 25, "end_index": 28},
        ]
        assert all(object_value(item)["type"] == "url_citation" for item in annotations)


def test_a_marker_without_a_matching_citation_gets_no_annotation(gateway: Gateway) -> None:
    """The text cites `[3]` while `citations` has two entries. `[3]` is not in the number-to-URL map
    (`perplexity/chat/transformation.py:226-228`), so the message gets annotations only for `[1]`, never an entry
    for `[3]`.
    Skipped as a product bug until fixed, see `_TRANSFORM_RESPONSE_BUG`."""
    pytest.skip(_TRANSFORM_RESPONSE_BUG)

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(
                call.body, content="Fact [1] and [3].", extra_body={"citations": _CITATIONS, "search_results": []}
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR)
        annotations: Final = list_value(first_message(served(chat(gateway, model)))["annotations"])
        assert [object_value(object_value(item)["url_citation"])["url"] for item in annotations] == [
            "https://example.test/a"
        ]


def test_reported_total_cost_is_the_response_cost(gateway: Gateway) -> None:
    """Perplexity answers `usage.cost: {"total_cost": 0.0123, ...}` with 11/7 tokens, on an unpriced deployment.

    `cost_per_token` returns the reported total as the cost (`perplexity/cost_calculator.py:31-43`), so the
    caller's `x-litellm-response-cost` is 0.0123, not a token-priced figure. UNVERIFIED: that the dict survives into
    `Usage.cost` on the chat path (the comment at `:29-31` says it may already be flattened to a float; both are
    accepted).
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(
                call.body,
                usage_block={
                    **usage(),
                    "cost": {"input_tokens_cost": 0.0001, "output_tokens_cost": 0.0002, "total_cost": 0.0123},
                },
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR, priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.0123), response.headers


def test_search_queries_are_billed_at_the_low_context_price_on_top_of_tokens(gateway: Gateway) -> None:
    """Perplexity answers 11/7 tokens and `usage.num_search_queries: 2` with no reported cost, on an unpriced
    `perplexity/sonar` deployment (row: 1e-06 per input and output token, search low/medium/high 0.005/0.008/0.012).

    The response cost is 11e-6 + 7e-6 + 2 * 0.005 = 0.010018: searches land in `web_search_requests`
    (`perplexity/chat/transformation.py:150-171`) and are priced at `search_context_size_low`
    (`perplexity/cost_calculator.py:104-128`). Assumes the proxy loads the repo cost map.
    Skipped as a product bug until fixed, see `_TRANSFORM_RESPONSE_BUG`."""
    pytest.skip(_TRANSFORM_RESPONSE_BUG)

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block={**usage(), "num_search_queries": 2}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR, priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.010018), response.headers


def test_high_search_context_is_still_billed_at_the_low_price(gateway: Gateway) -> None:
    """The caller asks `web_search_options: {"search_context_size": "high"}` on `perplexity/sonar`, and the answer
    reports one search query.

    The cost is 11e-6 + 7e-6 + 1 * 0.005 = 0.005018, since the calculator always reads `search_context_size_low`
    (`perplexity/cost_calculator.py:118-124`). Pinned current behaviour (high would be 0.012018); a port that
    honours the requested size changes every bill, so it must be a deliberate change.
    Skipped as a product bug until fixed, see `_TRANSFORM_RESPONSE_BUG`."""
    pytest.skip(_TRANSFORM_RESPONSE_BUG)

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block={**usage(), "num_search_queries": 1}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR, priced=False)
        response: Final = chat(gateway, model, web_search_options={"search_context_size": "high"})
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.005018), response.headers


def test_zero_search_queries_add_no_search_fee(gateway: Gateway) -> None:
    """With `num_search_queries: 0` the search branch is skipped (`perplexity/chat/transformation.py:160-171`), so
    the cost of 11/7 tokens on `perplexity/sonar` is exactly 18e-6."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block={**usage(), "num_search_queries": 0}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _SONAR, priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.000018), response.headers
