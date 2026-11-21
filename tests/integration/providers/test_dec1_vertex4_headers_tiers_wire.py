"""Deployment headers (provisioned throughput) and served traffic tiers for Vertex AI.

Provisioned throughput. Vertex selects Provisioned Throughput or pay-as-you-go per request from the
`X-Vertex-AI-LLM-Request-Type` header (`dedicated` or `shared`). LiteLLM has no setting of its own for it
(no product code names the header), so operators put it in the deployment's `extra_headers`. Where those
headers go differs by route: Gemini merges them after the minted `Authorization`
(`_fresh_headers`, `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:3696` and `:3714`), so a
deployment `Authorization` even replaces the token; Claude chat merges them before its own `Authorization`
(`vertex_ai_partner_models/main.py:224-227`); Mistral and Codestral pass the OpenAI-like handler only the
labels header or nothing (`main.py:273-292`), so deployment headers never reach them.

Served tiers. Gemini on Vertex reports how a call was served in `usageMetadata.trafficType`, which is kept in
the response's hidden params (`vertex_and_google_ai_studio_gemini.py:3421-3423`, and `:4414-4416` for a
stream) and mapped to a billing tier in `_map_traffic_type_to_service_tier` (`litellm/cost_calculator.py:1230-1259`):
`ON_DEMAND_PRIORITY` is priority, `FLEX`/`BATCH`/`ON_DEMAND_FLEX` are flex, `ON_DEMAND` is base, matched after
`upper()`. Any other value, `PROVISIONED_THROUGHPUT` included, maps to `default` (`:1314`), which is base
pricing and beats the caller's requested tier (`_resolve_billable_service_tier`, `:1287-1294`). The AI Studio
twin of these rules is pinned in `test_dec1_gemini2_pricing_tiers.py`; this module pins them on `vertex_ai/`.
Rates for gemini-2.5-flash from `model_prices_and_context_window.json`: input 3e-07 / output 2.5e-06,
priority 5.4e-07 / 4.5e-06, flex 1.5e-07 / 1.25e-06.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini2_support import one_row, spend_of
from integration.providers.dec1_vertex3_support import LLAMA, MISTRAL, openai_echo
from integration.providers.dec1_vertex4_support import bearer, vertex_echo
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    gemini_reply,
    gemini_sse,
    json_of,
    marker,
    path_of,
    usage_metadata,
    vertex,
)
from pydantic import JsonValue

_REQUEST_TYPE: Final = "X-Vertex-AI-LLM-Request-Type"
_FLASH_BASE: Final = 1000 * 3e-07 + 100 * 2.5e-06
_FLASH_PRIORITY: Final = 1000 * 5.4e-07 + 100 * 4.5e-06
_FLASH_FLEX: Final = 1000 * 1.5e-07 + 100 * 1.25e-06


def _served(traffic_type: str) -> Callable[[Request], Reply]:
    """A Gemini responder reporting 1,000 prompt and 100 output tokens served as `traffic_type`."""
    usage: Final = usage_metadata(1000, 100, trafficType=traffic_type)

    def respond(request: Request) -> Reply:
        if path_of(request).endswith(":streamGenerateContent"):
            return gemini_sse(usage=usage)
        return gemini_reply(usage=usage)

    return respond


def _messages() -> dict[str, JsonValue]:
    return {"max_tokens": 32, "messages": [{"role": "user", "content": f"pt {marker()}"}]}


# ---------------------------------------------------------------- deployment headers


def test_a_deployment_request_type_header_reaches_gemini_generate_content(gateway: Gateway) -> None:
    """A Gemini deployment with `extra_headers: {"X-Vertex-AI-LLM-Request-Type": "dedicated"}` sends that
    header unchanged on `generateContent`, next to the minted bearer (`_fresh_headers`, `:3714`)."""
    with vertex(gateway, vertex_echo, extra_headers={_REQUEST_TYPE: "dedicated"}) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert call.headers.get(_REQUEST_TYPE.lower()) == "dedicated"
        assert bearer(call) in rig.peer.plan.issued


def test_a_deployment_request_type_header_reaches_a_streamed_gemini_call(gateway: Gateway) -> None:
    """The same header is sent on `:streamGenerateContent` (`_fresh_headers_async` is used by the streaming
    path, `vertex_and_google_ai_studio_gemini.py:3909`)."""
    with vertex(gateway, vertex_echo, extra_headers={_REQUEST_TYPE: "shared"}) as rig:
        response, _text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call).endswith(":streamGenerateContent")
        assert call.headers.get(_REQUEST_TYPE.lower()) == "shared"


def test_a_deployment_request_type_header_reaches_claude_chat_raw_predict(gateway: Gateway) -> None:
    """A Vertex Claude deployment's `extra_headers` reach `:rawPredict` on chat; the partner handler builds
    `{**headers, "Authorization": "Bearer <token>"}` (`vertex_ai_partner_models/main.py:224-227`)."""
    with vertex(gateway, vertex_echo, backend=CLAUDE, extra_headers={_REQUEST_TYPE: "dedicated"}) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert call.headers.get(_REQUEST_TYPE.lower()) == "dedicated"
        assert bearer(call) in rig.peer.plan.issued


def test_a_deployment_request_type_header_reaches_claude_on_messages(gateway: Gateway) -> None:
    """On `/v1/messages` the Vertex Claude request headers start from the provider headers and only gain
    `Authorization`, `content-type` and betas (`anthropic/messages/transformation.py:180-239`), so the
    deployment's `X-Vertex-AI-LLM-Request-Type` is sent. UNVERIFIED: assumes deployment `extra_headers`
    feed those provider headers on this route (as round 1 assumed for `Authorization`)."""
    with vertex(gateway, vertex_echo, backend=CLAUDE, extra_headers={_REQUEST_TYPE: "dedicated"}) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert rig.peer.one_call().headers.get(_REQUEST_TYPE.lower()) == "dedicated"


def test_a_deployment_request_type_header_reaches_mistral_raw_predict(gateway: Gateway) -> None:
    """A Mistral on Vertex deployment's `extra_headers` should reach `:rawPredict` like every other Vertex
    route, so provisioned throughput can be chosen for Mistral too."""
    pytest.skip(
        "BUG: VertexAIPartnerModels.completion passes the OpenAI-like handler `headers=None` (or only "
        "X-Vertex-AI-Labels) for Mistral and Codestral (vertex_ai_partner_models/main.py:273-292), dropping "
        "the deployment extra_headers that Llama and Claude keep"
    )
    with vertex(gateway, openai_echo, backend=MISTRAL, extra_headers={_REQUEST_TYPE: "dedicated"}) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_call().headers.get(_REQUEST_TYPE.lower()) == "dedicated"


def test_a_deployment_request_type_header_reaches_the_llama_openapi_route(gateway: Gateway) -> None:
    """A Llama-family MaaS deployment goes through `base_llm_http_handler.completion(headers=headers)`
    (`vertex_ai_partner_models/main.py:256-271`), so its `extra_headers` reach
    `.../endpoints/openapi/chat/completions`. UNVERIFIED: assumes the Vertex Llama config merges the passed
    headers into the outgoing request."""
    with vertex(gateway, openai_echo, backend=LLAMA, extra_headers={_REQUEST_TYPE: "dedicated"}) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_call().headers.get(_REQUEST_TYPE.lower()) == "dedicated"


def test_a_deployment_authorization_header_replaces_the_minted_token_on_gemini(gateway: Gateway) -> None:
    """Gemini spreads deployment `extra_headers` after `Authorization: Bearer <minted>` (`:3714`), so a
    deployment `Authorization` header is what Vertex receives, although a token was still minted. `/v1/messages`
    does the opposite (round 1 pins that the minted token wins there); a port that unifies the two must
    pick one on purpose."""
    with vertex(gateway, vertex_echo, extra_headers={"Authorization": "Bearer dec1-deployment-token"}) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_call().headers["authorization"] == "Bearer dec1-deployment-token"


# ---------------------------------------------------------------- served traffic tiers on vertex_ai/


def test_provisioned_throughput_traffic_is_billed_at_on_demand_rates(gateway: Gateway) -> None:
    """A call Vertex reports as `trafficType: PROVISIONED_THROUGHPUT` (1,000 in, 100 out) is billed like
    on-demand traffic: `1000 * 3e-07 + 100 * 2.5e-06 = 0.00055`. The value is not in the tier map, so
    `_served_service_tier` returns `default` (`cost_calculator.py:1314`). Surprise for operators who prepay
    PT: LiteLLM spend still counts list price."""
    with vertex(gateway, _served("PROVISIONED_THROUGHPUT")) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_BASE)


def test_provisioned_throughput_on_a_priority_request_is_billed_at_base_rates(gateway: Gateway) -> None:
    """A caller asks for `service_tier: "priority"` but Vertex serves it from provisioned throughput: the
    served `default` tier wins over the request (`_resolve_billable_service_tier`, `:1287-1294`) and the row
    is 0.00055, not the priority 0.00099."""
    with vertex(gateway, _served("PROVISIONED_THROUGHPUT")) as rig:
        assert rig.chat(service_tier="priority").status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_BASE)


def test_on_demand_priority_traffic_on_vertex_is_billed_at_priority_rates(gateway: Gateway) -> None:
    """On `vertex_ai/gemini-2.5-flash`, `trafficType: ON_DEMAND_PRIORITY` bills at priority rates
    `1000 * 5.4e-07 + 100 * 4.5e-06 = 0.00099` with no tier in the request (`cost_calculator.py:1232`)."""
    with vertex(gateway, _served("ON_DEMAND_PRIORITY")) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_PRIORITY)


def test_a_lower_case_flex_traffic_type_is_billed_at_flex_rates(gateway: Gateway) -> None:
    """The traffic type is upper-cased before the lookup (`cost_calculator.py:1258`), so `on_demand_flex`
    bills at flex rates `1000 * 1.5e-07 + 100 * 1.25e-06 = 0.000275`."""
    with vertex(gateway, _served("on_demand_flex")) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_FLEX)


def test_a_streamed_priority_traffic_type_is_billed_at_priority_rates(gateway: Gateway) -> None:
    """A streamed call whose final frame's `usageMetadata.trafficType` is `ON_DEMAND_PRIORITY` keeps it in
    the stream's hidden params (`vertex_and_google_ai_studio_gemini.py:4414-4416`) and its one spend row
    is 0.00099. UNVERIFIED: relies on the chunk builder carrying `provider_specific_fields` to the logged
    response (`streaming_chunk_builder_utils.py:1950`)."""
    with vertex(gateway, _served("ON_DEMAND_PRIORITY")) as rig:
        response, _text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_PRIORITY)


def test_the_provisioned_throughput_traffic_type_is_not_in_the_caller_response(gateway: Gateway) -> None:
    """`trafficType` lives only in hidden params: the caller's JSON has no `traffic_type` or
    `PROVISIONED_THROUGHPUT` anywhere, and `usage` keeps the OpenAI counts 1000 / 100 / 1100."""
    with vertex(gateway, _served("PROVISIONED_THROUGHPUT")) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200
        assert "PROVISIONED_THROUGHPUT" not in response.text
        assert "traffic_type" not in response.text
        usage: Final = json_of(response)["usage"]
        assert isinstance(usage, dict)
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (1000, 100, 1100)
