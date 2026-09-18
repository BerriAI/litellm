"""Token-pricing e2e: every (map entry, case) cell derived from cost_map.json x
cases.json runs a scripted-usage call through a deployment registered on the
cost-map proxy, and the spend row plus response-cost header must equal the
reviewed golden in the case's ``expected`` cell verbatim -- no rate arithmetic
lives here.

Nothing here touches a real provider or the bundled cost map: the proxy's
upstream is the scripted-provider sidecar and its entire cost map is
tests/e2e/cost_map.json.
"""

from __future__ import annotations

import pytest
from typing import Final

from conftest import CostCalcClient, cost_rows, register_scenario_deployment
from cost_matrix import (
    AUDIO_INPUT_DATA_URL,
    FRONTIER_MODELS,
    IMAGE_INPUT_DATA_URL,
    SERVICE_TIER_REQUEST_WIRES,
    VIDEO_INPUT_DATA_URL,
    Case,
    FrontierModel,
    cases_for,
    matrix_data_errors,
    recount_cost,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import (
    CacheControl,
    ChatAudio,
    ChatBody,
    ChatMessage,
    ChatStreamOptions,
    ChatTool,
    ChatToolFunction,
    FileContentPart,
    FileObject,
    FileSearchTool,
    GoogleMapsTool,
    GoogleSearchTool,
    HostedWebSearchTool,
    ImageContentPart,
    ImageUrl,
    InputAudio,
    InputAudioContentPart,
    TextContentPart,
    WebSearchOptions,
)
from scripted_provider import ScriptedUsage, Wire

pytestmark: Final = [pytest.mark.e2e, pytest.mark.cost_map_stack]  # mutable-ok: pytest only accepts a list for pytestmark

if _data_errors := matrix_data_errors():
    raise ValueError("\n".join(_data_errors))

_MATRIX: Final[tuple[tuple[FrontierModel, Case], ...]] = tuple(
    (model, case) for model in FRONTIER_MODELS for case in cases_for(model)
)


def _case_id(param: tuple[FrontierModel, Case]) -> str:
    model, case = param
    return f"{model.map_key.replace('/', '-')}-{case.name}"


_CACHE_WIRES: Final = frozenset({"anthropic_messages", "bedrock_converse"})
_WEB_SEARCH_OPTION_WIRES: Final = frozenset({"openai_chat", "azure_chat", "openai_responses"})


def _cache_control(usage: ScriptedUsage, wire: Wire) -> CacheControl | None:
    if wire not in _CACHE_WIRES:
        return None
    if not (usage.cache_read_tokens or usage.cache_write_5m_tokens or usage.cache_write_1h_tokens):
        return None
    return CacheControl(type="ephemeral", ttl="1h" if usage.cache_write_1h_tokens else None)


def _chat_body(model: FrontierModel, case: Case, model_name: str, marker: str) -> ChatBody:
    usage: Final = case.usage_for(model.map_key)
    user_parts: Final = (
        TextContentPart(
            text=f"{marker} summarize the attached material in one line and name the city weather",
        ),
        *(
            (ImageContentPart(image_url=ImageUrl(url=IMAGE_INPUT_DATA_URL, detail="high")),)
            if case.image_input
            else ()
        ),
        *(
            (
                InputAudioContentPart(
                    input_audio=InputAudio(data=AUDIO_INPUT_DATA_URL.split(",", 1)[1], format="wav")
                ),
            )
            if case.audio_input
            else ()
        ),
        *(
            (FileContentPart(file=FileObject(file_data=VIDEO_INPUT_DATA_URL, format="mp4")),)
            if case.video_input
            else ()
        ),
    )
    tools: Final = (
        *(
            (
                ChatTool(
                    function=ChatToolFunction(
                        name="get_weather",
                        description="Get the current weather and a short forecast for a city.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                                "days": {"type": "integer", "description": "Forecast horizon in days"},
                                "units": {"type": "string", "enum": ["metric", "imperial"]},
                            },
                            "required": ["city"],
                        },
                    )
                ),
            )
            if case.tool_call
            else ()
        ),
        *(
            (HostedWebSearchTool(type="web_search_20250305", name="web_search", max_uses=5),)
            if case.web_search is not None and model.wire == "anthropic_messages"
            else ()
        ),
        *(
            (GoogleSearchTool(),)
            if case.web_search is not None and model.wire in ("gemini_generate", "vertex_generate")
            else ()
        ),
        *((GoogleMapsTool(),) if case.google_maps else ()),
        *((FileSearchTool(vector_store_ids=["vs_cost_calc_fixture"]),) if case.file_search else ()),
    )
    return ChatBody(
        model=model_name,
        messages=(
            ChatMessage(
                role="system",
                content=[
                    TextContentPart(
                        text=(
                            "You are a deterministic pricing-harness assistant. "
                            "Keep answers to a single short line."
                        ),
                        cache_control=_cache_control(usage, model.wire),
                    )
                ],
            ),
            ChatMessage(role="user", content=list(user_parts)),
        ),
        stream=case.stream,
        stream_options=ChatStreamOptions(include_usage=True) if case.stream else None,
        service_tier=(
            case.service_tier
            if case.service_tier is not None and model.wire in SERVICE_TIER_REQUEST_WIRES
            else None
        ),
        reasoning_effort="medium" if case.reasoning else None,
        modalities=(
            ["text"] if case.audio_input else (["text", "audio"] if case.audio_output else None)
        ),
        audio=(
            ChatAudio(voice="alloy", format="pcm16") if case.audio_output else None
        ),
        web_search_options=(
            WebSearchOptions(search_context_size=case.web_search)
            if case.web_search is not None and model.wire in _WEB_SEARCH_OPTION_WIRES
            else None
        ),
        tools=tools or None,
        tool_choice="auto" if case.tool_call and model.wire != "bedrock_converse" else None,
        # The test-owned cost map carries no supports_* flags, so litellm's
        # optional-params gate rejects the realistic request fields; allowlist
        # exactly the ones this case sends.
        allowed_openai_params=[
            name
            for name, sent in (
                ("tool_choice", case.tool_call and model.wire != "bedrock_converse"),
                ("modalities", case.audio_input or case.audio_output),
                ("audio", case.audio_output),
                ("web_search_options", case.web_search is not None),
                ("reasoning_effort", case.reasoning),
            )
            if sent
        ],
    )


class TestTokenPricing:
    @pytest.mark.parametrize("model_case", _MATRIX, ids=_case_id)
    @pytest.mark.covers("quota_management.spend_tracking.cost_matrix.logs_cost")
    def test_scripted_usage_bills_at_map_rates(
        self,
        client: CostCalcClient,
        resources: ResourceManager,
        scoped_key: str,
        model_case: tuple[FrontierModel, Case],
    ) -> None:
        model, case = model_case
        marker: Final = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response: Final = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=_chat_body(model, case, model_name, marker),
            stream=case.stream,
        )
        assert response.ok, (
            f"{model.map_key}/{case.name}: proxy returned {response.status_code}: {response.body[:400]}"
        )
        assert response.stream_error is None, f"stream carried an error event: {response.stream_error}"

        row: Final = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, f"no spend row with a cost breakdown landed for {model.map_key}/{case.name}"

        if not case.exact_spend:
            # stream_usage=absent: the provider reported no usage, so the row's
            # token counts are the proxy's own recount; assert the recount
            # billed both directions at the case's rates.
            assert row.prompt_tokens is not None and row.prompt_tokens > 0, (
                f"no-usage stream counted no input tokens: {row}"
            )
            assert row.completion_tokens is not None and row.completion_tokens > 0, (
                f"no-usage stream counted no output tokens: {row}"
            )
            if case.image_input:
                assert row.prompt_tokens < 4000, (
                    f"image data URL looks tokenized as text: prompt_tokens={row.prompt_tokens}"
                )
            assert row.spend is not None and cost_rows.approx_equal(
                row.spend,
                recount_cost(model, case, row.prompt_tokens, row.completion_tokens),
            ), f"no-usage stream spend {row.spend} != recount at map rates: {row}"
            cost_rows.assert_total_is_sum_of_components(row)
            return

        golden: Final = case.expected_for(model)

        if not case.stream:
            # Streamed responses commit headers before the bill is computed, so
            # the x-litellm-response-cost header is asserted only on non-stream
            # calls.
            assert response.response_cost is not None and cost_rows.approx_equal(
                response.response_cost, golden.spend
            ), (
                f"x-litellm-response-cost {response.response_cost} != golden {golden.spend}"
            )

        assert row.spend is not None and cost_rows.approx_equal(row.spend, golden.spend), (
            f"{model.map_key}/{case.name}: spend {row.spend} != golden {golden.spend} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        breakdown: Final = row.breakdown
        assert breakdown.input_cost is not None and cost_rows.approx_equal(
            breakdown.input_cost, golden.input_cost
        ), (
            f"{model.map_key}/{case.name}: gross input_cost {breakdown.input_cost} "
            f"!= golden {golden.input_cost}; cached/written tokens billed at the input rate"
        )
        assert breakdown.output_cost is not None and cost_rows.approx_equal(
            breakdown.output_cost, golden.output_cost
        ), (
            f"{model.map_key}/{case.name}: output_cost {breakdown.output_cost} "
            f"!= golden {golden.output_cost}"
        )
        assert row.prompt_tokens == golden.prompt_tokens, (
            f"prompt_tokens {row.prompt_tokens} != {golden.prompt_tokens}"
        )
        assert row.completion_tokens == golden.completion_tokens, (
            f"completion_tokens {row.completion_tokens} != {golden.completion_tokens}"
        )
        cost_rows.assert_total_is_sum_of_components(row)
