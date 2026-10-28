"""The same conversation contract on every (endpoint, deployment, auth) cell.
A deployment is one provider model (openai/gpt-4o-mini, anthropic/claude-haiku-4-5, ...).

/chat/completions, /v1/messages and /v1/responses each have their own
translation code in the proxy, so a bug fixed on one surface tends to survive
on the others. Every test here runs once per cell in `CELLS`
(conversational_matrix.py), so a change to a shared helper is proven against all
surfaces and providers at once, and a new model or provider is one row in `DEPLOYMENTS`.

Edge-wired: OpenAI and Anthropic traffic goes through the provider edge in
record and replay, so the whole matrix replays with zero provider calls.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from llm_translation.conversational_matrix import (
    GREETING_PROMPT,
    WEATHER_PROMPT,
    WEATHER_REPORT,
    WEATHER_TOOL_NAME,
    Cell,
    Deployments,
    Surface,
    SurfaceName,
    ToolCall,
    build_surfaces,
    cells_covering,
    register_deployments,
)
from llm_translation.sdk_clients import SdkClients
from models import SpendLogRow
from proxy_client import ProxyClient

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


@pytest.fixture(scope="module")
def deployments(proxy: ProxyClient) -> Iterator[Deployments]:
    yield from register_deployments(proxy)


@pytest.fixture(scope="module")
def surfaces(sdk: SdkClients) -> Mapping[SurfaceName, Surface]:
    return build_surfaces(sdk)


def _weather_call(surface: Surface, key: str, model: str) -> ToolCall:
    first: Final = surface.reply(key, model, WEATHER_PROMPT, with_tool=True)
    assert len(first.tool_calls) == 1, (
        f"{surface.name} forced tool_choice={WEATHER_TOOL_NAME} with parallel calls off, "
        f"got {len(first.tool_calls)} tool call(s): {first.tool_calls} text={first.text!r}"
    )
    call: Final = first.tool_calls[0]
    assert call.name == WEATHER_TOOL_NAME, f"{surface.name} called {call.name!r}, not the forced {WEATHER_TOOL_NAME!r}"
    assert call.call_id, f"{surface.name} tool call has no id, so the caller cannot answer it: {call}"
    assert "paris" in call.parsed().location.lower(), f"{surface.name} tool arguments lost the location: {call}"
    return call


class TestConversationalMatrix:
    @pytest.mark.parametrize("cell", cells_covering("basic", "nonstream", "works"))
    def test_reply_carries_assistant_text_and_usage(
        self,
        cell: Cell,
        deployments: Deployments,
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        surface: Final = surfaces[cell.surface]
        reply: Final = surface.reply(resources.key(), deployments.alias(cell), GREETING_PROMPT)

        assert reply.response_id, f"{cell.id}: response has no id"
        assert reply.text.strip(), f"{cell.id}: response carried no assistant text"
        assert reply.usage is not None and reply.usage.input_tokens > 0 and reply.usage.output_tokens > 0, (
            f"{cell.id}: usage missing or zero, so the caller cannot account for this call: {reply.usage}"
        )
        assert reply.call_id_header, f"{cell.id}: x-litellm-call-id header missing"

    @pytest.mark.parametrize("cell", cells_covering("basic", "stream", "works"))
    def test_stream_delivers_text_usage_and_a_terminal_event(
        self,
        cell: Cell,
        deployments: Deployments,
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        surface: Final = surfaces[cell.surface]
        streamed: Final = surface.stream(resources.key(), deployments.alias(cell), GREETING_PROMPT)

        assert streamed.event_count > 1, f"{cell.id}: stream arrived as {streamed.event_count} event(s), not a stream"
        assert streamed.text.strip(), f"{cell.id}: stream carried no text deltas"
        assert streamed.finished, f"{cell.id}: stream never sent its terminal event"
        assert streamed.usage_reported, f"{cell.id}: stream never reported usage"

    @pytest.mark.parametrize("cell", cells_covering("basic", "nonstream", "cost_logged"))
    def test_cost_header_matches_the_spend_log(
        self,
        cell: Cell,
        deployments: Deployments,
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
        proxy: ProxyClient,
    ) -> None:
        key: Final = resources.key()
        surface: Final = surfaces[cell.surface]
        reply: Final = surface.reply(key, deployments.alias(cell), f"{GREETING_PROMPT} {unique_marker()}")

        assert reply.cost_header is not None, f"{cell.id}: x-litellm-response-cost header missing"
        header_cost: Final = float(reply.cost_header)
        assert header_cost > 0, f"{cell.id}: x-litellm-response-cost is not positive: {header_cost}"

        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(row.spend is not None and row.spend > 0 for row in rows)

        rows: Final = proxy.poll_logs_for_key(key, predicate=_priced)
        priced: Final = tuple(row for row in rows if row.spend is not None and row.spend > 0)
        assert len(priced) == 1, f"{cell.id}: expected exactly one priced spend row for a fresh key, got {rows}"
        row: Final = priced[0]
        assert (row.prompt_tokens or 0) > 0 and (row.completion_tokens or 0) > 0, (
            f"{cell.id}: spend row has no token counts, so the cost is not real usage: {row}"
        )
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"{cell.id}: logged spend {row.spend} disagrees with x-litellm-response-cost {header_cost}"
        )
        assert row.model and cell.deployment.backend.endswith(row.model), (
            f"{cell.id}: spend row logged model {row.model!r}, not the deployment's {cell.deployment.backend!r}"
        )

    @pytest.mark.parametrize("cell", cells_covering("tool_use", "nonstream", "works"))
    def test_tool_call_is_returned_named_and_addressable(
        self,
        cell: Cell,
        deployments: Deployments,
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        _weather_call(surfaces[cell.surface], resources.key(), deployments.alias(cell))

    @pytest.mark.parametrize("cell", cells_covering("multi_turn", "nonstream", "works"))
    def test_tool_result_round_trip_reaches_the_model(
        self,
        cell: Cell,
        deployments: Deployments,
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        key: Final = resources.key()
        model: Final = deployments.alias(cell)
        surface: Final = surfaces[cell.surface]
        call: Final = _weather_call(surface, key, model)

        answer: Final = surface.reply_to_tool_result(key, model, WEATHER_PROMPT, call, WEATHER_REPORT)
        assert "22" in answer.text, f"{cell.id}: the model never saw the tool result: {answer.text!r}"
