from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.reasoning_utils import encrypted_reasoning_signature
from litellm.llms.anthropic.messages.responses_adapters.configuration_updates import Next, Update, encode_stamp
from litellm.llms.anthropic.messages.responses_adapters.handler import ResponsesPlan, responses_plan
from litellm.llms.anthropic.messages.responses_adapters.transformation import LiteLLMAnthropicToResponsesAPIAdapter
from litellm.types.llms.anthropic import AnthropicOutputConfig
from litellm.types.llms.openai import ResponsesAPIResponse

_ADAPTER: Final = LiteLLMAnthropicToResponsesAPIAdapter()
_GATED_MODEL: Final = "openai/gpt-6.2-sol"
_OPENAI: Final = {"custom_llm_provider": "openai", "api_base": "https://api.openai.com/v1"}
_COMPATIBLE_HOST: Final = {"custom_llm_provider": "openai", "api_base": "https://gateway.example/v1"}
_SCHEMA: Final = {
    "type": "json_schema",
    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
}
_INLINE_TYPES: Final = ("configuration_update", "additional_tools")
_ITEMS: Final = TypeAdapter(list[dict[str, object]])
_OUTPUT_CONFIG: Final = TypeAdapter(AnthropicOutputConfig)


def _tool(name: str, description: str = "A tool") -> dict[str, object]:
    return {"name": name, "description": description, "input_schema": {"type": "object", "properties": {}}}


READ: Final = _tool("Read")
WRITE: Final = _tool("Write")
WEB_SEARCH: Final[dict[str, object]] = {"type": "web_search_20250305", "name": "web_search"}
_SCHEMA_B: Final = {
    "type": "json_schema",
    "schema": {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]},
}


@dataclass(frozen=True, slots=True)
class Turn:
    effort: str = "low"
    tools: tuple[dict[str, object], ...] = (READ,)
    output_format: Mapping[str, object] | None = None
    thinking: bool = True
    summary: str | None = None
    tool_choice: dict[str, object] | None = None
    calls_tool: bool = False


@dataclass(frozen=True, slots=True)
class Exchange:
    plan: ResponsesPlan
    history: tuple[Mapping[str, object], ...]


def _blocks(message: Mapping[str, object]) -> list[dict[str, object]]:
    return _ITEMS.validate_python(message["content"])


def _input(plan: ResponsesPlan) -> list[dict[str, object]]:
    return _ITEMS.validate_python(plan.kwargs["input"])


def _user_message(index: int, previous: Exchange | None) -> Mapping[str, object]:
    called_tool: Final = previous is not None and _blocks(previous.history[-1])[-1]["type"] == "tool_use"
    if called_tool:
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_read", "content": "ok"}]}
    return {"role": "user", "content": f"question {index}"}


def _response(index: int, turn: Turn) -> ResponsesAPIResponse:
    reasoning: Final = {
        "type": "reasoning",
        "id": f"rs_{index}",
        "encrypted_content": f"gAAAA_blob_{index}",
        "summary": [{"type": "summary_text", "text": f"Thinking {index}."}],
    }
    answer: Final[dict[str, object]] = (
        {"type": "function_call", "id": "fc_1", "call_id": "call_read", "name": "Read", "arguments": "{}"}
        if turn.calls_tool
        else {
            "type": "message",
            "id": f"msg_{index}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": f"answer {index}", "annotations": []}],
        }
    )
    return ResponsesAPIResponse.model_validate(
        {
            "id": f"resp_{index}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6.2-sol",
            "output": [reasoning, answer],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    )


def _plan(
    turn: Turn,
    messages: Sequence[Mapping[str, object]],
    model: str = _GATED_MODEL,
    extra: Mapping[str, object] = _OPENAI,
) -> ResponsesPlan:
    output_config: Final = _OUTPUT_CONFIG.validate_python(
        {"effort": turn.effort, **({"format": turn.output_format} if turn.output_format else {})}
    )
    return responses_plan(
        model=model,
        max_tokens=64,
        messages=[dict(message) for message in messages],
        thinking={"type": "adaptive", **({"summary": turn.summary} if turn.summary else {})} if turn.thinking else None,
        output_config=output_config,
        tools=list(turn.tools),
        tool_choice=turn.tool_choice,
        extra_kwargs=extra,
    )


def _converse(
    turns: Sequence[Turn],
    model: str = _GATED_MODEL,
    extra: Mapping[str, object] = _OPENAI,
    previous: Exchange | None = None,
) -> tuple[Exchange, ...]:
    """Each turn's request and the history after its response, every assistant turn echoed back as the client got it."""
    if not turns:
        return ()
    index: Final = 0 if previous is None else len(previous.history) // 2
    messages: Final = (*(previous.history if previous else ()), _user_message(index, previous))
    plan: Final = _plan(turns[0], messages, model, extra)
    reply: Final = _ADAPTER.translate_response(_response(index, turns[0]), reasoning_stamp=plan.reasoning_stamp)
    exchange: Final = Exchange(
        plan=plan, history=(*messages, {"role": "assistant", "content": reply.get("content", [])})
    )
    return (exchange, *_converse(turns[1:], model, extra, exchange))


def _top_level(plan: ResponsesPlan) -> Mapping[str, object]:
    return {key: plan.kwargs[key] for key in ("reasoning", "text", "tools") if key in plan.kwargs}


def _inline_items(plan: ResponsesPlan) -> list[dict[str, object]]:
    return [item for item in _input(plan) if item["type"] in _INLINE_TYPES]


def _shape(plan: ResponsesPlan) -> list[object]:
    return [item["role"] if item["type"] == "message" else item["type"] for item in _input(plan)]


def _first_turn(turn: Turn) -> ResponsesPlan:
    return _plan(turn, ({"role": "user", "content": "question 0"},))


_MIXED_CHANGES: Final = (
    Turn(effort="low"),
    Turn(effort="high", calls_tool=True),
    Turn(effort="medium", tools=(READ, WRITE)),
    Turn(effort="medium", tools=(READ, WRITE), output_format=_SCHEMA),
    Turn(effort="medium", tools=(READ, WRITE)),
)


_TOOLLESS_CHANGES: Final = (
    Turn(effort="low", tools=()),
    Turn(effort="high", tools=()),
    Turn(effort="high", tools=(), output_format=_SCHEMA),
)
_NAMELESS_TOOL_CHANGES: Final = (
    Turn(effort="low", tools=(WEB_SEARCH, READ)),
    Turn(effort="high", tools=(WEB_SEARCH, READ)),
    Turn(effort="high", tools=(WEB_SEARCH, READ, WRITE)),
)


@pytest.mark.parametrize(
    "turns",
    [
        pytest.param(_MIXED_CHANGES, id="with-tools"),
        pytest.param(_TOOLLESS_CHANGES, id="no-tools"),
        pytest.param(_NAMELESS_TOOL_CHANGES, id="with-a-nameless-tool"),
    ],
)
def test_every_settings_change_keeps_each_request_a_prefix_of_the_next(turns: tuple[Turn, ...]) -> None:
    """regression: a mid-conversation change on GPT-6+ never rewrites the cached prefix, with or without tools."""
    plans: Final = tuple(exchange.plan for exchange in _converse(turns))

    for earlier, later in zip(plans, plans[1:]):
        assert _input(later)[: len(_input(earlier))] == _input(earlier)
        assert _top_level(later) == _top_level(plans[0])
    assert _top_level(plans[0])["reasoning"] == {"effort": "low"}
    assert len(_inline_items(plans[-1])) == sum(
        (turn.effort != previous.effort)
        + (turn.output_format != previous.output_format)
        + (turn.tools != previous.tools)
        for previous, turn in zip(turns, turns[1:])
    )


def test_each_change_is_replayed_where_it_happened() -> None:
    last: Final = _converse(_MIXED_CHANGES)[-1].plan
    write_tool: Final = _first_turn(Turn(tools=(WRITE,))).kwargs["tools"]

    assert _shape(last) == [
        "user",
        "reasoning",
        "assistant",
        "configuration_update",
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
        "configuration_update",
        "additional_tools",
        "reasoning",
        "assistant",
        "configuration_update",
        "user",
        "reasoning",
        "assistant",
        "configuration_update",
        "user",
    ]
    assert _inline_items(last) == [
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
        {"type": "configuration_update", "reasoning": {"effort": "medium"}},
        {"type": "additional_tools", "role": "developer", "tools": write_tool},
        {"type": "configuration_update", "text": _first_turn(_MIXED_CHANGES[3]).kwargs["text"]},
        {"type": "configuration_update", "text": {"format": {"type": "text"}}},
    ]


def test_a_turn_without_changes_sends_no_inline_item() -> None:
    plans: Final = tuple(exchange.plan for exchange in _converse((Turn(), Turn(), Turn())))

    assert all(not _inline_items(plan) for plan in plans)
    assert all(_top_level(plan) == _top_level(plans[0]) for plan in plans)


_UNEXPRESSIBLE_CHANGES: Final = (
    pytest.param(Turn(effort="high", tools=(WRITE,)), id="tool-removed"),
    pytest.param(Turn(effort="high", tools=(_tool("Read", "Reads a file"),)), id="tool-redefined"),
    pytest.param(Turn(effort="high", tools=(READ, WEB_SEARCH)), id="non-function-tool-added"),
    pytest.param(Turn(effort="high", thinking=False), id="thinking-off"),
    pytest.param(Turn(effort="high", summary="detailed"), id="summary-changed"),
)


@pytest.mark.parametrize("changed", _UNEXPRESSIBLE_CHANGES)
def test_a_change_inline_items_cannot_express_starts_a_new_baseline(changed: Turn) -> None:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="medium"), changed, replace(changed, effort="low")))
    fallback, after = exchanges[2].plan, exchanges[3].plan

    assert _top_level(fallback) == _top_level(_first_turn(changed))
    assert not _inline_items(fallback)
    assert _input(after)[: len(_input(fallback))] == _input(fallback)
    assert _top_level(after) == _top_level(fallback)


def _with_assistant_content(
    history: Sequence[Mapping[str, object]], position: int, content: Sequence[object]
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {**message, "content": list(content)} if index == position else message for index, message in enumerate(history)
    )


def _thinking_with_signature(signature: str) -> dict[str, object]:
    return {"type": "thinking", "thinking": "Thinking 1.", "signature": signature}


_ANSWER: Final = {"type": "text", "text": "answer 1"}
_BROKEN_HISTORIES: Final = (
    pytest.param([_ANSWER], id="no-thinking-block"),
    pytest.param([_thinking_with_signature(encrypted_reasoning_signature("gAAAA_blob_1")), _ANSWER], id="no-stamp"),
    pytest.param(
        [_thinking_with_signature(encrypted_reasoning_signature("gAAAA_blob_1", "eyJ2IjoxLCJuIjoxfQ")), _ANSWER],
        id="stamp-missing-its-update",
    ),
)


@pytest.mark.parametrize("assistant_content", _BROKEN_HISTORIES)
def test_a_regime_turn_missing_its_stamp_falls_back_to_top_level_settings(assistant_content: list[object]) -> None:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="medium")))
    history: Final = _with_assistant_content(exchanges[1].history, 3, assistant_content)
    plan: Final = _plan(Turn(effort="high"), (*history, {"role": "user", "content": "question 2"}))

    assert _top_level(plan) == _top_level(_first_turn(Turn(effort="high")))
    assert not _inline_items(plan)
    assert plan.reasoning_stamp is not None


def test_a_stamp_moved_onto_another_blob_is_ignored() -> None:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="medium")))
    signature: Final = str(_blocks(exchanges[1].history[-1])[0]["signature"])
    moved: Final = signature.replace("gAAAA_blob_1", "gAAAA_blob_other")
    history: Final = _with_assistant_content(exchanges[1].history, 3, [_thinking_with_signature(moved), _ANSWER])
    plan: Final = _plan(Turn(effort="high"), (*history, {"role": "user", "content": "question 2"}))

    assert _top_level(plan) == _top_level(_first_turn(Turn(effort="high")))
    assert not _inline_items(plan)


_UNGATED_ROUTES: Final = (
    pytest.param("openai/gpt-5.5", _OPENAI, id="gpt-5.5"),
    pytest.param(
        "azure/gpt-6.2-sol", {"custom_llm_provider": "azure", "api_base": "https://x.openai.azure.com"}, id="azure"
    ),
    pytest.param(_GATED_MODEL, {**_OPENAI, "extra_body": {"tools": []}}, id="caller-owned-tools"),
    pytest.param(
        _GATED_MODEL, {**_OPENAI, "extra_body": {"reasoning": {"effort": "low"}}}, id="caller-owned-reasoning"
    ),
    pytest.param(_GATED_MODEL, {**_OPENAI, "conversation": "conv_1"}, id="conversation"),
    pytest.param(_GATED_MODEL, _COMPATIBLE_HOST, id="openai-compatible-host"),
)


@pytest.mark.parametrize(("model", "extra"), _UNGATED_ROUTES)
def test_routes_outside_the_gate_send_settings_top_level_and_unstamped(model: str, extra: Mapping[str, object]) -> None:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="high", tools=(READ, WRITE))), model, extra)
    second: Final = exchanges[1].plan

    assert second.reasoning_stamp is None
    assert not _inline_items(second)
    assert [tool["name"] for tool in _ITEMS.validate_python(second.kwargs["tools"])] == ["Read", "Write"]
    assert _blocks(exchanges[0].history[-1])[0]["signature"] == encrypted_reasoning_signature("gAAAA_blob_0")


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        pytest.param("openai/my-team-model", {**_OPENAI, "base_model": "gpt-6.2-sol"}, id="gpt-6-base-model-alias"),
        pytest.param(
            _GATED_MODEL, {**_COMPATIBLE_HOST, "prompt_cache_options": {"mode": "implicit"}}, id="host-vouched-for"
        ),
    ],
)
def test_gated_routes_send_an_effort_change_inline(model: str, extra: Mapping[str, object]) -> None:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="high")), model, extra)

    assert _inline_items(exchanges[1].plan) == [{"type": "configuration_update", "reasoning": {"effort": "high"}}]
    assert _top_level(exchanges[1].plan) == _top_level(exchanges[0].plan)


def _out_of_order_history() -> tuple[tuple[Mapping[str, object], ...], Turn]:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="medium"), Turn(effort="high")))
    history: Final = exchanges[2].history
    return _with_assistant_content(history, 3, _blocks(history[5])), Turn(effort="high")


def _effort_without_reasoning_history() -> tuple[tuple[Mapping[str, object], ...], Turn]:
    exchanges: Final = _converse((Turn(thinking=False), Turn(thinking=False)))
    return _with_forged_update(exchanges[1].history, Update(effort="high")), Turn(thinking=False)


def _with_forged_update(history: Sequence[Mapping[str, object]], update: Update) -> tuple[Mapping[str, object], ...]:
    signature: Final = encrypted_reasoning_signature("gAAAA_blob_1", encode_stamp(Next(n=1, update=update)))
    return _with_assistant_content(history, 3, [_thinking_with_signature(signature), _ANSWER])


def _forged_update_history(update: Update) -> Callable[[], tuple[tuple[Mapping[str, object], ...], Turn]]:
    def build() -> tuple[tuple[Mapping[str, object], ...], Turn]:
        exchanges: Final = _converse((Turn(effort="low"), Turn(effort="low")))
        return _with_forged_update(exchanges[1].history, update), Turn(effort="low")

    return build


def _adjacent_assistant_turns_history() -> tuple[tuple[Mapping[str, object], ...], Turn]:
    exchanges: Final = _converse((Turn(effort="low"), Turn(effort="medium")))
    history: Final = exchanges[1].history
    return (history[0], history[1], history[3]), Turn(effort="medium")


def _forged_earlier_text_history() -> tuple[tuple[Mapping[str, object], ...], Turn]:
    turns: Final = (Turn(), Turn(output_format=_SCHEMA), Turn(output_format=_SCHEMA_B))
    history: Final = _converse(turns)[2].history
    forged: Final = Update(text={"format": {"type": "text"}, "verbosity": "high"})
    return _with_forged_update(history, forged), Turn(output_format=_SCHEMA_B)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_out_of_order_history, id="turns-out-of-order"),
        pytest.param(_adjacent_assistant_turns_history, id="adjacent-assistant-turns"),
        pytest.param(_effort_without_reasoning_history, id="effort-change-without-reasoning"),
        pytest.param(_forged_update_history(Update(effort="bogus")), id="unknown-effort"),
        pytest.param(_forged_update_history(Update(effort="minimal")), id="effort-the-model-lowers"),
        pytest.param(
            _forged_update_history(Update(text={"format": {"type": "text"}, "verbosity": "high"})),
            id="text-beyond-format",
        ),
        pytest.param(_forged_earlier_text_history, id="earlier-text-beyond-format"),
    ],
)
def test_stamps_that_do_not_fit_the_history_are_not_replayed(
    build: Callable[[], tuple[tuple[Mapping[str, object], ...], Turn]],
) -> None:
    history, turn = build()
    plan: Final = _plan(turn, (*history, {"role": "user", "content": "next question"}))

    assert not _inline_items(plan)
    assert _top_level(plan) == _top_level(_first_turn(turn))


def test_a_text_setting_beyond_the_output_format_changing_starts_a_new_baseline() -> None:
    extra: Final = {**_OPENAI, "text": {"verbosity": "low"}}
    turns: Final = (Turn(), Turn(output_format=_SCHEMA))
    second: Final = _converse(turns, extra=extra)[1].plan

    assert not _inline_items(second)
    assert _top_level(second) == _top_level(_plan(turns[1], ({"role": "user", "content": "question 0"},), extra=extra))


def test_a_request_ending_on_an_assistant_turn_keeps_its_settings_top_level() -> None:
    history: Final = _converse((Turn(effort="low"), Turn(effort="medium")))[1].history
    plan: Final = _plan(Turn(effort="high"), history)

    assert not _inline_items(plan)
    assert _top_level(plan) == _top_level(_first_turn(Turn(effort="high")))


def test_a_format_change_reverted_to_plain_text_sends_nothing_once_settled() -> None:
    turns: Final = (Turn(), Turn(output_format=_SCHEMA), Turn(output_format=_SCHEMA_B), Turn(), Turn())
    plans: Final = tuple(exchange.plan for exchange in _converse(turns))

    assert [len(_inline_items(plan)) for plan in plans] == [0, 1, 2, 3, 3]
    assert _inline_items(plans[-1])[-1] == {"type": "configuration_update", "text": {"format": {"type": "text"}}}


def test_a_tool_added_to_a_toolless_conversation_keeps_a_tool_choice_naming_it() -> None:
    """The shape OpenAI accepts (probed 2026-11-24): no top-level tools, the tool in additional_tools, tool_choice."""
    choice: Final[dict[str, object]] = {"type": "tool", "name": "Write"}
    second: Final = _converse((Turn(tools=()), Turn(tools=(WRITE,), tool_choice=choice)))[1].plan

    assert "tools" not in second.kwargs
    assert second.kwargs["tool_choice"] == {"type": "function", "name": "Write"}
    assert _inline_items(second) == [
        {"type": "additional_tools", "role": "developer", "tools": _first_turn(Turn(tools=(WRITE,))).kwargs["tools"]}
    ]
