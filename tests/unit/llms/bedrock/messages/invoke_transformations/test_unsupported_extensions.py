import copy

import pytest

from litellm.llms.bedrock.messages.invoke_transformations.unsupported_extensions import (
    Extension,
    Offender,
    OptIns,
    Refused,
    Sanitized,
    Unchanged,
    extension_betas,
    sanitize_for_bedrock_invoke,
)

ALL_OPT_INS = OptIns(drop_params=True, modify_params=True)
NO_OPT_INS = OptIns(drop_params=False, modify_params=False)
NOTHING_SUPPORTED = frozenset()
EVERYTHING_SUPPORTED = frozenset(Extension)
TOOL_ADDITION = {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__linear__list_issues"}}
TOOL_REMOVAL = {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "mcp__linear__list_issues"}}
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "Read",
    "input": {"path": "/tmp/a.txt", "flags": [1, 2.5, None]},
}
TERSE = {"type": "text", "text": "Answer tersely."}

DISPLAY_OFFENDER = Offender(path="thinking.display (value 'updates')", extension=Extension.THINKING_DISPLAY_UPDATES)
EFFORT_OFFENDER = Offender(path="messages[1].output_config", extension=Extension.MESSAGE_OUTPUT_CONFIG)
ADDITION_OFFENDER = Offender(path="messages[2].content[0] (type 'tool_addition')", extension=Extension.TOOL_CHANGES)
REMOVAL_OFFENDER = Offender(path="messages[5].content[1] (type 'tool_removal')", extension=Extension.TOOL_CHANGES)


def _claude_code_request():
    return {
        "max_tokens": 100,
        "output_config": {"effort": "high"},
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "thinking": {"type": "adaptive", "display": "updates"},
        "messages": [
            {"role": "user", "content": "read /tmp/a.txt"},
            {"role": "system", "content": [], "output_config": {"effort": "low"}},
            {"role": "system", "content": [TOOL_ADDITION]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}}, TOOL_USE],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi"}]},
            {"role": "system", "content": [TERSE, TOOL_REMOVAL]},
        ],
        "tools": [{"name": "mcp__linear__list_issues", "defer_loading": True, "input_schema": {"type": "object"}}],
    }


@pytest.mark.parametrize(
    "request_body",
    [
        {"messages": "not a list", "thinking": "not a dict"},
        {"messages": ["bare string", 7, None], "thinking": {"type": "adaptive"}},
        {"messages": [{"role": "system", "content": ["bare string", 7, {"type": "text", "text": "ok"}]}]},
        {"messages": [{"role": "system", "content": [{"type": ["unhashable"]}]}], "thinking": {"display": ["x"]}},
        {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive", "display": "summarized"}},
        {"messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "high"}},
    ],
)
def test_sanitize_leaves_requests_without_extensions_alone(request_body):
    snapshot = copy.deepcopy(request_body)

    outcome = sanitize_for_bedrock_invoke(request_body, NO_OPT_INS, supported=NOTHING_SUPPORTED)

    assert outcome == Unchanged()
    assert request_body == snapshot


def test_sanitize_leaves_extensions_the_model_supports_alone_even_without_opt_ins():
    assert sanitize_for_bedrock_invoke(_claude_code_request(), NO_OPT_INS, supported=EVERYTHING_SUPPORTED) == (
        Unchanged()
    )


def test_sanitize_removes_every_unsupported_extension_and_drops_only_the_system_messages_it_empties():
    request_body = _claude_code_request()
    snapshot = copy.deepcopy(request_body)
    messages = snapshot["messages"]

    outcome = sanitize_for_bedrock_invoke(request_body, ALL_OPT_INS, supported=NOTHING_SUPPORTED)

    assert isinstance(outcome, Sanitized)
    assert {**request_body, **outcome.replacements} == {
        **snapshot,
        "thinking": {"type": "adaptive"},
        "messages": [messages[0], messages[3], messages[4], {"role": "system", "content": [TERSE]}],
    }
    assert outcome.removed == (DISPLAY_OFFENDER, EFFORT_OFFENDER, ADDITION_OFFENDER, REMOVAL_OFFENDER)
    assert request_body == snapshot


def test_sanitize_strips_only_the_extensions_missing_from_the_supported_set():
    request_body = _claude_code_request()
    snapshot = copy.deepcopy(request_body)

    outcome = sanitize_for_bedrock_invoke(
        request_body, ALL_OPT_INS, supported=frozenset({Extension.TOOL_CHANGES, Extension.THINKING_DISPLAY_UPDATES})
    )

    assert isinstance(outcome, Sanitized)
    assert outcome.removed == (EFFORT_OFFENDER,)
    assert {**request_body, **outcome.replacements} == {
        **snapshot,
        "messages": [snapshot["messages"][0], *snapshot["messages"][2:]],
    }


def test_sanitize_keeps_a_supported_effort_message_while_stripping_unsupported_tool_changes():
    request_body = _claude_code_request()
    snapshot = copy.deepcopy(request_body)
    messages = snapshot["messages"]

    outcome = sanitize_for_bedrock_invoke(
        request_body,
        ALL_OPT_INS,
        supported=frozenset({Extension.MESSAGE_OUTPUT_CONFIG, Extension.THINKING_DISPLAY_UPDATES}),
    )

    assert isinstance(outcome, Sanitized)
    assert outcome.removed == (ADDITION_OFFENDER, REMOVAL_OFFENDER)
    assert {**request_body, **outcome.replacements} == {
        **snapshot,
        "messages": [messages[0], messages[1], messages[3], messages[4], {"role": "system", "content": [TERSE]}],
    }


def test_sanitize_fills_an_emptied_user_or_assistant_turn_instead_of_dropping_it():
    outcome = sanitize_for_bedrock_invoke(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [TOOL_ADDITION]}]},
        ALL_OPT_INS,
        supported=NOTHING_SUPPORTED,
    )

    assert isinstance(outcome, Sanitized)
    assert outcome.replacements["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Please continue."}]},
    ]


@pytest.mark.parametrize(
    ("opt_ins", "blocked"),
    [
        (OptIns(drop_params=True, modify_params=False), (ADDITION_OFFENDER, REMOVAL_OFFENDER)),
        (OptIns(drop_params=False, modify_params=True), (DISPLAY_OFFENDER, EFFORT_OFFENDER)),
        (NO_OPT_INS, (DISPLAY_OFFENDER, EFFORT_OFFENDER, ADDITION_OFFENDER, REMOVAL_OFFENDER)),
    ],
    ids=["drop-params-only", "modify-params-only", "no-opt-ins"],
)
def test_sanitize_refuses_with_only_the_unsupported_extensions_whose_opt_in_is_off(opt_ins, blocked):
    assert sanitize_for_bedrock_invoke(_claude_code_request(), opt_ins, supported=NOTHING_SUPPORTED) == Refused(
        offenders=blocked
    )


def test_sanitize_never_refuses_an_extension_the_model_supports():
    outcome = sanitize_for_bedrock_invoke(
        _claude_code_request(), NO_OPT_INS, supported=frozenset({Extension.TOOL_CHANGES})
    )

    assert outcome == Refused(offenders=(DISPLAY_OFFENDER, EFFORT_OFFENDER))


@pytest.mark.parametrize(
    ("request_body", "betas"),
    [
        (_claude_code_request(), frozenset(extension.beta for extension in Extension)),
        (
            {"messages": [{"role": "system", "content": [TOOL_REMOVAL]}]},
            frozenset({"mid-conversation-tool-changes-2026-07-01"}),
        ),
        ({"thinking": {"type": "adaptive", "display": "updates"}}, frozenset({"thinking-display-updates-2026-08-18"})),
        ({"messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "high"}}, frozenset()),
        ({"messages": "not a list"}, frozenset()),
    ],
    ids=["every-extension", "tool-removal", "display-updates", "top-level-effort-only", "unparseable"],
)
def test_extension_betas_names_the_beta_of_each_extension_the_request_uses(request_body, betas):
    assert extension_betas(request_body) == betas
