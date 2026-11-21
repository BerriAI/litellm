import json
from pathlib import Path

import pytest

from litellm.integrations.lens.lens_trace_adapter import (
    AgentCall,
    ToolExchange,
    agent_call,
    agent_name,
    tool_exchanges,
    user_task,
)

_LS = ToolExchange(call_id="call-ls", name="Bash", arguments='{"command": "ls"}', result="a.txt")
_PWD = ToolExchange(call_id="call-pwd", name="Bash", arguments='{"command": "pwd"}', result="/work")


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("claude-cli/2.1.329 (external, sdk-cli)", "claude-code"),
        ("codex_exec/0.171.0 (Ubuntu 22.04; x86_64)", "codex-cli"),
        ("Codex Desktop/1.0", "codex-cli"),
        ("opencode/1.19.6 ai-sdk/provider-utils/4.0.46 runtime/bun/1.4.4", "opencode"),
        ("pi (linux 6.8.0; x64)", "pi"),
        ("OpenAI/Python 1.99.0", "openai"),
        ("my-agent", "my-agent"),
        ("", None),
        (None, None),
    ],
)
def test_agent_is_named_from_the_caller_user_agent(user_agent: str | None, expected: str | None) -> None:
    assert agent_name(user_agent) == expected


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                {"role": "user", "content": "list then print"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running"},
                        {"type": "tool_use", "id": "call-ls", "name": "Bash", "input": {"command": "ls"}},
                        {"type": "tool_use", "id": "call-pwd", "name": "Bash", "input": {"command": "pwd"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-ls", "content": "a.txt"},
                        {"type": "tool_result", "tool_use_id": "call-pwd", "content": "/work"},
                    ],
                },
            ],
            id="anthropic-blocks",
        ),
        pytest.param(
            [
                {"role": "user", "content": "list then print"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-ls",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                        },
                        {
                            "id": "call-pwd",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": '{"command": "pwd"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
                {"role": "tool", "tool_call_id": "call-pwd", "content": "/work"},
            ],
            id="openai-chat",
        ),
        pytest.param(
            [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list then print"}]},
                {"type": "function_call", "name": "Bash", "arguments": '{"command": "ls"}', "call_id": "call-ls"},
                {"type": "function_call", "name": "Bash", "arguments": '{"command": "pwd"}', "call_id": "call-pwd"},
                {"type": "function_call_output", "call_id": "call-ls", "output": "a.txt"},
                {"type": "function_call_output", "call_id": "call-pwd", "output": "/work"},
            ],
            id="openai-responses",
        ),
    ],
)
def test_parallel_tool_results_become_one_exchange_each_in_every_request_shape(messages: list[object]) -> None:
    assert tool_exchanges(messages) == (_LS, _PWD)


def test_a_result_already_answered_by_the_model_is_not_reported_again() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
        },
        {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-pwd", "function": {"name": "Bash", "arguments": '{"command": "pwd"}'}}],
        },
        {"role": "tool", "tool_call_id": "call-pwd", "content": "/work"},
    ]

    assert tool_exchanges(messages) == (_PWD,)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "user", "content": "hello"}], id="no-tools"),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "thanks"},
            ],
            id="result-before-the-last-model-turn",
        ),
        pytest.param([{"role": "tool", "tool_call_id": "unknown", "content": "x"}], id="result-without-its-call"),
        pytest.param(
            [{"role": "assistant", "tool_calls": [{"id": "call-ls", "function": {"name": "Bash"}}]}],
            id="call-without-result",
        ),
        pytest.param("a plain prompt", id="string-prompt"),
        pytest.param([7, None, {"role": 3}], id="malformed-items"),
        pytest.param(None, id="no-prompt"),
    ],
)
def test_a_prompt_with_no_new_paired_tool_result_yields_no_exchange(messages: object) -> None:
    assert tool_exchanges(messages) == ()


def test_structured_tool_output_is_kept_as_json() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-ls", "name": "Bash", "input": {"command": "ls"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-ls", "content": [{"type": "text", "text": "a.txt"}]}
            ],
        },
    ]

    assert tool_exchanges(messages) == (
        ToolExchange(
            call_id="call-ls", name="Bash", arguments='{"command": "ls"}', result='[{"type": "text", "text": "a.txt"}]'
        ),
    )


def test_a_model_call_payload_gives_the_agent_and_its_new_tool_results() -> None:
    payload = {
        "metadata": {"user_agent": "claude-cli/2.1.329"},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
        ],
    }

    assert agent_call(payload) == AgentCall(
        agent="claude-code", tool_exchanges=(_LS,), integration="claude-code", task=None, derived_session=None
    )


@pytest.mark.parametrize("payload", [None, "text", {"metadata": "broken"}])
def test_an_unreadable_payload_names_no_agent_and_no_tool_result(payload: object) -> None:
    assert agent_call(payload) == AgentCall()


def _recorded_payload(agent: str) -> dict[str, object]:
    """The logging payload the gateway builds from a request the agent really sent, recorded on the wire."""
    recorded = json.loads((Path(__file__).parent / "fixtures" / f"{agent}_gateway_request.json").read_text())
    body = recorded["body"]
    system = [{"role": "system", "content": body["system"]}] if "system" in body else []
    return {
        "metadata": {"user_agent": recorded["headers"]["user-agent"], "user_api_key_hash": "key-hash"},
        "messages": [*system, *(body.get("messages") or body["input"])],
    }


@pytest.mark.parametrize(
    ("recording", "agent", "integration", "task"),
    [
        ("claude_code", "claude-code", "claude-code", "Count the words in notes.txt and report the number"),
        ("codex", "codex-cli", "openai-codex", "Count the words in notes.txt and report the number"),
        ("opencode", "opencode", "opencode", '"Count the words in notes.txt and report the number"'),
        ("pi", "pi", "pi", "Count the words in notes.txt and report the number"),
    ],
)
def test_a_recorded_agent_request_gives_its_identity_and_the_task_the_user_typed(
    recording: str, agent: str, integration: str, task: str
) -> None:
    made_by = agent_call(_recorded_payload(recording))

    assert (made_by.agent, made_by.integration, made_by.task) == (agent, integration, task)
    assert len(made_by.tool_exchanges) == 1


def test_pi_is_recognised_by_its_system_prompt_because_it_sends_its_sdk_user_agent() -> None:
    payload = _recorded_payload("pi")

    assert agent_name(payload["metadata"]["user_agent"]) == "anthropic"
    assert agent_call(payload).agent == "pi"
    assert agent_call({**payload, "messages": payload["messages"][1:]}).agent == "anthropic"


def _first_call(later: dict[str, object], first_tool_call_id: str) -> dict[str, object]:
    """The opening call of a recorded conversation: no reply in the request yet, the reply is the response."""
    reply = {"role": "assistant", "content": "", "tool_calls": [{"id": first_tool_call_id, "function": {}}]}
    return {**later, "messages": later["messages"][:2], "response": {"choices": [{"message": reply}]}}


def test_calls_of_one_conversation_derive_one_session_and_another_task_or_key_derives_another() -> None:
    later = _recorded_payload("pi")
    first = _first_call(later, "toolu_01FuwdifsnyPSCzfzcMzYANw")
    other_task = {**later, "messages": [later["messages"][0], {"role": "user", "content": "Fix the build"}]}
    other_key = {**later, "metadata": {**later["metadata"], "user_api_key_hash": "other-key"}}

    session = agent_call(later).derived_session

    assert session is not None
    assert agent_call(first).derived_session == session
    assert agent_call(other_task).derived_session not in (None, session)
    assert agent_call(other_key).derived_session not in (None, session)


def test_the_same_prompt_run_again_is_a_new_session_because_its_first_reply_differs() -> None:
    run = _recorded_payload("pi")
    rerun_first_call = _first_call(run, "toolu_another_run")

    assert agent_call(rerun_first_call).derived_session not in (None, agent_call(run).derived_session)


def test_a_first_reply_with_no_tool_call_is_told_apart_by_its_text() -> None:
    run = _recorded_payload("pi")
    opening = run["messages"][:2]
    answered = {**run, "messages": opening, "response": {"choices": [{"message": {"content": "2 words"}}]}}
    follow_up = {
        **run,
        "messages": [*opening, {"role": "assistant", "content": [{"type": "text", "text": "2 words"}]}],
    }
    other_answer = {**answered, "response": {"choices": [{"message": {"content": "3 words"}}]}}

    session = agent_call(answered).derived_session

    assert agent_call(follow_up).derived_session == session
    assert agent_call(other_answer).derived_session not in (None, session)


def test_an_unknown_agent_gets_its_task_but_no_logo_and_no_derived_session() -> None:
    payload = {
        "metadata": {"user_agent": "my-agent/1.0"},
        "messages": [{"role": "user", "content": "<setup>cwd</setup>\n\nFix the build"}],
    }

    assert agent_call(payload) == AgentCall(agent="my-agent", task="Fix the build")


def test_the_session_the_caller_sent_wins_over_a_derived_one() -> None:
    made_by = agent_call(_recorded_payload("pi"))

    assert made_by.attributes(None)["gen_ai.conversation.id"] == made_by.derived_session
    assert "gen_ai.conversation.id" not in made_by.attributes("sent-by-caller")
    assert made_by.attributes("sent-by-caller") == {
        "gen_ai.agent.name": "pi",
        "langsmith.metadata.ls_integration": "pi",
        "lens.task": "Count the words in notes.txt and report the number",
    }


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param([{"role": "user", "content": "Fix the build"}], "Fix the build", id="plain"),
        pytest.param(
            [{"role": "user", "content": "<a>1</a>\n<b>2</b>\n\nFix the build"}], "Fix the build", id="several-blocks"
        ),
        pytest.param(
            [{"role": "user", "content": "<env>cwd</env>"}, {"role": "user", "content": "Fix the build"}],
            "Fix the build",
            id="setup-only-message-skipped",
        ),
        pytest.param(
            [{"role": "user", "content": "<div> does not render"}], "<div> does not render", id="unclosed-tag"
        ),
        pytest.param([{"role": "user", "content": "<env>cwd</env>"}], None, id="only-setup"),
        pytest.param([{"role": "assistant", "content": "hello"}], None, id="no-user-message"),
        pytest.param(
            [{"role": "user", "content": "<r>x</r>\n" * 50_000 + "Fix the build"}],
            "Fix the build",
            id="very-many-blocks",
        ),
        pytest.param("text", None, id="not-a-conversation"),
    ],
)
def test_the_task_is_the_first_user_text_after_leading_setup_blocks(messages: object, expected: str | None) -> None:
    assert user_task(messages) == expected


def test_the_title_call_opencode_makes_is_named_but_its_task_is_its_own_prompt() -> None:
    made_by = agent_call(_recorded_payload("opencode_title"))

    assert (made_by.agent, made_by.task) == ("opencode", "Generate a title for this conversation:")
