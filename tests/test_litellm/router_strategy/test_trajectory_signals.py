"""
Tests for tool-trajectory signals: the four fractions the Complexity Router reads off the
assistant's own recent tool calls, across both Anthropic Messages and chat-completions shapes.
"""

import json

import pytest

from litellm.router_strategy.complexity_router.trajectory_signals import (
    _MAX_SIGNATURE_CHARS,
    compute_trajectory_signals,
    resolve_tool_intent,
    tool_call_signature,
)


def _anthropic_call(call_id: str, name: str, arguments: dict, *, is_error: bool) -> list[dict]:
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "is_error": is_error, "content": "result"}],
        },
    ]


def _chat_completions_call(call_id: str, name: str, arguments_json: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments_json}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "result"},
    ]


class TestToolCallSignatureIsBounded:
    """A caller controls tool arguments, and the window holds one signature per call, so an
    oversized argument must not become an oversized signature held several times over."""

    @staticmethod
    def _big_arguments(marker: str) -> dict:
        return {"path": marker, "blob": "x" * (_MAX_SIGNATURE_CHARS * 4)}

    def test_a_huge_argument_yields_a_bounded_signature(self):
        _, signature = tool_call_signature("write_file", self._big_arguments("a"))
        assert len(signature) <= _MAX_SIGNATURE_CHARS
        assert signature.startswith("sha256:")

    def test_a_normal_argument_keeps_its_readable_form(self):
        _, signature = tool_call_signature("write_file", {"path": "a.txt"})
        assert signature == '{"path": "a.txt"}'

    def test_two_different_huge_arguments_stay_distinguishable(self):
        """Bounding must not collapse distinct calls into one, or a busy agent would read as
        spinning purely because its arguments were large."""
        _, first = tool_call_signature("write_file", self._big_arguments("a"))
        _, second = tool_call_signature("write_file", self._big_arguments("b"))
        assert first != second

    def test_the_same_huge_argument_still_compares_equal(self):
        _, first = tool_call_signature("write_file", self._big_arguments("a"))
        _, second = tool_call_signature("write_file", self._big_arguments("a"))
        assert first == second

    def test_a_huge_argument_still_matches_across_surfaces(self):
        """The cross-surface guarantee has to survive the bound: the same call sent as a dict
        and as a JSON string is still one repeated call."""
        arguments = self._big_arguments("a")
        assert tool_call_signature("write_file", arguments) == tool_call_signature(
            "write_file", json.dumps(arguments)
        )


class TestResolveToolIntent:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("read_file", "read"),
            ("Read", "read"),
            ("Glob", "read"),
            ("WebFetch", "read"),
            ("get_issue", "read"),
            ("write_file", "write"),
            ("Edit", "write"),
            ("NotebookEdit", "write"),
            ("str_replace_editor", "write"),
            ("bash", "execute"),
            ("run_tests", "execute"),
            ("Terminal", "execute"),
        ],
    )
    def test_common_tool_names_resolve_by_verb(self, name: str, expected: str):
        assert resolve_tool_intent(name) == expected

    def test_verbs_match_name_tokens_not_substrings(self):
        """'spreadsheet' contains 'read' and 'thread' contains 'read', but neither is a read."""
        assert resolve_tool_intent("spreadsheet") == "unknown"
        assert resolve_tool_intent("thread_summary") == "unknown"

    def test_a_writing_tool_that_also_runs_counts_as_writing(self):
        assert resolve_tool_intent("apply_patch_and_run") == "write"

    def test_unrecognized_names_are_unknown(self):
        assert resolve_tool_intent("frobnicate") == "unknown"
        assert resolve_tool_intent("") == "unknown"

    def test_operator_override_wins_over_the_verbs(self):
        assert resolve_tool_intent("Glob", {"glob": "write"}) == "write"

    def test_override_lets_an_operator_name_their_own_tool(self):
        assert resolve_tool_intent("frobnicate", {"frobnicate": "write"}) == "write"


class TestComputeTrajectorySignals:
    def test_no_messages_reports_no_evidence(self):
        for messages in (None, []):
            signals = compute_trajectory_signals(messages, window=6)
            assert signals.observed_calls == 0
            assert signals.error_severity == 0.0
            assert signals.spinning == 0.0

    def test_a_turn_with_no_tool_calls_reports_no_evidence(self):
        """Zero signals with observed_calls 0 is 'nothing to read', not 'nothing happening'."""
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert compute_trajectory_signals(messages, window=6).observed_calls == 0

    def test_error_severity_is_the_fraction_of_failing_calls(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "a"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "b"}, is_error=True),
            *_anthropic_call("t3", "bash", {"cmd": "c"}, is_error=True),
            *_anthropic_call("t4", "bash", {"cmd": "d"}, is_error=False),
        ]
        assert compute_trajectory_signals(messages, window=4).error_severity == 0.75

    def test_error_severity_is_zero_on_chat_completions(self):
        """A chat-completions tool message carries no standard error flag, so a reading taken
        on that surface rests on the other three signals."""
        messages = [
            *_chat_completions_call("c1", "bash", '{"cmd": "a"}'),
            *_chat_completions_call("c2", "bash", '{"cmd": "b"}'),
        ]
        signals = compute_trajectory_signals(messages, window=6)
        assert signals.error_severity == 0.0
        assert signals.observed_calls == 2

    def test_spinning_is_high_when_the_same_call_repeats(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t4", "bash", {"cmd": "pytest"}, is_error=False),
        ]
        assert compute_trajectory_signals(messages, window=4).spinning == 0.75

    def test_spinning_is_zero_when_every_call_differs(self):
        messages = [
            *_anthropic_call("t1", "read_file", {"path": "a"}, is_error=False),
            *_anthropic_call("t2", "read_file", {"path": "b"}, is_error=False),
            *_anthropic_call("t3", "read_file", {"path": "c"}, is_error=False),
        ]
        assert compute_trajectory_signals(messages, window=6).spinning == 0.0

    def test_the_same_call_counts_as_a_repeat_across_surfaces(self):
        """Arguments arrive as a dict on one surface and a JSON string on the other."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_chat_completions_call("c2", "bash", '{"cmd": "pytest"}'),
        ]
        assert compute_trajectory_signals(messages, window=6).spinning == 0.5

    def test_exploring_counts_reads_and_production_counts_writes(self):
        messages = [
            *_anthropic_call("t1", "read_file", {"path": "a"}, is_error=False),
            *_anthropic_call("t2", "read_file", {"path": "b"}, is_error=False),
            *_anthropic_call("t3", "read_file", {"path": "c"}, is_error=False),
            *_anthropic_call("t4", "write_file", {"path": "d"}, is_error=False),
        ]
        signals = compute_trajectory_signals(messages, window=4)
        assert signals.exploring == 0.75
        assert signals.production_intensity == 0.25

    def test_executing_counts_toward_neither_exploring_nor_production(self):
        """Running something is ambiguous between verifying work and churning on it, so it
        contributes to no signal rather than being guessed into one."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "ruff"}, is_error=False),
        ]
        signals = compute_trajectory_signals(messages, window=6)
        assert signals.exploring == 0.0
        assert signals.production_intensity == 0.0
        assert signals.observed_calls == 2

    def test_unknown_tools_contribute_to_neither_signal(self):
        messages = [
            *_anthropic_call("t1", "frobnicate", {"x": 1}, is_error=False),
            *_anthropic_call("t2", "write_file", {"path": "a"}, is_error=False),
        ]
        signals = compute_trajectory_signals(messages, window=6)
        assert signals.exploring == 0.0
        assert signals.production_intensity == 0.5

    def test_operator_intents_reclassify_their_own_tools(self):
        messages = [
            *_anthropic_call("t1", "frobnicate", {"x": 1}, is_error=False),
            *_anthropic_call("t2", "frobnicate", {"x": 2}, is_error=False),
        ]
        signals = compute_trajectory_signals(messages, window=6, tool_intents={"frobnicate": "write"})
        assert signals.production_intensity == 1.0

    def test_only_the_newest_window_of_calls_is_read(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t3", "read_file", {"path": "a"}, is_error=False),
            *_anthropic_call("t4", "read_file", {"path": "b"}, is_error=False),
        ]
        signals = compute_trajectory_signals(messages, window=2)
        assert signals.observed_calls == 2
        assert signals.error_severity == 0.0
        assert signals.exploring == 1.0

    def test_a_zero_window_reports_no_evidence(self):
        messages = [*_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=True)]
        assert compute_trajectory_signals(messages, window=0).observed_calls == 0

    def test_an_error_is_read_off_a_result_that_arrives_several_messages_later(self):
        """The result answering a call need not be the very next message. The backward walk
        collects results as it goes, so anything newer than the call is still matched to it."""
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"c": "a"}}]},
            {"role": "assistant", "content": "let me check that"},
            {"role": "user", "content": "ok"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "boom"}],
            },
        ]
        assert compute_trajectory_signals(messages, window=6).error_severity == 1.0

    def test_evidence_survives_a_new_human_ask(self):
        """A plain follow-up must not erase the tool calls before it, matching how stall
        detection reads the whole visible conversation."""
        messages = [
            *_anthropic_call("t1", "write_file", {"path": "a"}, is_error=False),
            {"role": "user", "content": [{"type": "text", "text": "try again"}]},
        ]
        assert compute_trajectory_signals(messages, window=6).production_intensity == 1.0
