"""
Tests for mid-task stall detection: repeated identical tool calls or repeated tool
errors, read from both Anthropic Messages and chat-completions tool-call shapes.
"""

from litellm.router_strategy.complexity_router.stall_detector import detect_stalled_task


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


class TestDetectStalledTask:
    def test_repeated_identical_anthropic_calls_are_stalled(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_repeated_errors_are_stalled_even_with_varied_arguments(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest tests/a.py"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "pytest tests/b.py"}, is_error=True),
            *_anthropic_call("t3", "bash", {"cmd": "pytest tests/c.py"}, is_error=True),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_varied_successful_calls_are_not_stalled(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "ls"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t3", "grep", {"pattern": "x"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is False

    def test_chat_completions_repeats_are_stalled(self):
        messages = [
            *_chat_completions_call("c1", "bash", '{"cmd": "pytest"}'),
            *_chat_completions_call("c2", "bash", '{"cmd": "pytest"}'),
            *_chat_completions_call("c3", "bash", '{"cmd": "pytest"}'),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_chat_completions_has_no_structured_error_signal(self):
        """A chat-completions tool message carries no standard error flag, so varied calls
        whose content happens to read like failures still aren't flagged on error alone."""
        messages = [
            *_chat_completions_call("c1", "bash", '{"cmd": "a"}'),
            *_chat_completions_call("c2", "bash", '{"cmd": "b"}'),
            *_chat_completions_call("c3", "bash", '{"cmd": "c"}'),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is False

    def test_dict_and_json_string_arguments_compare_equal_across_surfaces(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_chat_completions_call("c2", "bash", '{"cmd": "pytest"}'),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_below_repeat_threshold_is_not_stalled(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is False

    def test_evidence_older_than_the_window_does_not_count(self):
        """Only the most recent `window` tool calls are considered, so a stall the model
        already recovered from does not keep re-triggering forever."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t4", "grep", {"pattern": "a"}, is_error=False),
            *_anthropic_call("t5", "grep", {"pattern": "b"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=2, repeat_threshold=2) is False

    def test_evidence_survives_a_new_human_ask(self):
        """A follow-up like 'try again' must not erase evidence from before it: detection
        reads the whole message list, not just the turns since the newest human ask."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=False),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=False),
            {"role": "user", "content": [{"type": "text", "text": "try again"}]},
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_a_recovered_task_is_not_stalled_while_its_old_failures_sit_in_the_window(self):
        """The three identical failures stay in the window for a few turns after the model
        breaks out of them, and counting them on their own would escalate a request that is
        already making progress again."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t4", "read_file", {"path": "conftest.py"}, is_error=False),
            *_anthropic_call("t5", "edit_file", {"path": "conftest.py"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is False

    def test_a_retry_loop_broken_up_by_an_unrelated_call_still_counts(self):
        """Anchoring on the newest call must not require the repeats to be adjacent: a model
        re-running the same failing command around a lookup in between is still stuck."""
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t2", "read_file", {"path": "conftest.py"}, is_error=False),
            *_anthropic_call("t3", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t4", "bash", {"cmd": "pytest"}, is_error=True),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is True

    def test_errors_only_count_while_the_newest_call_is_still_failing(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest a"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "pytest b"}, is_error=True),
            *_anthropic_call("t3", "bash", {"cmd": "pytest c"}, is_error=True),
            *_anthropic_call("t4", "bash", {"cmd": "pytest d"}, is_error=False),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=3) is False

    def test_no_messages_is_not_stalled(self):
        assert detect_stalled_task(None, window=6, repeat_threshold=3) is False
        assert detect_stalled_task([], window=6, repeat_threshold=3) is False

    def test_zero_threshold_never_flags_stalled(self):
        messages = [
            *_anthropic_call("t1", "bash", {"cmd": "pytest"}, is_error=True),
            *_anthropic_call("t2", "bash", {"cmd": "pytest"}, is_error=True),
        ]
        assert detect_stalled_task(messages, window=6, repeat_threshold=0) is False
