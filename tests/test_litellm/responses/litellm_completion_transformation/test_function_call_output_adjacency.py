"""
Tests for ``_reorder_function_call_outputs_adjacent``.

Bedrock Converse rejects requests where a ``toolResult`` is not located in
the ``user`` message *immediately* following the assistant message that
emitted the matching ``toolUse``. The OpenAI Responses API itself does not
require strict adjacency, so upstream clients can legally inject a
``message`` between a ``function_call`` and its ``function_call_output``.

The helper normalises the Responses-API input so the converted Chat
Completions / Bedrock Converse payload always satisfies the strictest
contract. These tests pin the helper's behaviour at the
Responses → Chat Completions boundary.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

reorder = LiteLLMCompletionResponsesConfig._reorder_function_call_outputs_adjacent


def _function_call(call_id: str, name: str = "exec_command") -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _function_call_output(call_id: str, output: str = "ok") -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _user_message(text: str) -> dict:
    return {"type": "message", "role": "user", "content": text}


def _assistant_message(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": text}


def test_reorder_handles_empty_input():
    assert reorder([]) == []


def test_reorder_passes_through_non_list_input():
    # The helper must tolerate the non-list shapes the surrounding code
    # may pass during validation/edge cases.
    assert reorder(None) is None  # type: ignore[arg-type]


def test_reorder_is_noop_when_already_adjacent():
    items = [_function_call("c1"), _function_call_output("c1")]
    out = reorder(items)
    assert out == items


def test_reorder_is_idempotent_for_already_adjacent_items():
    items = [_function_call("c1"), _function_call_output("c1")]
    assert reorder(reorder(items)) == items


def test_reorder_moves_output_adjacent_to_call_when_message_is_interleaved():
    """Reproduces the OpenAI Codex CLI ``apply_patch`` heredoc trigger.

    Codex detects an ``apply_patch <<EOF`` heredoc inside an
    ``exec_command``, short-circuits via the native ``apply_patch`` tool,
    and inserts a user-visible warning message between the
    ``function_call`` and the synthesised ``function_call_output``.

    The helper must restore adjacency so Bedrock Converse accepts the
    converted request.
    """
    apply_patch_warning = (
        "Warning: apply_patch was requested via exec_command. Use the "
        "apply_patch tool instead of exec_command."
    )
    items = [
        _user_message("hello"),
        _function_call("c1"),
        _user_message(apply_patch_warning),
        _function_call_output("c1"),
    ]

    out = reorder(items)

    assert out == [
        _user_message("hello"),
        _function_call("c1"),
        _function_call_output("c1"),
        _user_message(apply_patch_warning),
    ]


def test_reorder_is_idempotent_after_a_real_reorder():
    items = [
        _function_call("c1"),
        _user_message("interlude"),
        _function_call_output("c1"),
    ]
    once = reorder(items)
    twice = reorder(once)
    assert twice == once


def test_reorder_handles_multiple_interleaved_pairs():
    items = [
        _function_call("a"),
        _assistant_message("thinking..."),
        _function_call("b"),
        _user_message("interrupt"),
        _function_call_output("a"),
        _function_call_output("b"),
    ]

    out = reorder(items)

    def index_of(call_id: str, kind: str) -> int:
        for i, it in enumerate(out):
            if (
                isinstance(it, dict)
                and it.get("type") == kind
                and it.get("call_id") == call_id
            ):
                return i
        raise AssertionError(f"missing {kind} for {call_id}: {out!r}")

    # Each function_call_output sits immediately after its matching
    # function_call.
    assert index_of("a", "function_call_output") == index_of("a", "function_call") + 1
    assert index_of("b", "function_call_output") == index_of("b", "function_call") + 1


def test_reorder_leaves_orphan_outputs_in_place():
    """Orphan ``function_call_output`` items keep their original position.

    We deliberately do *not* drop them: that's a separate concern (clients
    sending a malformed history) and is handled in a different normalisation
    step.
    """
    items = [
        _function_call_output("orphan"),
        _user_message("hi"),
    ]
    assert reorder(items) == items


def test_reorder_tolerates_non_dict_items():
    class Sentinel:
        pass

    items = [
        Sentinel(),
        _function_call("c1"),
        _function_call_output("c1"),
    ]
    out = reorder(items)
    # Sentinel is preserved at position 0; the call/output pair stays
    # adjacent (here a no-op since they were already adjacent).
    assert isinstance(out[0], Sentinel)
    assert out[1] == _function_call("c1")
    assert out[2] == _function_call_output("c1")


def test_reorder_is_a_noop_when_no_matching_call_exists():
    """A ``function_call_output`` whose ``call_id`` matches no preceding
    ``function_call`` must stay where it is rather than being silently moved
    to a wrong location.
    """
    items = [
        _user_message("first"),
        _function_call_output("ghost"),
    ]
    assert reorder(items) == items


def test_reorder_keeps_relative_order_of_unrelated_items():
    items = [
        _user_message("u1"),
        _function_call("c1"),
        _assistant_message("a1"),
        _user_message("u2"),
        _function_call_output("c1"),
        _assistant_message("a2"),
    ]

    out = reorder(items)

    # function_call_output landed right after function_call
    fc_idx = out.index(_function_call("c1"))
    assert out[fc_idx + 1] == _function_call_output("c1")
    # The other items keep their relative order
    other = [it for it in out if it.get("type") == "message"]
    assert other == [
        _user_message("u1"),
        _assistant_message("a1"),
        _user_message("u2"),
        _assistant_message("a2"),
    ]
