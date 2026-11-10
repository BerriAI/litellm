import base64
import copy
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.reasoning_utils import (
    DUMMY_GEMINI_THOUGHT_SIGNATURE,
    ENCRYPTED_REASONING_SIGNATURE_PREFIX,
    GEMINI_THOUGHT_SIGNATURE_PREFIX,
    converse_redacted_content,
    encrypted_content_from_signature,
    encrypted_reasoning_signature,
    gemini_thought_signature,
    is_bridge_reasoning_block,
    is_encrypted_reasoning_block,
    is_gemini_thought_signature_block,
    is_unsignable_thinking_block,
    native_redacted_thinking_data,
    responses_reasoning_items_from_thinking_blocks,
    strip_encrypted_reasoning_from_messages,
    thought_signature_of_block,
)


class TestEncryptedReasoningReplay:
    """Regression for https://github.com/BerriAI/litellm/issues/40288."""

    def test_signature_round_trips_the_encrypted_content(self):
        assert encrypted_content_from_signature(encrypted_reasoning_signature("gAAAA_bytes")) == "gAAAA_bytes"

    @pytest.mark.parametrize(
        "signature", [None, "", "ErcBCkgIValidAnthropicSignature", "litellm_encrypted_reasoning:", 7]
    )
    def test_anything_else_is_not_encrypted_content(self, signature):
        assert encrypted_content_from_signature(signature) is None

    def test_encrypted_thinking_block_replays_its_own_item(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [{"type": "thinking", "thinking": "Plan.", "signature": encrypted_reasoning_signature("gAAAA_1")}]
        )
        assert items == (
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Plan."}],
                "encrypted_content": "gAAAA_1",
            },
        )

    def test_encrypted_redacted_block_replays_with_an_empty_summary(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [{"type": "redacted_thinking", "data": encrypted_reasoning_signature("gAAAA_1")}]
        )
        assert items == ({"type": "reasoning", "summary": [], "encrypted_content": "gAAAA_1"},)

    def test_plain_blocks_collapse_into_one_summary_item_around_encrypted_ones(self):
        items = responses_reasoning_items_from_thinking_blocks(
            [
                {"type": "thinking", "thinking": "A.", "signature": None},
                {"type": "thinking", "thinking": "B.", "signature": ""},
                {"type": "thinking", "thinking": "C.", "signature": encrypted_reasoning_signature("gAAAA_c")},
                {"type": "redacted_thinking", "data": "anthropic-minted-opaque-data"},
                {"type": "thinking", "thinking": "D."},
            ]
        )
        assert items == (
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "A."}, {"type": "summary_text", "text": "B."}],
            },
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "C."}], "encrypted_content": "gAAAA_c"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "D."}]},
        )
        assert all("id" not in item for item in items)

    def test_blocks_without_text_or_encrypted_content_produce_nothing(self):
        assert responses_reasoning_items_from_thinking_blocks([{"type": "thinking", "thinking": ""}]) == ()
        assert responses_reasoning_items_from_thinking_blocks([]) == ()

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ({"type": "thinking", "thinking": "x", "signature": encrypted_reasoning_signature("g")}, True),
            ({"type": "redacted_thinking", "data": encrypted_reasoning_signature("g")}, True),
            ({"type": "thinking", "thinking": "x", "signature": ENCRYPTED_REASONING_SIGNATURE_PREFIX}, True),
            ({"type": "redacted_thinking", "data": ENCRYPTED_REASONING_SIGNATURE_PREFIX}, True),
            ({"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}, False),
            ({"type": "redacted_thinking", "data": "EmwKAhgBEgy"}, False),
            ({"type": "text", "text": encrypted_reasoning_signature("g")}, False),
            ("not a block", False),
        ],
    )
    def test_is_encrypted_reasoning_block(self, block, expected):
        assert is_encrypted_reasoning_block(block) is expected

    def test_strip_drops_every_bridge_tagged_block_and_leaves_no_unsigned_thinking_behind(self):
        assistant_content = [
            {"type": "thinking", "thinking": "minted by Anthropic", "signature": "ErcBCkgIValid"},
            {"type": "thinking", "thinking": "packed by the bridge", "signature": encrypted_reasoning_signature("g1")},
            {"type": "redacted_thinking", "data": encrypted_reasoning_signature("g2")},
            {"type": "thinking", "thinking": "", "signature": encrypted_reasoning_signature("g3")},
            {"type": "text", "text": "answer"},
        ]
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
        ]

        strip_encrypted_reasoning_from_messages(messages)

        assert messages[1]["content"] is assistant_content
        assert assistant_content == [
            {"type": "thinking", "thinking": "minted by Anthropic", "signature": "ErcBCkgIValid"},
            {"type": "text", "text": "answer"},
        ]
        assert all(block["signature"] for block in assistant_content if block["type"] == "thinking")
        assert messages[0] == {"role": "user", "content": "question"}
        assert messages[2] == {"role": "user", "content": [{"type": "text", "text": "follow-up"}]}

    def test_strip_uses_predicate_to_keep_selected_encrypted_blocks(self):
        kept_signature = encrypted_reasoning_signature("keep")
        stripped_signature = encrypted_reasoning_signature("strip")
        content = [
            {"type": "thinking", "thinking": "keep", "signature": kept_signature},
            {"type": "thinking", "thinking": "strip", "signature": stripped_signature},
        ]
        messages = [{"role": "assistant", "content": content}]

        strip_encrypted_reasoning_from_messages(messages, should_strip=lambda block: block.get("thinking") == "strip")

        assert messages[0]["content"] is content
        assert content == [{"type": "thinking", "thinking": "keep", "signature": kept_signature}]

    @pytest.mark.parametrize(
        "messages",
        [
            "not a list",
            None,
            [{"role": "user", "content": None}],
            [{"role": "user", "content": "plain string"}],
            ["not a message"],
            [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}]}],
        ],
    )
    def test_strip_leaves_history_without_bridge_reasoning_untouched(self, messages):
        before = copy.deepcopy(messages)

        strip_encrypted_reasoning_from_messages(messages)

        assert messages == before


_NATIVE_REDACTED_DATA: Final = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP"


_CONVERSE_REDACTED_DATA: Final = base64.b64encode(_NATIVE_REDACTED_DATA.encode()).decode()


@pytest.mark.parametrize(
    "data, native",
    [
        pytest.param(_NATIVE_REDACTED_DATA, _NATIVE_REDACTED_DATA, id="claude-native"),
        pytest.param(_CONVERSE_REDACTED_DATA, _NATIVE_REDACTED_DATA, id="converse-wrapped-legacy-history"),
        pytest.param(
            encrypted_reasoning_signature("gAAAA_1"), encrypted_reasoning_signature("gAAAA_1"), id="openai-encrypted"
        ),
        pytest.param("", "", id="blank"),
        pytest.param(base64.b64encode(b"not base64!").decode(), base64.b64encode(b"not base64!").decode(), id="b64-once"),
    ],
)
def test_native_redacted_thinking_data_only_unwraps_the_converse_base64_layer(data: str, native: str) -> None:
    assert native_redacted_thinking_data(data) == native


@pytest.mark.parametrize("data", [_NATIVE_REDACTED_DATA, _CONVERSE_REDACTED_DATA], ids=["native", "legacy-wrapped"])
def test_converse_redacted_content_wraps_claude_native_data_exactly_once(data: str) -> None:
    assert converse_redacted_content(data) == _CONVERSE_REDACTED_DATA


_GEMINI_SIG: Final = "EoUMCoIMAWkUfRNAt6WiTWGx73C1cgarbMkd0STsaBTv0DSrICGKWeyBN9og=="


def test_gemini_thought_signature_helpers_distinguish_bridge_dummy_and_claude_blocks() -> None:
    wrapped: Final = gemini_thought_signature(_GEMINI_SIG)
    assert wrapped == f"{GEMINI_THOUGHT_SIGNATURE_PREFIX}{_GEMINI_SIG}"
    assert gemini_thought_signature(wrapped) == wrapped

    thinking_block: Final = {"type": "thinking", "thinking": "plan", "signature": wrapped}
    redacted_block: Final = {"type": "redacted_thinking", "data": wrapped}
    dummy_block: Final = {
        "type": "thinking",
        "thinking": "plan",
        "signature": gemini_thought_signature(DUMMY_GEMINI_THOUGHT_SIGNATURE),
    }
    claude_block: Final = {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_claude"}

    assert is_gemini_thought_signature_block(thinking_block) is True
    assert is_gemini_thought_signature_block(redacted_block) is True
    assert is_bridge_reasoning_block(thinking_block) is True
    assert is_bridge_reasoning_block(redacted_block) is True
    assert is_unsignable_thinking_block(thinking_block) is True
    assert is_unsignable_thinking_block(redacted_block) is True
    assert thought_signature_of_block(thinking_block) == _GEMINI_SIG
    assert thought_signature_of_block(redacted_block) == _GEMINI_SIG
    assert thought_signature_of_block(dummy_block) is None

    assert is_gemini_thought_signature_block(claude_block) is False
    assert is_bridge_reasoning_block(claude_block) is False
    assert is_unsignable_thinking_block(claude_block) is False
    assert thought_signature_of_block(claude_block) is None

    assert responses_reasoning_items_from_thinking_blocks([thinking_block, redacted_block]) == (
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
    )
