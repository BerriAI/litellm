import json
from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.compaction_items import (
    compaction_output_item,
    holds_compaction_summary,
    response_holds_compaction_block,
    with_compaction_summary,
)
from litellm.responses.utils import ResponsesAPIRequestUtils

_BLOCK: Final = {"type": "compaction", "content": "Turn one planned Tokyo."}
_ITEM: Final = {"type": "compaction", "id": "cmp_1", "encrypted_content": json.dumps(_BLOCK)}


@pytest.mark.parametrize(
    "newer_entry",
    [
        pytest.param("not a block", id="not_a_mapping"),
        pytest.param({"type": "compaction", "content": None}, id="summary_not_streamed_yet"),
        pytest.param({"type": "compaction", "content": ""}, id="empty_summary"),
        pytest.param({"type": "text", "content": "Hi."}, id="not_a_compaction"),
    ],
)
def test_compaction_output_item_carries_the_newest_block_that_holds_a_summary(newer_entry: object) -> None:
    item: Final = compaction_output_item({"compaction_blocks": [_BLOCK, newer_entry]})

    assert item is not None
    assert json.loads(item.encrypted_content) == _BLOCK


@pytest.mark.parametrize(
    ("response", "holds"),
    [
        pytest.param(
            {"choices": [{"message": {"role": "assistant", "provider_specific_fields": {"compaction_blocks": [_BLOCK]}}}]},
            True,
            id="chat_answer_compacted",
        ),
        pytest.param({"choices": [{"message": {"role": "assistant", "content": "Hi."}}]}, False, id="chat_answer"),
        pytest.param({"object": "response", "output": [_ITEM]}, True, id="responses_output_compacted"),
        pytest.param(
            {"object": "response", "output": [{"type": "compaction", "id": "cmp_openai", "encrypted_content": "gAAAA"}]},
            False,
            id="responses_output_with_an_opaque_compaction",
        ),
        pytest.param({"choices": "not a list"}, False, id="malformed_choices"),
        pytest.param(None, False, id="no_stored_response"),
    ],
)
def test_response_holds_compaction_block_reads_chat_and_responses_shaped_rows(response: object, holds: bool) -> None:
    assert response_holds_compaction_block(response) is holds


@pytest.mark.parametrize(
    ("encrypted_content", "holds"),
    [
        pytest.param(json.dumps(_BLOCK), True, id="litellm_block"),
        pytest.param(
            ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(json.dumps(_BLOCK), "deployment-1"),
            True,
            id="affinity_wrapped_block",
        ),
        pytest.param("gAAAAABopaque", False, id="openai_token"),
        pytest.param(None, False, id="missing"),
    ],
)
def test_only_a_compaction_item_litellm_can_decode_holds_a_summary(encrypted_content: object, holds: bool) -> None:
    assert holds_compaction_summary([{"type": "compaction", "id": "cmp_1", "encrypted_content": encrypted_content}]) is (
        holds
    )


def test_with_compaction_summary_rewrites_the_summary_and_keeps_the_affinity_wrap() -> None:
    wrapped: Final = ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(json.dumps(_BLOCK), "deployment-1")

    rewritten: Final = with_compaction_summary(wrapped, "[REDACTED]")
    model_id, original = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(rewritten)

    assert model_id == "deployment-1"
    assert json.loads(original) == {**_BLOCK, "content": "[REDACTED]"}


def test_with_compaction_summary_leaves_a_token_it_cannot_read_unchanged() -> None:
    assert with_compaction_summary("gAAAAABopaque", "[REDACTED]") == "gAAAAABopaque"
