import json

from litellm.responses.litellm_completion_transformation.reasoning_items import (
    decode_thinking_blocks,
    encode_thinking_blocks,
    is_litellm_minted_reasoning_item,
    is_minted_reasoning_item_id,
    mint_reasoning_item_id,
)

A_PROVIDER_OWNED_REASONING_ITEM_ID = "rs_08d3a89dbb92277a006abf04f4266087d0b4eedacd7848f306"
A_PROVIDER_OWNED_ENCRYPTED_BLOB = "gAAAAABo-opaque-provider-blob"
SIGNED_BLOCK = {"type": "thinking", "thinking": "Paris first.", "signature": "sig-paris"}
UNSIGNED_BLOCK = {"type": "thinking", "thinking": "never signed"}
REDACTED_BLOCK = {"type": "redacted_thinking", "data": "opaque"}


def test_minted_ids_are_recognized_and_provider_owned_ids_are_not():
    minted = mint_reasoning_item_id()
    assert is_minted_reasoning_item_id(minted)
    assert not is_minted_reasoning_item_id(A_PROVIDER_OWNED_REASONING_ITEM_ID)
    assert not is_minted_reasoning_item_id(minted.replace("-", ""))
    assert not is_minted_reasoning_item_id(minted.removeprefix("rs_"))
    assert not is_minted_reasoning_item_id(None)


def test_encoded_thinking_blocks_decode_back_to_the_verifiable_blocks_only():
    encoded = encode_thinking_blocks([SIGNED_BLOCK, UNSIGNED_BLOCK, REDACTED_BLOCK])
    assert encoded is not None
    assert decode_thinking_blocks(encoded) == (SIGNED_BLOCK, REDACTED_BLOCK)
    assert encode_thinking_blocks([UNSIGNED_BLOCK]) is None
    assert decode_thinking_blocks(A_PROVIDER_OWNED_ENCRYPTED_BLOB) is None
    assert decode_thinking_blocks(json.dumps(SIGNED_BLOCK)) is None
    assert decode_thinking_blocks(json.dumps([{"type": "text", "text": "not thinking"}])) is None


def test_decoding_keeps_the_verifiable_blocks_of_a_mixed_array_and_skips_the_rest():
    mixed = json.dumps([SIGNED_BLOCK, "a stray string", 7, None, UNSIGNED_BLOCK, {"type": "thinking"}, REDACTED_BLOCK])
    assert decode_thinking_blocks(mixed) == (SIGNED_BLOCK, REDACTED_BLOCK)
    assert decode_thinking_blocks(json.dumps(["only", "strings", 3])) is None
    assert decode_thinking_blocks(json.dumps([UNSIGNED_BLOCK])) is None


def test_a_reasoning_item_is_litellm_minted_by_its_id_or_by_its_encoded_thinking_blocks():
    assert is_litellm_minted_reasoning_item({"type": "reasoning", "id": mint_reasoning_item_id(), "summary": []})
    assert is_litellm_minted_reasoning_item(
        {
            "type": "reasoning",
            "id": A_PROVIDER_OWNED_REASONING_ITEM_ID,
            "encrypted_content": encode_thinking_blocks([SIGNED_BLOCK]),
        }
    )
    assert not is_litellm_minted_reasoning_item(
        {
            "type": "reasoning",
            "id": A_PROVIDER_OWNED_REASONING_ITEM_ID,
            "summary": [],
            "encrypted_content": A_PROVIDER_OWNED_ENCRYPTED_BLOB,
        }
    )
    assert not is_litellm_minted_reasoning_item({"type": "message", "id": mint_reasoning_item_id(), "role": "assistant"})
    assert not is_litellm_minted_reasoning_item("a bare string input")
