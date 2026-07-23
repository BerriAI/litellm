"""Offline tests for provider-aware native Skills IDs."""

from litellm.litellm_core_utils.skill_id_utils import (
    decode_skill_id,
    encode_skill_id,
    get_model_from_skill_id,
    get_original_skill_id,
    rewrite_skill_references,
)


def test_skill_id_round_trip_and_response_reference_rewrite():
    wrapped_id = encode_skill_id("skill_native", "openai-account")

    assert decode_skill_id(wrapped_id) == {"id": "skill_native", "model": "openai-account"}
    assert get_original_skill_id(wrapped_id) == "skill_native"
    assert get_model_from_skill_id(wrapped_id) == "openai-account"
    assert rewrite_skill_references({"container": {"skills": [{"skill_id": wrapped_id}]}}) == {
        "container": {"skills": [{"skill_id": "skill_native"}]}
    }
