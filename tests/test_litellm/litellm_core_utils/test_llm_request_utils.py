"""Unit tests for the extra_body merge guards in llm_request_utils.

VERIA-130/151/177: a client-supplied ``extra_body`` must not overwrite the
validated request fields (model/messages/input/safetySettings/...) the provider
transform already produced, while genuine provider passthrough still works.
"""

from litellm.litellm_core_utils.llm_request_utils import (
    safe_merge_extra_body,
    strip_validated_keys_from_extra_body,
)


class TestSafeMergeExtraBody:
    def test_none_or_empty_extra_body_is_noop(self):
        assert safe_merge_extra_body({"model": "m"}, None) == {"model": "m"}
        assert safe_merge_extra_body({"model": "m"}, {}) == {"model": "m"}

    def test_validated_key_is_not_overwritten(self):
        data = {"model": "authorized", "messages": [{"role": "user", "content": "x"}]}
        result = safe_merge_extra_body(
            data,
            {"model": "premium", "messages": [{"role": "user", "content": "evil"}]},
        )
        assert result["model"] == "authorized"
        assert result["messages"] == [{"role": "user", "content": "x"}]

    def test_new_key_passes_through(self):
        result = safe_merge_extra_body(
            {"model": "m"}, {"provider_flag": True, "route": "fallback"}
        )
        assert result["provider_flag"] is True
        assert result["route"] == "fallback"

    def test_nested_dict_collision_validated_wins_new_subkeys_merge(self):
        data = {"generationConfig": {"temperature": 0.0, "topK": 5}}
        result = safe_merge_extra_body(
            data, {"generationConfig": {"temperature": 2.0, "candidateCount": 4}}
        )
        # validated sub-keys win on collision; genuinely new sub-keys are added
        assert result["generationConfig"]["temperature"] == 0.0
        assert result["generationConfig"]["topK"] == 5
        assert result["generationConfig"]["candidateCount"] == 4

    def test_scalar_and_dict_collisions_validated_wins(self):
        data = {"compartmentId": "tenant-a", "servingMode": {"modelId": "m"}}
        result = safe_merge_extra_body(
            data, {"compartmentId": "attacker", "servingMode": {"modelId": "other"}}
        )
        assert result["compartmentId"] == "tenant-a"
        assert result["servingMode"]["modelId"] == "m"


class TestStripValidatedKeysFromExtraBody:
    def test_strips_model_and_messages_from_nested_extra_body(self):
        body = {
            "model": "m",
            "messages": [1],
            "extra_body": {"model": "evil", "messages": [2], "keep": 1},
        }
        result = strip_validated_keys_from_extra_body(body)
        assert result["extra_body"] == {"keep": 1}
        assert result["model"] == "m"

    def test_no_extra_body_is_noop_same_object(self):
        body = {"model": "m"}
        assert strip_validated_keys_from_extra_body(body) is body

    def test_extra_body_without_protected_keys_unchanged(self):
        body = {"model": "m", "extra_body": {"reasoning_effort": "high"}}
        result = strip_validated_keys_from_extra_body(body)
        assert result["extra_body"] == {"reasoning_effort": "high"}

    def test_custom_keys(self):
        body = {"extra_body": {"input": "evil", "keep": 1}}
        result = strip_validated_keys_from_extra_body(body, keys=("input",))
        assert result["extra_body"] == {"keep": 1}

    def test_does_not_mutate_original(self):
        original = {"model": "m", "extra_body": {"model": "evil", "keep": 1}}
        strip_validated_keys_from_extra_body(original)
        # the input's nested extra_body must be left untouched (a copy is returned)
        assert original["extra_body"] == {"model": "evil", "keep": 1}
