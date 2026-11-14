import pytest

from litellm.integrations.custom_logger_v2 import (
    Abort,
    Field,
    InvalidPatchError,
    MergeField,
    Patch,
    Respond,
    SetField,
)

F = Field


def test_patch_builders_return_new_patches_in_call_order() -> None:
    empty = Patch()
    patch = empty.set(F.messages, ["masked"]).merge(F.metadata, {"pii": True})
    assert empty.ops == ()
    assert patch.ops == (SetField(F.messages, ["masked"]), MergeField(F.metadata, {"pii": True}))
    assert patch.fields == frozenset({F.messages, F.metadata})


def test_merge_keeps_its_own_copy_of_the_keys() -> None:
    keys = {"tag": "a"}
    patch = Patch().merge(F.metadata, keys)
    keys["tag"] = "b"
    assert patch.ops == (MergeField(F.metadata, {"tag": "a"}),)


@pytest.mark.parametrize("field", [F.messages, F.response, F.model])
def test_merge_on_a_non_mapping_field_is_rejected(field: Field) -> None:
    with pytest.raises(InvalidPatchError, match=f"{field.value} is not a mapping field"):
        Patch().merge(field, {"k": "v"})


def test_abort_carries_a_json_body_and_a_guardrail_info_patch() -> None:
    body = {"error": {"message": "blocked", "violations": ["pii"]}}
    patch = Patch().merge(F.guardrail_info, {"pii_guard": "intervened"}).merge(F.metadata, {"blocked": True})
    abort = Abort(status=400, message="blocked", body=body, patch=patch)
    assert (abort.body, abort.patch) == (body, patch)


@pytest.mark.parametrize(
    "patch",
    [Patch().set(F.messages, []), Patch().merge(F.guardrail_info, {"g": 1}).set(F.response, None)],
)
def test_outcome_patch_outside_guardrail_info_and_metadata_is_rejected(patch: Patch) -> None:
    with pytest.raises(InvalidPatchError, match="may only patch guardrail_info and metadata"):
        Abort(status=400, message="blocked", patch=patch)
    with pytest.raises(InvalidPatchError, match="may only patch guardrail_info and metadata"):
        Respond(response={"choices": []}, patch=patch)
