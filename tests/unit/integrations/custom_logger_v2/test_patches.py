from types import MappingProxyType

import pytest

from litellm.integrations.custom_logger_v2 import Field, InvalidPatchError, Patch, apply_patch, apply_patches

F = Field
BASE = MappingProxyType({F.model: "gpt", F.metadata: MappingProxyType({"keep": 1, "replace": 1})})


def test_set_replaces_one_field_and_leaves_the_base_untouched() -> None:
    patched = apply_patch(BASE, Patch().set(F.model, "claude"))

    assert patched == {F.model: "claude", F.metadata: {"keep": 1, "replace": 1}}
    assert BASE[F.model] == "gpt"


def test_merge_keeps_keys_it_does_not_name() -> None:
    patched = apply_patch(BASE, Patch().merge(F.metadata, {"replace": 2, "new": 3}))

    assert patched[F.metadata] == {"keep": 1, "replace": 2, "new": 3}


def test_merge_into_an_absent_field_starts_from_its_keys() -> None:
    patched = apply_patch(BASE, Patch().merge(F.guardrail_info, {"blocked": True}))

    assert patched[F.guardrail_info] == {"blocked": True}
    assert len(patched) == 3


def test_patches_apply_in_order_so_the_later_set_wins() -> None:
    patched = apply_patches(BASE, (Patch().set(F.model, "first"), Patch().set(F.model, "second")))

    assert patched[F.model] == "second"


def test_merge_into_a_non_mapping_value_is_rejected() -> None:
    with pytest.raises(InvalidPatchError):
        apply_patch({F.metadata: "text"}, Patch().merge(F.metadata, {"a": 1}))
