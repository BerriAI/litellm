from typing import Final

import pytest

from litellm.integrations.custom_logger_v2 import Field, InvalidPatchError, Patch, apply_patch
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS
from litellm.integrations.custom_logger_v2.hook_point_fields import CandidateFields, ResultFields


def test_candidate_fields_show_legacy_the_list_and_v2_callbacks_a_tuple_and_keep_only_a_narrowing() -> None:
    first: Final[dict[str, object]] = {"id": "a"}
    second: Final[dict[str, object]] = {"id": "b"}
    deployments: Final[list[object]] = [first, second]
    fields: Final = CandidateFields(deployments, "group")

    narrowed: Final = apply_patch(fields, Patch().set(Field.candidate_deployments, (second,)))

    assert fields[Field.deprecated_kwargs] is deployments
    assert fields[Field.candidate_deployments] == (first, second)
    assert fields[Field.model] == "group"
    assert narrowed[Field.deprecated_kwargs] == [second]
    assert narrowed[Field.deprecated_kwargs][0] is second
    assert set(narrowed) == {Field.deprecated_kwargs, Field.candidate_deployments, Field.model}


def test_candidate_fields_carry_whatever_a_legacy_filter_returned() -> None:
    fields: Final = apply_patch(CandidateFields([{"id": "a"}], "group"), Patch().set(Field.deprecated_kwargs, None))

    assert fields[Field.deprecated_kwargs] is None
    assert Field.candidate_deployments not in fields
    with pytest.raises(InvalidPatchError, match="stay a sequence"):
        apply_patch(fields, Patch().set(Field.candidate_deployments, []))


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param(Patch().set(Field.candidate_deployments, [{"id": "a"}]), id="an equal copy"),
        pytest.param(Patch().set(Field.model, "other"), id="another field"),
    ],
)
def test_candidate_fields_refuse_anything_but_a_narrowing(patch: Patch) -> None:
    with pytest.raises(InvalidPatchError):
        apply_patch(CandidateFields([{"id": "a"}], "group"), patch)


def test_result_fields_keep_one_live_slot_that_legacy_and_v2_callback_patches_both_replace() -> None:
    request: Final[dict[str, object]] = {"model": "m", "messages": ["hi"]}
    fields: Final = ResultFields(request, PROXY_REQUEST_KEYS, Field.response, "original")

    after_legacy: Final = apply_patch(fields, Patch().set(Field.deprecated_response_obj, "from legacy"))
    seen_by_v2: Final = fields[Field.response]
    after_v2: Final = apply_patch(after_legacy, Patch().set(Field.response, "from v2"))

    assert after_legacy is fields
    assert after_v2 is fields
    assert seen_by_v2 == "from legacy"
    assert fields.value == "from v2"
    assert fields[Field.deprecated_kwargs] is request
    assert fields[Field.model] == "m"
    assert set(fields) == {
        Field.deprecated_kwargs,
        Field.model,
        Field.messages,
        Field.response,
        Field.deprecated_response_obj,
    }


def test_result_fields_refuse_a_patch_outside_the_slot_and_leave_it_unchanged() -> None:
    fields: Final = ResultFields({"model": "m"}, PROXY_REQUEST_KEYS, Field.error, "original")

    with pytest.raises(InvalidPatchError, match="only error"):
        apply_patch(fields, Patch().set(Field.error, "changed").set(Field.model, "other"))

    assert fields.value == "original"
