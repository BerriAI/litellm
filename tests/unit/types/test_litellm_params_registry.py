from collections.abc import Callable, Mapping
from typing import Final

import pytest

from litellm.types.litellm_params_registry import LITELLM_PARAMS, LiteLLMParam, ParamGroup, names_in
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    StandardCallbackDynamicParams,
    agentic_loop_internal_litellm_params,
    all_litellm_params,
    bedrock_batch_litellm_params,
)
from litellm.utils import (
    filter_out_litellm_params,
    get_non_default_completion_params,
    get_non_default_transcription_params,
)

PROVIDER_KNOB: Final = "registry_test_provider_only_knob"

OWNED_NAMES: Final = tuple(
    dict.fromkeys(
        (
            *(param.name for param in LITELLM_PARAMS),
            *StandardCallbackDynamicParams.__annotations__,
            *CustomPricingLiteLLMParams.model_fields,
        )
    )
)

CLASSIFIERS: Final[Mapping[str, Callable[[dict[str, object]], dict[str, object]]]] = {
    "completion": get_non_default_completion_params,
    "transcription": get_non_default_transcription_params,
    "filter_out": filter_out_litellm_params,
}


@pytest.mark.parametrize("classifier_name", CLASSIFIERS)
@pytest.mark.parametrize("name", OWNED_NAMES)
def test_owned_name_is_kept_out_of_provider_params(name: str, classifier_name: str) -> None:
    provider_value: Final = object()
    kwargs: Final[dict[str, object]] = {name: object(), PROVIDER_KNOB: provider_value}

    result: Final = CLASSIFIERS[classifier_name](kwargs)

    assert result == {PROVIDER_KNOB: provider_value}
    assert result[PROVIDER_KNOB] is provider_value


@pytest.mark.parametrize("group", ParamGroup)
def test_every_group_declares_at_least_one_name(group: ParamGroup) -> None:
    assert names_in(group), f"{group.name} has no members"


def test_groups_partition_the_registry_without_loss_or_repetition() -> None:
    assert sum(len(names_in(group)) for group in ParamGroup) == len(LITELLM_PARAMS)


def test_agentic_loop_names_concatenate_as_a_list() -> None:
    extended: Final = agentic_loop_internal_litellm_params + ["caller_added"]

    assert extended == [*names_in(ParamGroup.AGENTIC_LOOP_STATE), "caller_added"]


def test_bedrock_batch_names_concatenate_as_a_tuple() -> None:
    extended: Final = bedrock_batch_litellm_params + ("caller_added",)

    assert extended == (*names_in(ParamGroup.BEDROCK_BATCH_CONFIG), "caller_added")


def test_all_litellm_params_concatenates_with_a_list_like_the_completion_entrypoint_does() -> None:
    extended: Final = ["aembedding", "extra_headers"] + all_litellm_params

    assert extended == ["aembedding", "extra_headers", *all_litellm_params]


def test_stream_chunk_size_stays_out_of_provider_params() -> None:
    assert get_non_default_completion_params({"stream_chunk_size": 64, PROVIDER_KNOB: 1}) == {PROVIDER_KNOB: 1}


@pytest.mark.parametrize("param", LITELLM_PARAMS, ids=lambda param: f"{param.group.name}:{param.name}")
def test_every_param_is_found_in_its_own_group_and_no_other(param: LiteLLMParam) -> None:
    containing_groups: Final = frozenset(group for group in ParamGroup if param.name in names_in(group))

    assert containing_groups == {param.group}
    assert param.name.strip() == param.name != ""
