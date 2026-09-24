import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

import pytest
from pydantic import BaseModel

from litellm.litellm_core_utils.get_litellm_params import (
    get_litellm_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy carrier
)
from litellm.types.litellm_params_registry import LITELLM_PARAMS, LiteLLMParam, ParamGroup, names_in
from litellm.types.router import CredentialLiteLLMParams, RouterConfig, UpdateRouterConfig
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

Classifier: TypeAlias = Callable[[dict[str, object]], dict[str, object]]  # mutable-ok: classifiers use dict

CLASSIFIERS: Final[Mapping[str, Classifier]] = MappingProxyType(
    {
        "completion": get_non_default_completion_params,
        "transcription": get_non_default_transcription_params,
        "filter_out": filter_out_litellm_params,
    }
)


@pytest.mark.parametrize("classifier_name", CLASSIFIERS)
@pytest.mark.parametrize("name", OWNED_NAMES)
def test_owned_name_is_kept_out_of_provider_params(name: str, classifier_name: str) -> None:
    provider_value: Final = object()
    classify: Final = CLASSIFIERS[classifier_name]

    result: Final = classify({name: object(), PROVIDER_KNOB: provider_value})  # mutable-ok: classifiers take a dict

    assert result == MappingProxyType({PROVIDER_KNOB: provider_value})
    assert result[PROVIDER_KNOB] is provider_value


@pytest.mark.parametrize("group", ParamGroup)
def test_every_group_declares_at_least_one_name(group: ParamGroup) -> None:
    assert names_in(group), f"{group.name} has no members"


def test_agentic_loop_names_concatenate_as_a_list() -> None:
    extended: Final = agentic_loop_internal_litellm_params + ["caller_added"]  # mutable-ok: list contract under test

    assert (type(extended), tuple(extended)) == (list, (*names_in(ParamGroup.AGENTIC_LOOP_STATE), "caller_added"))


def test_bedrock_batch_names_concatenate_as_a_tuple() -> None:
    extended: Final = bedrock_batch_litellm_params + ("caller_added",)

    assert extended == (*names_in(ParamGroup.BEDROCK_BATCH_CONFIG), "caller_added")


def test_all_litellm_params_concatenates_with_a_list_like_the_completion_entrypoint_does() -> None:
    extended: Final = ["aembedding", "extra_headers"] + all_litellm_params  # mutable-ok: list contract under test

    assert (type(extended), tuple(extended)) == (list, ("aembedding", "extra_headers", *all_litellm_params))


def test_stream_chunk_size_stays_out_of_provider_params() -> None:
    kwargs: Final[dict[str, object]] = {"stream_chunk_size": 64, PROVIDER_KNOB: 1}  # mutable-ok: classifier input type

    assert get_non_default_completion_params(kwargs) == MappingProxyType({PROVIDER_KNOB: 1})


def _param_id(param: LiteLLMParam) -> str:
    return f"{param.group.name}:{param.name}"


@pytest.mark.parametrize("param", LITELLM_PARAMS, ids=_param_id)
def test_every_param_is_found_in_its_own_group_and_no_other(param: LiteLLMParam) -> None:
    containing_groups: Final = frozenset(group for group in ParamGroup if param.name in names_in(group))

    assert containing_groups == frozenset((param.group,))
    assert param.name.strip() == param.name != ""


CARRIED_AND_FORWARDED: Final = frozenset(("drop_params", "hugging_face", "no_log", "replicate", "together_ai"))

CARRIED_PARAMS: Final = tuple(
    name
    for name in inspect.signature(get_litellm_params).parameters  # pyright: ignore[reportUnknownArgumentType]  # untyped legacy carrier, only its parameter names are read
    if name != "kwargs" and name not in CARRIED_AND_FORWARDED
)


@pytest.mark.parametrize("name", CARRIED_PARAMS)
def test_every_param_get_litellm_params_carries_is_kept_out_of_provider_params(name: str) -> None:
    provider_value: Final = object()

    result: Final = CLASSIFIERS["completion"](
        {name: object(), PROVIDER_KNOB: provider_value}  # mutable-ok: classifier input type
    )

    assert result == MappingProxyType({PROVIDER_KNOB: provider_value})


TYPED_GROUP_SOURCES: Final[Mapping[str, tuple[tuple[type[BaseModel], ...], frozenset[ParamGroup]]]] = MappingProxyType(
    {
        "credentials": (
            (CredentialLiteLLMParams,),
            frozenset((ParamGroup.CREDENTIALS_AND_ENDPOINT, ParamGroup.BEDROCK_BATCH_CONFIG)),
        ),
        "router": ((RouterConfig, UpdateRouterConfig), frozenset((ParamGroup.ROUTING_AND_RELIABILITY,))),
        "pricing": ((CustomPricingLiteLLMParams,), frozenset((ParamGroup.COST_AND_BUDGET,))),
    }
)


def _typed_params(source: str) -> tuple[LiteLLMParam, ...]:
    models: Final = TYPED_GROUP_SOURCES[source][0]
    return tuple(param for param in LITELLM_PARAMS if any(param.name in model.model_fields for model in models))


TYPED_GROUP_CASES: Final = tuple(
    pytest.param(source, param, id=f"{source}:{param.name}")
    for source in TYPED_GROUP_SOURCES
    for param in _typed_params(source)
)


@pytest.mark.parametrize(("source", "param"), TYPED_GROUP_CASES)
def test_param_declared_by_a_typed_config_model_is_in_that_models_group(source: str, param: LiteLLMParam) -> None:
    assert param.group in TYPED_GROUP_SOURCES[source][1]


@pytest.mark.parametrize("source", TYPED_GROUP_SOURCES)
def test_every_typed_config_model_declares_registry_names(source: str) -> None:
    assert _typed_params(source)
