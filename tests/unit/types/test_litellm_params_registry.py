import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

import pytest
from pydantic import BaseModel

from litellm.litellm_core_utils.get_litellm_params import (
    get_litellm_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy carrier
)
from litellm.types.litellm_params_registry import (
    ADDRESSED_RESPONSE_ID_FIELD,
    LITELLM_PARAMS,
    TRUSTED_CALLBACK_VARS_FIELD,
    LiteLLMParam,
    ParamGroup,
)
from litellm.types.router import CredentialLiteLLMParams, RouterConfig, UpdateRouterConfig
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    StandardCallbackDynamicParams,
    agentic_loop_internal_litellm_params,
    all_litellm_params,
    bedrock_batch_litellm_params,
)
from litellm.utils import (
    filter_out_litellm_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy classifier
    get_non_default_completion_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy classifier
    get_non_default_transcription_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy classifier
)

PROVIDER_KNOB: Final = "registry_test_provider_only_knob"

LOAD_BEARING_NAMES: Final = (
    "api_key",
    "api_base",
    "fallbacks",
    "context_window_fallback_dict",
    "num_retries",
    "tags",
    "guardrails",
    "caching",
    "cache",
    "metadata",
    "litellm_call_id",
    "mock_response",
    "stream_chunk_size",
    "max_agentic_loops",
    "s3_bucket_name",
    "input_cost_per_token",
    "turn_off_message_logging",
)

OWNED_NAMES: Final = tuple(
    dict.fromkeys(
        (
            *LOAD_BEARING_NAMES,
            *(param.name for param in LITELLM_PARAMS),
            *StandardCallbackDynamicParams.__annotations__,
            *CustomPricingLiteLLMParams.model_fields,
        )
    )
)

Classifier: TypeAlias = Callable[[dict[str, object]], dict[str, object]]  # mutable-ok: classifiers use dict

CLASSIFIERS: Final[Mapping[str, Classifier]] = MappingProxyType(
    {  # pyright: ignore[reportUnknownArgumentType]  # untyped legacy classifiers
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


AGENTIC_LOOP_NAMES: Final = (
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_agentic_loop_api_surface",
    "max_agentic_loops",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_session_scoped",
    "_code_interpreter_interception_converted_stream",
    "_websearch_interception_emit_native_blocks",
    "_websearch_interception_converted_stream",
    "_headroom_interception_converted_stream",
)

BEDROCK_BATCH_NAMES: Final = (
    "aws_batch_role_arn",
    "s3_bucket_name",
    "s3_region_name",
    "s3_endpoint_url",
    "s3_output_bucket_name",
    "s3_bucket_owner",
    "s3_access_key_id",
    "s3_secret_access_key",
    "s3_encryption_key_id",
    "bedrock_tags",
)


def test_agentic_loop_names_concatenate_as_a_list() -> None:
    extended: Final = agentic_loop_internal_litellm_params + ["caller_added"]  # mutable-ok: list contract under test

    assert (type(extended), tuple(extended)) == (list, (*AGENTIC_LOOP_NAMES, "caller_added"))


def test_bedrock_batch_names_concatenate_as_a_tuple() -> None:
    extended: Final = bedrock_batch_litellm_params + ("caller_added",)

    assert extended == (*BEDROCK_BATCH_NAMES, "caller_added")


def test_proxy_stamped_fields_keep_their_wire_names() -> None:
    assert (TRUSTED_CALLBACK_VARS_FIELD, ADDRESSED_RESPONSE_ID_FIELD) == (
        "litellm_trusted_callback_vars",
        "_litellm_addressed_response_id",
    )


def test_all_litellm_params_concatenates_with_a_list_like_the_completion_entrypoint_does() -> None:
    extended: Final = ["aembedding", "extra_headers"] + all_litellm_params  # mutable-ok: list contract under test

    assert (type(extended), tuple(extended)) == (list, ("aembedding", "extra_headers", *all_litellm_params))


def test_all_litellm_params_keeps_every_entry_of_every_source() -> None:
    assert len(all_litellm_params) == len(LITELLM_PARAMS) + len(StandardCallbackDynamicParams.__annotations__) + len(
        CustomPricingLiteLLMParams.model_fields
    )


CARRIED_AND_FORWARDED: Final = frozenset(("drop_params", "hugging_face", "no_log", "replicate", "together_ai"))

CARRIER_SIGNATURE: Final = inspect.signature(get_litellm_params)  # pyright: ignore[reportUnknownArgumentType]  # legacy

CARRIED_PARAMS: Final = tuple(
    name for name in CARRIER_SIGNATURE.parameters if name != "kwargs" and name not in CARRIED_AND_FORWARDED
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
