import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, TypeAlias

import pytest
from pydantic import BaseModel

from litellm.caching.caching import Cache
from litellm.litellm_core_utils.get_litellm_params import (
    get_litellm_params,  # pyright: ignore[reportUnknownVariableType]  # untyped legacy carrier
)
from litellm.types import litellm_params
from litellm.types import utils as types_utils
from litellm.types.litellm_params import (
    ADDRESSED_RESPONSE_ID_FIELD,
    LITELLM_OWNED_ROOTS,
    TRUSTED_CALLBACK_VARS_FIELD,
    CachingOptions,
    ConnectionSettings,
    InternalState,
    LiteLLMOptions,
    owned_wire_names,
    wire,
    wire_names,
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

CONNECTION_NAMES: Final = (
    "api_key",
    "api_base",
    "api_version",
    "region_name",
    "headers",
    "provider_specific_header",
    "client",
    "shared_session",
    "ssl_verify",
    "request_timeout",
    "force_timeout",
    "stream_timeout",
    "max_retries",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_username",
    "azure_password",
    "azure_scope",
    "azure_ad_token_provider",
    "litellm_credential_name",
    "configurable_clientside_auth_params",
    "use_xai_oauth",
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

OPTION_NAMES: Final = (
    "custom_llm_provider",
    "azure",
    "use_litellm_proxy",
    "use_chat_completions_api",
    "use_in_pass_through",
    "allowed_openai_params",
    "fallbacks",
    "context_window_fallback_dict",
    "num_retries",
    "retry_policy",
    "retry_strategy",
    "routing_strategy",
    "cooldown_time",
    "allowed_model_region",
    "enable_tag_filtering",
    "fastest_response",
    "provider_affinity_header",
    "search_tool_name",
    "model_list",
    "model_alias_map",
    "model_info",
    "rpm",
    "tpm",
    "itpm",
    "otpm",
    "default_api_key_rpm_limit",
    "default_api_key_tpm_limit",
    "max_parallel_requests",
    "weight",
    "order",
    "auto_router_config_path",
    "auto_router_config",
    "auto_router_default_model",
    "auto_router_embedding_model",
    "auto_router_max_input_chars",
    "auto_router_routing_compression",
    "auto_router_model_compression",
    "complexity_router_config",
    "complexity_router_default_model",
    "adaptive_router_config",
    "adaptive_router_default_model",
    "quality_router_config",
    "quality_router_default_model",
    "caching",
    "cache",
    "ttl",
    "enable_prompt_caching",
    "caching_groups",
    "cost_per_query",
    "base_model",
    "max_budget",
    "budget_duration",
    "id",
    "metadata",
    "litellm_metadata",
    "tags",
    "litellm_trace_id",
    "litellm_session_id",
    "litellm_request_debug",
    "logger_fn",
    "verbose",
    "no-log",
    "max_agentic_loops",
    "guardrails",
    "prompt_id",
    "prompt_variables",
    "prompt_version",
    "prompt_environment",
    "prompt_label",
    "litellm_system_prompt",
    "custom_prompt_dict",
    "roles",
    "final_prompt_value",
    "bos_token",
    "eos_token",
    "hf_model_name",
    "supports_system_message",
    "ensure_alternating_roles",
    "user_continue_message",
    "assistant_continue_message",
    "disable_add_transform_inline_image_block",
    "merge_reasoning_content_in_choices",
    "enable_json_schema_validation",
    "complete_response",
    "stream_chunk_size",
    "keepalive_seconds",
    "allow_client_keepalive_override",
    "mock_response",
    "mock_timeout",
)

AGENTIC_LOOP_STATE_NAMES: Final = (
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_agentic_loop_api_surface",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_session_scoped",
    "_code_interpreter_interception_converted_stream",
    "_websearch_interception_emit_native_blocks",
    "_websearch_interception_converted_stream",
    "_headroom_interception_converted_stream",
)

INTERNAL_STATE_NAMES: Final = (
    "litellm_call_id",
    "completion_call_id",
    "data_residency",
    "litellm_logging_obj",
    "preset_cache_key",
    "cache_key",
    "stream_response",
    "_context_compaction_state",
    *AGENTIC_LOOP_STATE_NAMES,
    "_router_weights",
    "fallback_depth",
    "max_fallbacks",
    "attempted_targets",
    "proxy_server_request",
    "secret_fields",
    "litellm_trusted_callback_vars",
    "_litellm_addressed_response_id",
    "_litellm_strip_stream_usage",
    "client_side_timeout",
    "model_file_id_mapping",
    "acompletion",
    "aembedding",
    "aimg_generation",
    "atext_completion",
    "text_completion",
    "allm_passthrough_route",
    "async_call",
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

ARTIFACT_NAMES: Final = ("self", "use_client", "model_config", "rust")

CALLBACK_VAR_NAMES: Final = (
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_environment",
    "langfuse_span_scope",
    "langfuse_prompt_version",
    "gcs_bucket_name",
    "gcs_path_service_account",
    "langsmith_api_key",
    "langsmith_project",
    "langsmith_base_url",
    "langsmith_sampling_rate",
    "langsmith_tenant_id",
    "humanloop_api_key",
    "arize_api_key",
    "arize_space_key",
    "arize_space_id",
    "arize_success_sampling_rate",
    "arize_error_sampling_rate",
    "posthog_api_key",
    "posthog_api_url",
    "wandb_api_key",
    "weave_project_id",
    "dd_api_key",
    "dd_site",
    "dd_agent_host",
    "dd_agent_port",
    "newrelic_api_key",
    "newrelic_region",
    "turn_off_message_logging",
    "litellm_disabled_callbacks",
)

PRICING_NAMES: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_character",
    "output_cost_per_character",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "tiered_pricing",
    "input_cost_per_second",
    "output_cost_per_second",
    "output_cost_per_second_1080p",
    "output_cost_per_second_480p",
    "output_cost_per_second_720p",
    "output_cost_per_second_768p",
    "output_cost_per_second_2k",
    "output_cost_per_second_4k",
    "output_cost_per_image_512",
    "output_cost_per_image_1024",
    "output_cost_per_image_1536",
    "input_cost_per_pixel",
    "output_cost_per_pixel",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_token_ultrafast",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens_priority",
    "cache_creation_input_token_cost_above_272k_tokens_flex",
    "cache_creation_input_token_cost_flex",
    "cache_creation_input_token_cost_priority",
    "cache_creation_input_token_cost_ultrafast",
    "cache_creation_input_audio_token_cost",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "cache_read_input_token_cost_ultrafast",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens_flex",
    "cache_read_input_token_cost_batches",
    "cache_read_input_token_cost_above_272k_tokens_batches",
    "cache_creation_input_token_cost_batches",
    "cache_creation_input_token_cost_above_272k_tokens_batches",
    "cache_read_input_audio_token_cost",
    "cache_read_input_image_token_cost",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_batches",
    "input_cost_per_query",
    "input_cost_per_image",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_audio_per_second",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_video_per_second",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_per_second_above_15s_interval",
    "input_cost_per_video_per_second_above_8s_interval",
    "input_cost_per_audio_token_batches",
    "input_cost_per_image_token_batches",
    "input_cost_per_token_batches",
    "input_cost_per_video_token_batches",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_token_ultrafast",
    "output_cost_per_audio_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_batches",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_token",
    "output_cost_per_video_token",
    "output_cost_per_reasoning_token",
    "output_cost_per_reasoning_token_flex",
    "output_cost_per_reasoning_token_priority",
    "output_cost_per_video_per_second",
    "output_cost_per_audio_per_second",
    "search_context_cost_per_query",
    "google_maps_grounding_cost_per_query",
    "citation_cost_per_token",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost_above_512k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_video_token",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_512k_tokens",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_512k_tokens",
    "output_vector_size",
    "ocr_cost_per_page",
    "ocr_cost_per_page_batches",
    "ocr_cost_per_credit",
    "annotation_cost_per_page",
    "annotation_cost_per_page_batches",
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "regional_endpoint_uplift_multiplier",
)

DECLARED_BY_ROOT: Final[Mapping[type, tuple[str, ...]]] = MappingProxyType(
    {ConnectionSettings: CONNECTION_NAMES, LiteLLMOptions: OPTION_NAMES, InternalState: INTERNAL_STATE_NAMES}
)

OWNED_NAMES: Final = (
    *CONNECTION_NAMES,
    *OPTION_NAMES,
    *INTERNAL_STATE_NAMES,
    *ARTIFACT_NAMES,
    *CALLBACK_VAR_NAMES,
    *PRICING_NAMES,
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


def test_a_name_no_object_declares_reaches_the_provider() -> None:
    result: Final = CLASSIFIERS["completion"]({PROVIDER_KNOB: 1})  # mutable-ok: classifier input type

    assert result == MappingProxyType({PROVIDER_KNOB: 1})


def _cache_key_for_model_group(model_group: str, options: CachingOptions) -> str:
    return Cache().get_cache_key(  # pyright: ignore[reportUnknownMemberType]  # untyped legacy key builder
        model=model_group,
        messages=(MappingProxyType({"role": "user", "content": "shared prompt"}),),
        metadata=MappingProxyType({"caching_groups": options.caching_groups, "model_group": model_group}),
    )


def test_caching_groups_is_a_flat_sequence_of_model_groups_that_share_one_cache_key() -> None:
    options: Final = CachingOptions(caching_groups=(("gpt-4", "gpt-4o"), ("claude-3",)))

    keys: Final = tuple(_cache_key_for_model_group(group, options) for group in ("gpt-4", "gpt-4o", "claude-3"))

    assert (keys[0] == keys[1], keys[0] == keys[2]) == (True, False)


def test_all_litellm_params_is_exactly_the_owned_inventory() -> None:
    assert frozenset(all_litellm_params) == frozenset(OWNED_NAMES)


def test_callback_vars_are_the_fields_of_the_callback_typed_dict() -> None:
    assert tuple(StandardCallbackDynamicParams.__annotations__) == CALLBACK_VAR_NAMES


def test_pricing_names_are_the_fields_of_the_pricing_model() -> None:
    assert tuple(CustomPricingLiteLLMParams.model_fields) == PRICING_NAMES


def test_every_owned_name_has_exactly_one_owner() -> None:
    duplicated: Final = tuple(name for name in dict.fromkeys(all_litellm_params) if all_litellm_params.count(name) > 1)

    assert duplicated == ()


@pytest.mark.parametrize("root", LITELLM_OWNED_ROOTS, ids=(root.__name__ for root in LITELLM_OWNED_ROOTS))
def test_root_declares_exactly_the_names_that_live_on_its_object(root: type) -> None:
    assert frozenset(owned_wire_names(root)) == frozenset(DECLARED_BY_ROOT[root])


@pytest.mark.parametrize(
    ("exported", "declared"),
    (
        pytest.param(
            types_utils.TRUSTED_CALLBACK_VARS_FIELD,
            litellm_params.TRUSTED_CALLBACK_VARS_FIELD,
            id="TRUSTED_CALLBACK_VARS_FIELD",
        ),
        pytest.param(
            types_utils.ADDRESSED_RESPONSE_ID_FIELD,
            litellm_params.ADDRESSED_RESPONSE_ID_FIELD,
            id="ADDRESSED_RESPONSE_ID_FIELD",
        ),
    ),
)
def test_types_utils_still_exports_the_field_constant(exported: str, declared: str) -> None:
    assert exported == declared


@dataclass(frozen=True, slots=True, kw_only=True)
class _Leaf:
    plain: int | None = None
    renamed: int | None = field(default=None, metadata=wire("wire-name"))


@dataclass(frozen=True, slots=True, kw_only=True)
class _OtherLeaf:
    plain: int | None = None
    trailing: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Root:
    first: _Leaf
    second: _OtherLeaf


@dataclass(frozen=True, slots=True, kw_only=True)
class _RootDeclaringAKwargDirectly:
    first: _Leaf
    stray: int | None = None


def test_wire_names_are_the_field_names_in_declaration_order_unless_wire_renames_them() -> None:
    assert wire_names(_Leaf) == ("plain", "wire-name")


def test_owned_wire_names_walk_leaves_in_declaration_order_and_keep_every_occurrence() -> None:
    assert owned_wire_names(_Root) == ("plain", "wire-name", "plain", "trailing")


def test_owned_wire_names_refuse_a_root_that_declares_a_kwarg_outside_a_leaf() -> None:
    with pytest.raises(TypeError):
        owned_wire_names(_RootDeclaringAKwargDirectly)


def test_agentic_loop_names_concatenate_as_a_list() -> None:
    extended: Final = agentic_loop_internal_litellm_params + ["caller_added"]  # mutable-ok: list contract under test

    assert (type(extended), len(extended), frozenset(extended)) == (
        list,
        len(AGENTIC_LOOP_STATE_NAMES) + 2,
        frozenset((*AGENTIC_LOOP_STATE_NAMES, "max_agentic_loops", "caller_added")),
    )


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

    assert (type(extended), frozenset(extended)) == (list, frozenset(("aembedding", "extra_headers", *OWNED_NAMES)))


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


TYPED_MODEL_OWNERS: Final[Mapping[str, tuple[tuple[type[BaseModel], ...], type]]] = MappingProxyType(
    {
        "credentials": ((CredentialLiteLLMParams,), ConnectionSettings),
        "router": ((RouterConfig, UpdateRouterConfig), LiteLLMOptions),
    }
)

DECLARED_NAMES: Final = frozenset(name for root in LITELLM_OWNED_ROOTS for name in owned_wire_names(root))


NAMES_SHARED_WITH_TYPED_MODELS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "credentials": (
            "api_base",
            "api_key",
            "api_version",
            "aws_batch_role_arn",
            "azure_password",
            "azure_scope",
            "azure_username",
            "bedrock_tags",
            "client_id",
            "client_secret",
            "region_name",
            "s3_access_key_id",
            "s3_bucket_name",
            "s3_bucket_owner",
            "s3_encryption_key_id",
            "s3_endpoint_url",
            "s3_output_bucket_name",
            "s3_region_name",
            "s3_secret_access_key",
            "tenant_id",
        ),
        "router": (
            "caching_groups",
            "cooldown_time",
            "enable_tag_filtering",
            "fallbacks",
            "model_list",
            "num_retries",
            "retry_policy",
            "routing_strategy",
        ),
    }
)


@pytest.mark.parametrize("source", TYPED_MODEL_OWNERS)
def test_names_a_typed_config_model_shares_with_its_root_are_exactly_the_declared_ones(source: str) -> None:
    models, root = TYPED_MODEL_OWNERS[source]
    model_names: Final = frozenset(name for model in models for name in model.model_fields)

    assert frozenset(owned_wire_names(root)) & model_names == frozenset(NAMES_SHARED_WITH_TYPED_MODELS[source])


@pytest.mark.parametrize("name", PRICING_NAMES)
def test_pricing_name_is_owned_by_the_pricing_model_alone(name: str) -> None:
    assert name not in DECLARED_NAMES
