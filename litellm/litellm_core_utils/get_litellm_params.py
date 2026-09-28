import reprlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import CONTROL_OPTIONS_KEY
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.llms.openai.data_residency import infer_openai_data_residency
from litellm.types.litellm_params import MAX_CONTROL_INT_DIGITS, ControlOptions
from litellm.types.router import CustomPricingLiteLLMParams

AWS_CREDENTIAL_KWARGS_KEYS: Final = frozenset(
    {
        "aws_region_name",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_session_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_external_id",
        "aws_session_tags",
        "aws_bedrock_runtime_endpoint",
        "aws_bedrock_project_id",
    }
)

PROVIDER_AFFINITY_HEADER_KWARG_KEY: Final = "provider_affinity_header"

# Pre-define optional kwargs keys as frozenset for O(1) lookups
# These are extracted from kwargs only if present, avoiding unnecessary .get() calls
OPTIONAL_KWARGS_KEYS: Final = (
    frozenset(
        {
            "azure_ad_token",
            "tenant_id",
            "client_id",
            "client_secret",
            "azure_username",
            "azure_password",
            "azure_scope",
            "timeout",
            "client_side_timeout",
            "gcs_bucket_name",
            "bucket_name",
            "s3_endpoint_url",
            "s3_region_name",
            "s3_access_key_id",
            "s3_secret_access_key",
            "vertex_credentials",
            "vertex_project",
            "vertex_location",
            "vertex_ai_project",
            "vertex_ai_location",
            "vertex_ai_credentials",
            "gigachat_scope",
            "gigachat_auth_url",
            "gigachat_access_token",
            "tpm",
            "rpm",
            "itpm",
            "otpm",
            "use_xai_oauth",
            PROVIDER_AFFINITY_HEADER_KWARG_KEY,
        }
    )
    | AWS_CREDENTIAL_KWARGS_KEYS
    | frozenset(CustomPricingLiteLLMParams.model_fields)
)

# Backward-compatible alias for existing imports/tests.
_OPTIONAL_KWARGS_KEYS: Final = OPTIONAL_KWARGS_KEYS

_CONTROL_OPTIONS: Final = TypeAdapter(ControlOptions)
_CONTROL_OPTION_NAMES: Final = tuple(field.name for field in fields(ControlOptions))
_MAX_SHOWN_INT_BITS: Final = 64
_EXPECTED: Final = f"expected a positive integer of at most {MAX_CONTROL_INT_DIGITS} digits"


class _BoundedRepr(reprlib.Repr):
    def repr_int(self, x: int, level: int) -> str:
        if x.bit_length() > _MAX_SHOWN_INT_BITS:
            return f"<int of {x.bit_length()} bits>"
        return super().repr_int(x, level)


_BOUNDED_REPR: Final = _BoundedRepr()


@dataclass(frozen=True, slots=True)
class InvalidControlOption:
    param: str
    message: str


def parse_control_options(kwargs: Mapping[str, object]) -> ControlOptions | InvalidControlOption:
    given: Final = {  # mutable-ok: TypeAdapter.validate_python takes a dict
        name: kwargs[name] for name in _CONTROL_OPTION_NAMES if name in kwargs
    }
    try:
        return _CONTROL_OPTIONS.validate_python(given)
    except ValidationError as e:
        param: Final = str(e.errors(include_url=False)[0]["loc"][0])
        return InvalidControlOption(
            param=param, message=f"Invalid {param}={_BOUNDED_REPR.repr(given[param])}: {_EXPECTED}"
        )


def stored_control_options(litellm_params: Mapping[str, object]) -> ControlOptions:
    control: Final = litellm_params.get(CONTROL_OPTIONS_KEY)
    return control if isinstance(control, ControlOptions) else ControlOptions()


def with_control_options(litellm_params: Mapping[str, object], control: ControlOptions) -> dict[str, object]:
    if control == ControlOptions():
        return dict(litellm_params)  # mutable-ok: completion() hands litellm_params to provider code typed as dict
    return {**litellm_params, CONTROL_OPTIONS_KEY: control}  # mutable-ok: same dict contract as above


def _get_base_model_from_litellm_call_metadata(
    metadata: dict | None,
) -> str | None:
    if metadata is None:
        return None
    model_info: Final = metadata.get("model_info")
    if model_info:
        return model_info.get("base_model")
    return None


def get_litellm_params(
    api_key: str | None = None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None,
    metadata: dict | None = None,
    model_info=None,
    proxy_server_request=None,
    acompletion=None,
    aembedding=None,
    allm_passthrough_route=None,
    preset_cache_key=None,
    no_log=None,
    input_cost_per_second=None,
    input_cost_per_token=None,
    output_cost_per_token=None,
    output_cost_per_second=None,
    cost_per_query=None,
    cooldown_time=None,
    text_completion=None,
    azure_ad_token_provider=None,
    user_continue_message=None,
    base_model: str | None = None,
    litellm_trace_id: str | None = None,
    litellm_session_id: str | None = None,
    hf_model_name: str | None = None,
    custom_prompt_dict: dict | None = None,
    litellm_metadata: dict | None = None,
    disable_add_transform_inline_image_block: bool | None = None,
    drop_params: bool | str | None = None,
    prompt_id: str | None = None,
    prompt_variables: dict | None = None,
    async_call: bool | None = None,
    ssl_verify: bool | None = None,
    merge_reasoning_content_in_choices: bool | None = None,
    use_litellm_proxy: bool | None = None,
    api_version: str | None = None,
    max_retries: int | None = None,
    litellm_request_debug: bool | None = None,
    **kwargs,
) -> dict:
    _litellm_metadata_dict: Final = litellm_metadata if isinstance(litellm_metadata, dict) else None
    resolved_metadata: Final = _litellm_metadata_dict.copy() if not metadata and _litellm_metadata_dict else metadata

    # Derive litellm_session_id / litellm_trace_id from metadata when not provided (call chaining)
    _meta: Final = resolved_metadata or {}
    if litellm_session_id is None:
        litellm_session_id = _meta.get("session_id") or _meta.get("trace_id")
    if litellm_trace_id is None:
        litellm_trace_id = _meta.get("trace_id") or _meta.get("session_id")

    data_residency: Final[str | None] = infer_openai_data_residency(custom_llm_provider, api_base)

    # Build base dict with explicit parameters (always included)
    litellm_params: Final = {
        "acompletion": acompletion,
        "allm_passthrough_route": allm_passthrough_route,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "data_residency": data_residency,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "aembedding": aembedding,
        "metadata": resolved_metadata,
        "model_info": model_info,
        "proxy_server_request": proxy_server_request,
        "preset_cache_key": preset_cache_key,
        "no-log": no_log or kwargs.get("no-log"),
        "stream_response": {},  # litellm_call_id: ModelResponse Dict
        "input_cost_per_token": input_cost_per_token,
        "input_cost_per_second": input_cost_per_second,
        "output_cost_per_token": output_cost_per_token,
        "output_cost_per_second": output_cost_per_second,
        "cost_per_query": cost_per_query,
        "cooldown_time": cooldown_time,
        "text_completion": text_completion,
        "azure_ad_token_provider": azure_ad_token_provider,
        "user_continue_message": user_continue_message,
        "base_model": base_model
        or (_get_base_model_from_litellm_call_metadata(metadata=metadata) if metadata else None),
        "litellm_trace_id": litellm_trace_id,
        "litellm_session_id": litellm_session_id,
        "hf_model_name": hf_model_name,
        "custom_prompt_dict": custom_prompt_dict,
        "litellm_metadata": litellm_metadata,
        "disable_add_transform_inline_image_block": disable_add_transform_inline_image_block,
        "drop_params": normalize_drop_params(drop_params),
        "prompt_id": prompt_id,
        "prompt_variables": prompt_variables,
        "async_call": async_call,
        "ssl_verify": ssl_verify,
        "merge_reasoning_content_in_choices": merge_reasoning_content_in_choices,
        "api_version": api_version,
        "max_retries": max_retries,
        "use_litellm_proxy": use_litellm_proxy,
        "litellm_request_debug": litellm_request_debug,
    }

    # Sparse extraction: only add kwargs keys that are actually present
    if kwargs:
        for key in OPTIONAL_KWARGS_KEYS:
            if key in kwargs:
                litellm_params[key] = kwargs[key]

    return litellm_params


def add_trusted_model_credentials_to_litellm_params(
    litellm_params_dict: MutableMapping[str, object], kwargs: Mapping[str, object]
) -> None:
    """
    Carry the immutable server-side credential snapshot into litellm_params.

    get_litellm_params has a fixed signature, so callers that need the snapshot to
    survive into the logging object and the downstream file read have to re-add it. Only
    a MappingProxyType is accepted, since providers resolve trusted configuration such
    as a Bedrock file bucket from it and must not read a request-supplied mapping.
    """
    trusted_model_credentials: Final = kwargs.get("_litellm_internal_model_credentials")
    if isinstance(trusted_model_credentials, MappingProxyType):
        litellm_params_dict["_litellm_internal_model_credentials"] = trusted_model_credentials
