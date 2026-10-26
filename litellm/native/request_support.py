from collections.abc import Callable, Mapping
from typing import Final, Protocol

from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.get_litellm_params import InvalidControlOption, parse_control_options
from litellm.responses.bridge_check import keep_registered_mode
from litellm.types.litellm_params import ControlOptions
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import CustomPricingLiteLLMParams


def resolve_control_options(kwargs: Mapping[str, object], model: str) -> ControlOptions:
    control: Final = parse_control_options(kwargs)
    match control:
        case ControlOptions():
            return control
        case InvalidControlOption(param=param, message=message):
            if litellm.drop_params is True or normalize_drop_params(kwargs.get("drop_params")) is True:
                return ControlOptions()
            raise litellm.BadRequestError(message=message, model=model, llm_provider=None, body={"param": param})
        case _:
            return assert_never(control)


class _AdmissionReservation(LiteLLMBaseModel):
    input_tokens: int | None = None


class _AdmissionMetadata(LiteLLMBaseModel):
    user_api_key_budget_reservation: _AdmissionReservation | None = None


def admission_input_tokens(kwargs: Mapping[str, object]) -> int | None:
    reservations: Final = (
        _AdmissionMetadata.model_validate(kwargs.get(key) or {}).user_api_key_budget_reservation
        for key in ("litellm_metadata", "metadata")
    )
    return next(
        (
            reservation.input_tokens
            for reservation in reservations
            if reservation and reservation.input_tokens is not None
        ),
        None,
    )


def admitted_prompt_token_counter(prompt_tokens: int | None) -> Callable[[], int] | None:
    if prompt_tokens is None:
        return None
    return lambda: prompt_tokens


class _ProxyAuthHeadersProvider(Protocol):
    def get_auth_headers(self) -> Mapping[str, str]: ...


def proxy_auth_headers(proxy_auth: _ProxyAuthHeadersProvider) -> Mapping[str, str]:
    return proxy_auth.get_auth_headers()


def _build_custom_pricing_entry(
    custom_llm_provider: str,
    kwargs: dict,
    model_info: dict | None = None,
) -> dict:
    """Build a complete model cost entry from kwargs and model_info.

    Collects all CustomPricingLiteLLMParams fields present in kwargs and
    merges metadata from model_info (mode, supports_prompt_caching, max_tokens)
    so that register_model() receives the full pricing configuration.
    """
    entry: Final[dict] = {"litellm_provider": custom_llm_provider}

    for field_name in CustomPricingLiteLLMParams.model_fields:
        value = kwargs.get(field_name)
        if value is not None:
            entry[field_name] = value

    if model_info and isinstance(model_info, dict):
        for key in ("mode", "supports_prompt_caching", "max_tokens"):
            if key in model_info and model_info[key] is not None:
                entry.setdefault(key, model_info[key])

    return entry


def _get_router_deployment_id(kwargs: dict) -> str | None:
    for metadata_key in ("litellm_metadata", "metadata"):
        metadata = kwargs.get(metadata_key) or {}
        if not isinstance(metadata, dict):
            continue
        deployment_model_info = metadata.get("model_info") or {}
        if not isinstance(deployment_model_info, dict):
            continue
        deployment_id = deployment_model_info.get("id")
        if deployment_id is not None:
            return str(deployment_id)
    return None


def register_custom_pricing_for_request(
    model: str,
    custom_llm_provider: str,
    kwargs: dict,
    model_info: dict | None,
) -> None:
    """Register per-request custom pricing in litellm.model_cost.

    Router-originated requests (identified by the deployment id the router puts
    in metadata) get their full pricing registered under that unique id only;
    the shared ``{provider}/{model}`` key receives the entry with pricing fields
    stripped, mirroring Router._create_deployment. This keeps one deployment's
    pricing overrides (e.g. a zero-cost wildcard) from clobbering built-in
    pricing used by sibling deployments of the same backend model. Direct SDK
    calls keep the legacy behavior of registering the shared key with pricing.
    Either way the shared key's ``mode`` follows ``keep_registered_mode``, as on
    deployment registration, since a call routes by its own ``model_info``.
    """
    entry: Final = _build_custom_pricing_entry(
        custom_llm_provider=custom_llm_provider,
        kwargs=kwargs,
        model_info=model_info,
    )
    shared_key: Final = f"{custom_llm_provider}/{model}"
    deployment_id: Final = _get_router_deployment_id(kwargs)
    if deployment_id is None:
        litellm.register_model({shared_key: keep_registered_mode(shared_key, entry)}, persist_across_reloads=False)
        return
    shared_entry: Final = keep_registered_mode(
        shared_key, CustomPricingLiteLLMParams.strip_custom_pricing_fields(entry)
    )
    litellm.register_model(
        {deployment_id: entry},
        persist_across_reloads=False,
        warning_display_name=shared_key,
        custom_llm_provider=custom_llm_provider,
        add_to_provider_model_sets=False,
    )
    litellm.register_model(
        {shared_key: shared_entry},
        persist_across_reloads=False,
        warning_display_name=shared_key,
        custom_llm_provider=custom_llm_provider,
        add_to_provider_model_sets=False,
    )
