"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

import hashlib
from collections.abc import Iterable, Mapping
from typing import Annotated, Final, NamedTuple

from pydantic import Field, TypeAdapter

from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model
from litellm.router_utils.common_utils import provider_for_generic_call
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import server_owned_wif_litellm_params

clientside_credential_keys: Final = ["api_key", "api_base", "base_url"]

FORWARDED_API_KEY_SCOPE_METADATA_KEY: Final = "litellm_proxy_forwarded_api_key_scope"
STR_KEYED_MAPPING: Final = TypeAdapter(Mapping[str, object])
DEPLOYMENT_LITELLM_PARAMS: Final[TypeAdapter[Mapping[str, object] | LiteLLM_Params]] = TypeAdapter(
    Annotated[Mapping[str, object] | LiteLLM_Params, Field(union_mode="left_to_right")]
)
_FORWARDED_API_KEY_HEADER: Final = "x-api-key"
_FORWARDED_HEADER_KWARGS: Final = ("headers", "extra_headers")
_METADATA_KWARGS: Final = ("litellm_metadata", "metadata")

# Set on a deployment whose api_base was client-redirected, so the Anthropic auth path refuses to
# mint a federation token there even when WIF is configured only through ANTHROPIC_* env vars (which
# cannot be cleared from litellm_params).
DISABLE_WORKLOAD_IDENTITY_PARAM: Final = "anthropic_disable_workload_identity_federation"
_WIF_CLEAR_ON_BASE_OVERRIDE: Final = tuple(sorted(server_owned_wif_litellm_params))


def _admin_config_fields_to_clear_on_base_override() -> list[str]:
    """
    Provider-specific credential / endpoint-targeting fields that must NOT
    flow through to a client-redirected upstream.

    Built dynamically from ``CredentialLiteLLMParams.model_fields`` so any
    new provider field added there (Bedrock endpoint, Watsonx region, etc.)
    is gated automatically — plus a fixed list of kwargs-only fields that
    aren't declared on the typed model.
    """
    from litellm.types.router import CredentialLiteLLMParams

    typed_fields: Final = [f for f in CredentialLiteLLMParams.model_fields if f not in clientside_credential_keys]
    kwargs_only_fields: Final = [
        # Caller-supplied via **kwargs, not declared on CredentialLiteLLMParams.
        "organization",
        "extra_body",
        "extra_headers",
        "default_headers",
        "api_type",
        "azure_ad_token",
        "azure_ad_token_provider",
        "aws_session_token",
        "aws_sts_endpoint",
        "aws_web_identity_token",
        "aws_role_name",
        # OCI provider — consumed by litellm/llms/oci/* via optional_params
        # and not declared on CredentialLiteLLMParams. Without these here,
        # an admin's OCI signing key / tenancy / fingerprint would flow
        # through to an attacker-redirected upstream.
        "oci_signer",
        "oci_user",
        "oci_fingerprint",
        "oci_tenancy",
        "oci_key",
        "oci_key_file",
        # NVIDIA Riva fields — consumed by
        # ``litellm/llms/nvidia_riva/audio_transcription/handler.py`` via
        # optional_params and not declared on CredentialLiteLLMParams.
        # Admin-pinned values must not flow through on a caller-redirected
        # ``api_base`` for the same reason as the OCI entries above.
        "nvcf_function_id",
        "use_ssl",
        # Workload-identity federation minting fields, restated here from
        # server_owned_wif_litellm_params the same way azure_ad_token above is restated
        # despite also being declared on CredentialLiteLLMParams (hence covered by
        # typed_fields too): a federation token minted for a client-redirected api_base
        # would send the workload's OIDC assertion, and then the minted bearer, to the
        # caller-chosen host, so this list must stay correct even if a field is ever
        # dropped from the typed model.
        *_WIF_CLEAR_ON_BASE_OVERRIDE,
    ]
    return typed_fields + kwargs_only_fields


_ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE: Final = _admin_config_fields_to_clear_on_base_override()


def is_clientside_credential(request_kwargs: dict) -> bool:
    """
    Check if the credential is a clientside credential.
    """
    return any(key in request_kwargs for key in clientside_credential_keys)


def get_dynamic_litellm_params(litellm_params: dict, request_kwargs: dict) -> dict:
    """
    Generate a unique model_id for the deployment.

    Returns
    - litellm_params: dict

    for generating a unique model_id.
    """
    # update litellm_params with clientside credentials
    for key in clientside_credential_keys:
        if key in request_kwargs:
            litellm_params[key] = request_kwargs[key]

    # If the caller redirected api_base/base_url to a client-controlled value,
    # don't forward the admin's organization / extra_body / region / token /
    # vertex / aws fields — those were meant for the original upstream.
    # Always drop the admin's value first, then write the caller's value back
    # if they resupplied the field. The naive
    # ``if field not in request_kwargs: pop`` shape lets a caller *echo* a
    # field name (with any value, including an empty string) to keep the
    # admin's value in ``litellm_params`` and have it forwarded to the
    # redirected upstream.
    if "api_base" in request_kwargs or "base_url" in request_kwargs:
        for field in _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE:
            litellm_params.pop(field, None)
            if field in request_kwargs:
                litellm_params[field] = request_kwargs[field]
        litellm_params[DISABLE_WORKLOAD_IDENTITY_PARAM] = True

    return litellm_params


class ForwardedApiKeyScope(NamedTuple):
    """The (provider, api_base) audiences a proxy-forwarded client api_key was sent for, and that key's sha256."""

    audiences: tuple[tuple[str, str], ...]
    key_sha256: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def deployment_audience(litellm_params: Mapping[str, object] | LiteLLM_Params) -> tuple[str, str] | None:
    params: Final = (
        litellm_params
        if isinstance(litellm_params, Mapping)
        else {
            "model": litellm_params.model,
            "custom_llm_provider": litellm_params.custom_llm_provider,
            "api_base": litellm_params.api_base,
        }
    )
    provider: Final = provider_for_generic_call(params)
    if provider is None:
        return None
    api_base: Final = params.get("api_base")
    return (provider, api_base.rstrip("/") if isinstance(api_base, str) else "")


def _is_strategy_router(litellm_params: Mapping[str, object]) -> bool:
    model: Final = litellm_params.get("model")
    return isinstance(model, str) and classify_strategy_router_model(model) is not None


def forwarded_api_key_scope(
    api_key: str, deployment_params: Iterable[Mapping[str, object]]
) -> ForwardedApiKeyScope | None:
    """None when the audiences cannot be known up front: no deployments, a strategy router, or an unresolvable one."""
    params: Final = tuple(deployment_params)
    if not params or any(_is_strategy_router(litellm_params) for litellm_params in params):
        return None
    audiences: Final = tuple(deployment_audience(litellm_params) for litellm_params in params)
    if None in audiences:
        return None
    return ForwardedApiKeyScope(
        audiences=tuple(sorted({audience for audience in audiences if audience is not None})),
        key_sha256=_sha256(api_key),
    )


def _stamped_scope(metadata: object) -> ForwardedApiKeyScope | None:
    if not isinstance(metadata, Mapping):
        return None
    stamp: Final = STR_KEYED_MAPPING.validate_python(metadata).get(FORWARDED_API_KEY_SCOPE_METADATA_KEY)
    return stamp if isinstance(stamp, ForwardedApiKeyScope) else None


def stamped_forwarded_api_key_scope(request_kwargs: Mapping[str, object]) -> ForwardedApiKeyScope | None:
    return next(
        (stamp for key in _METADATA_KWARGS if (stamp := _stamped_scope(request_kwargs.get(key))) is not None),
        None,
    )


def is_forwarded_api_key(value: object, scope: ForwardedApiKeyScope) -> bool:
    return isinstance(value, str) and _sha256(value) == scope.key_sha256


def _without_forwarded_x_api_key(headers: Mapping[str, object], scope: ForwardedApiKeyScope) -> Mapping[str, object]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() != _FORWARDED_API_KEY_HEADER or not is_forwarded_api_key(value, scope)
    }


def headers_without_forwarded_api_key(
    request_kwargs: Mapping[str, object], scope: ForwardedApiKeyScope
) -> Mapping[str, Mapping[str, object]]:
    """The request's header kwargs without the forwarded key's x-api-key copy, keyed by the kwarg that carried them."""
    return {
        key: _without_forwarded_x_api_key(STR_KEYED_MAPPING.validate_python(headers), scope)
        for key in _FORWARDED_HEADER_KWARGS
        if isinstance(headers := request_kwargs.get(key), Mapping)
    }
