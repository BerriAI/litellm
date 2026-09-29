"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url
- a forwarded Anthropic OAuth bearer, scoped to the anthropic provider in provider_specific_header

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

import hashlib
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_TOKEN_PREFIX
from litellm.types.utils import LlmProviders

clientside_credential_keys: Final = ["api_key", "api_base", "base_url"]
FORWARDED_OAUTH_CREDENTIAL_KEY: Final = "forwarded_oauth_credential_sha256"


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
    ]
    return typed_fields + kwargs_only_fields


_ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE: Final = _admin_config_fields_to_clear_on_base_override()


class _ScopedHeaders(TypedDict):
    custom_llm_provider: str
    extra_headers: dict[str, str]


_PROVIDER_SPECIFIC_HEADER_ADAPTER: Final[TypeAdapter[_ScopedHeaders | tuple[_ScopedHeaders, ...]]] = TypeAdapter(
    _ScopedHeaders | tuple[_ScopedHeaders, ...]
)


def forwarded_oauth_credential_fingerprint(
    request_kwargs: Mapping[str, object], custom_llm_provider: str | None
) -> str | None:
    """
    SHA-256 of the Anthropic OAuth bearer a caller forwarded for this deployment's provider, if any.

    The proxy forwards a Claude subscription token through provider_specific_header rather than api_key,
    so it would otherwise share the static deployment's cooldown identity with every other caller.
    """
    if custom_llm_provider != LlmProviders.ANTHROPIC.value:
        return None
    try:
        parsed: Final = _PROVIDER_SPECIFIC_HEADER_ADAPTER.validate_python(
            request_kwargs.get("provider_specific_header")
        )
    except ValidationError:
        return None
    scoped: Final = parsed if isinstance(parsed, tuple) else (parsed,)
    bearers: Final = tuple(
        value.removeprefix("Bearer ")
        for entry in scoped
        if custom_llm_provider in (p.strip() for p in entry["custom_llm_provider"].split(","))
        for name, value in entry["extra_headers"].items()
        if name.lower() == "authorization"
    )
    if len(bearers) != 1 or not bearers[0].startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
        return None
    return hashlib.sha256(bearers[0].encode()).hexdigest()


def is_clientside_credential(request_kwargs: Mapping[str, object], custom_llm_provider: str | None = None) -> bool:
    """
    Check if the credential is a clientside credential.
    """
    return any(key in request_kwargs for key in clientside_credential_keys) or (
        forwarded_oauth_credential_fingerprint(request_kwargs, custom_llm_provider) is not None
    )


def get_dynamic_litellm_params(
    litellm_params: dict[str, object], request_kwargs: Mapping[str, object], custom_llm_provider: str | None = None
) -> dict[str, object]:
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

    oauth_fingerprint: Final = forwarded_oauth_credential_fingerprint(request_kwargs, custom_llm_provider)
    if oauth_fingerprint is not None:
        litellm_params[FORWARDED_OAUTH_CREDENTIAL_KEY] = oauth_fingerprint

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

    return litellm_params
