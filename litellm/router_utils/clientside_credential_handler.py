"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

from collections.abc import MutableMapping
from typing import Final, cast  # noqa: TID251  # narrows the untyped request dict for mutation

from litellm.types.utils import server_owned_wif_litellm_params

clientside_credential_keys: Final = ["api_key", "api_base", "base_url"]

# Set on a deployment whose api_base was client-redirected, so the Anthropic auth path refuses to
# mint a federation token there even when WIF is configured only through ANTHROPIC_* env vars (which
# cannot be cleared from litellm_params).
DISABLE_WORKLOAD_IDENTITY_PARAM: Final = "anthropic_disable_workload_identity_federation"
_SERVER_OWNED_IDENTITY_CLEAR_ON_BASE_OVERRIDE: Final = tuple(sorted(server_owned_wif_litellm_params))


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
        # Server-owned federation and OAuth token-exchange fields, restated here from
        # server_owned_wif_litellm_params the same way azure_ad_token above is restated
        # despite also being declared on CredentialLiteLLMParams (hence covered by
        # typed_fields too): tokens minted for a client-redirected api_base could send an
        # assertion and bearer to the caller-chosen host, so this list must stay correct even
        # if a field is ever dropped from the typed model.
        *_SERVER_OWNED_IDENTITY_CLEAR_ON_BASE_OVERRIDE,
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
        from litellm.llms.github_copilot.per_user_auth import github_copilot_auth_mode

        # A per-user OAuth credential must survive the clear: dropping
        # litellm_credential_name / github_copilot_auth_type would flip the call to
        # shared mode and send the admin's Copilot token to the redirected host.
        # Per-user mode ignores the caller's api_base anyway (the session's
        # validated host wins), so keeping the mode is fail-closed.
        credential_name_value: Final[object] = litellm_params.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # litellm_params is the untyped request dict
            "litellm_credential_name"
        )
        auth_type: Final[object] = litellm_params.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # litellm_params is the untyped request dict
            "github_copilot_auth_type"
        )
        per_user_credential_locked: Final = github_copilot_auth_mode(
            credential_name_value,  # pyright: ignore[reportUnknownArgumentType]  # value comes from the untyped request dict
            auth_type,  # pyright: ignore[reportUnknownArgumentType]  # value comes from the untyped request dict
        )
        for field in _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE:
            if per_user_credential_locked and field in ("litellm_credential_name", "github_copilot_auth_type"):
                continue
            cast(  # cast-ok: litellm_params is a mutable request dict at runtime
                "MutableMapping[str, object]", litellm_params
            ).pop(field, None)
            if field in request_kwargs:
                litellm_params[field] = request_kwargs[field]
        litellm_params[DISABLE_WORKLOAD_IDENTITY_PARAM] = True

    return litellm_params
