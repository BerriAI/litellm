"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

import re
from collections.abc import Mapping
from typing import Final

clientside_credential_keys: Final = ["api_key", "api_base", "base_url"]

# Metadata key the proxy stamps with the admin opt-in scope that authorized
# the caller's clientside credentials at auth time. Consumed by
# strip_clientside_credentials_without_deployment_opt_in below to re-validate
# per-deployment opt-in when a re-dispatch (server-side router fallback) would
# otherwise forward those credentials to a deployment that never opted in.
# User-supplied values for this key are stripped by the proxy (see
# litellm/proxy/litellm_pre_call_utils.py _UNTRUSTED_METADATA_CONTROL_FIELDS),
# so the scope cannot be forged by a caller.
PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_METADATA_KEY: Final = "litellm_proxy_clientside_credential_scope"
PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PROXY_WIDE: Final = "proxy_wide"
PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PER_MODEL: Final = "per_model"


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

    return litellm_params


def _clientside_param_allowed_for_deployment(
    param: str,
    request_body_value: object,
    configurable_clientside_auth_params: object,
) -> bool:
    """
    Mirror of litellm.proxy.auth.auth_utils._is_param_allowed (kept here
    because router_utils cannot import from the proxy package without a
    circular import). A param is allowed when the deployment's
    ``configurable_clientside_auth_params`` names it, or — for ``api_base``
    only — a ``{"api_base": <pattern>}`` dict entry regex/equal-matches the
    caller-supplied value.
    """
    if configurable_clientside_auth_params is None:
        return False

    for item in configurable_clientside_auth_params:
        if isinstance(item, str) and param == item:
            return True
        if isinstance(item, dict) and param == "api_base" and isinstance(request_body_value, str):
            pattern = item.get("api_base")
            if isinstance(pattern, str) and (re.match(pattern, request_body_value) or pattern == request_body_value):
                return True

    return False


def strip_clientside_credentials_without_deployment_opt_in(deployment: Mapping[str, object], kwargs: dict) -> None:
    """
    Re-validate the proxy's per-deployment clientside-credential opt-in at
    (re-)dispatch time.

    Auth time (litellm.proxy.auth.auth_utils._check_banned_params) validates
    caller-supplied clientside credentials (api_key / api_base / base_url)
    against the model the caller DECLARED only. Server-side router fallbacks
    re-dispatch the same kwargs to a DIFFERENT deployment, which must re-consent:
    without this check, a deployment that never opted in gets its api_base
    overridden to the caller's URL while keeping its own api_key, exfiltrating
    that deployment's provider key to the caller-chosen host.

    Scoping (deliberately narrow so the SDK router's per-call credential
    feature is unchanged):
    - No scope stamp in kwargs metadata (plain SDK completion call, or a proxy
      route that never stamped one): unchanged behavior.
    - ``proxy_wide`` (general_settings.allow_client_side_credentials = true):
      the admin opted every deployment in; unchanged behavior.
    - ``per_model``: each clientside credential key in kwargs must be opted in
      by the deployment being dispatched (its model_info /
      litellm_params ``configurable_clientside_auth_params``); keys that are
      not are stripped so the dispatch uses the deployment's own config.

    Metadata buckets: the proxy stamps the scope into exactly one of
    ``metadata`` / ``litellm_metadata`` (``_get_metadata_variable_name``).
    On LITELLM_METADATA_ROUTES (/v1/messages, responses, batches, bedrock,
    files) and the thread/assistant routes that bucket is
    ``litellm_metadata``, while a caller-supplied provider-facing ``metadata``
    object survives as a SECOND kwargs bucket. This helper therefore inspects
    every bucket — reading only one let an unstamped caller ``metadata``
    object shadow the stamp and skip stripping on fallback re-dispatch.
    Non-dict buckets (e.g. a JSON-encoded string) cannot carry the stamp and
    are skipped. Caller-forged scope values are stripped upstream
    (_UNTRUSTED_METADATA_CONTROL_FIELDS), so any value found is
    proxy-authored; if buckets ever disagree, fail closed to the most
    restrictive scope (``per_model``), mirroring the both-bucket loops the
    proxy uses at its other security boundaries.
    """
    scope: str | None = None
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata_bucket: object = kwargs.get(metadata_key)
        if not isinstance(metadata_bucket, dict):
            continue
        bucket_scope: object = metadata_bucket.get(PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_METADATA_KEY)
        if bucket_scope == PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PER_MODEL:
            scope = PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PER_MODEL
            break
        if bucket_scope == PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PROXY_WIDE and scope is None:
            scope = PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PROXY_WIDE
    if scope != PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_PER_MODEL:
        return

    allowed_params: object = None
    model_info: object = deployment.get("model_info")
    if isinstance(model_info, Mapping):
        allowed_params = model_info.get("configurable_clientside_auth_params")
    if allowed_params is None:
        litellm_params: object = deployment.get("litellm_params")
        if isinstance(litellm_params, Mapping):
            allowed_params = litellm_params.get("configurable_clientside_auth_params")
        else:
            allowed_params = getattr(litellm_params, "configurable_clientside_auth_params", None)

    for key in clientside_credential_keys:
        if key in kwargs and not _clientside_param_allowed_for_deployment(key, kwargs[key], allowed_params):
            kwargs.pop(key, None)
