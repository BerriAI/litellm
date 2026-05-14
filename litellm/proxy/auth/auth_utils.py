import os
import re
import sys
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

from fastapi import HTTPException, Request, status

import litellm
from litellm import Router, provider_list
from litellm._logging import verbose_proxy_logger
from litellm.constants import STANDARD_CUSTOMER_ID_HEADERS
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.proxy._types import *
from litellm.types.router import CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS


def _get_request_ip_address(
    request: Request, use_x_forwarded_for: Optional[bool] = False
) -> Optional[str]:
    client_ip = None
    if use_x_forwarded_for is True and "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"]
    elif request.client is not None:
        client_ip = request.client.host
    else:
        client_ip = ""

    return client_ip


def _check_valid_ip(
    allowed_ips: Optional[List[str]],
    request: Request,
    use_x_forwarded_for: Optional[bool] = False,
) -> Tuple[bool, Optional[str]]:
    """
    Returns if ip is allowed or not
    """
    if allowed_ips is None:  # if not set, assume true
        return True, None

    # if general_settings.get("use_x_forwarded_for") is True then use x-forwarded-for
    client_ip = _get_request_ip_address(
        request=request, use_x_forwarded_for=use_x_forwarded_for
    )

    # Check if IP address is allowed
    if client_ip not in allowed_ips:
        return False, client_ip

    return True, client_ip


def check_complete_credentials(request_body: dict) -> bool:
    """
    if 'api_base' in request body. Check if complete credentials given. Prevent malicious attacks.

    Supplying an ``api_key`` is necessary but not sufficient: even with
    credentials supplied, an ``api_base`` / ``base_url`` that resolves to a
    private/internal/cloud-metadata address would still allow the proxy to
    be used as an SSRF pivot. Validate any URL fields here so the gate
    can't be bypassed with ``api_key=anything`` plus a malicious target.
    """
    given_model: Optional[str] = None

    given_model = request_body.get("model")
    if given_model is None:
        return False

    if (
        "sagemaker" in given_model
        or "bedrock" in given_model
        or "vertex_ai" in given_model
        or "vertex_ai_beta" in given_model
    ):
        # complex credentials - easier to make a malicious request
        return False

    api_key_value = request_body.get("api_key")
    if not (api_key_value and isinstance(api_key_value, str) and api_key_value.strip()):
        return False

    # ``validate_url`` itself doesn't consult the toggle; ``safe_get`` /
    # ``async_safe_get`` do. Mirror that here so admins who explicitly
    # disabled URL validation (e.g. for an internal Ollama endpoint they
    # accept the SSRF risk for) aren't blocked at the proxy boundary.
    if getattr(litellm, "user_url_validation", False):
        for url_field in ("api_base", "base_url"):
            url_value = request_body.get(url_field)
            if not url_value or not isinstance(url_value, str):
                continue
            try:
                validate_url(url_value)
            except SSRFError as e:
                raise ValueError(
                    f"Rejected request: client-side {url_field}={url_value!r} "
                    f"is rejected by the SSRF guard ({e})."
                )

    return True


def check_regex_or_str_match(request_body_value: Any, regex_str: str) -> bool:
    """
    Check if request_body_value matches the regex_str or is equal to param
    """
    if re.match(regex_str, request_body_value) or regex_str == request_body_value:
        return True
    return False


def _is_param_allowed(
    param: str,
    request_body_value: Any,
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS,
) -> bool:
    """
    Check if param is a str or dict and if request_body_value is in the list of allowed values
    """
    if configurable_clientside_auth_params is None:
        return False

    for item in configurable_clientside_auth_params:
        if isinstance(item, str) and param == item:
            return True
        elif isinstance(item, Dict):
            if param == "api_base" and check_regex_or_str_match(
                request_body_value=request_body_value,
                regex_str=item["api_base"],
            ):  # assume param is a regex
                return True

    return False


def _allow_model_level_clientside_configurable_parameters(
    model: str, param: str, request_body_value: Any, llm_router: Optional[Router]
) -> bool:
    """
    Check if model is allowed to use configurable client-side params
    - get matching model
    - check if 'clientside_configurable_parameters' is set for model
    -
    """
    if llm_router is None:
        return False
    # check if model is set
    model_info = llm_router.get_model_group_info(model_group=model)
    if model_info is None:
        # check if wildcard model is set
        if model.split("/", 1)[0] in provider_list:
            model_info = llm_router.get_model_group_info(
                model_group=model.split("/", 1)[0]
            )

    if model_info is None:
        return False

    if model_info is None or model_info.configurable_clientside_auth_params is None:
        return False

    return _is_param_allowed(
        param=param,
        request_body_value=request_body_value,
        configurable_clientside_auth_params=model_info.configurable_clientside_auth_params,
    )


# Config dicts whose entries are spread as ``**dict`` into outbound LLM
# API calls. ``litellm_embedding_config`` is consumed by the Milvus
# vector store transformer. ``extra_body`` is the OpenAI-SDK passthrough
# container: provider modules pull provider-auth fields out of it
# (e.g. Azure's ``extra_body.azure_ad_token``, Bedrock's
# ``extra_body.aws_web_identity_token``) without re-validating, so the
# banned-key check has to descend into it the same way it descends into
# ``litellm_embedding_config``.
_NESTED_CONFIG_KEYS: Tuple[str, ...] = ("litellm_embedding_config", "extra_body")

# Metadata containers that carry per-request configuration consumed by the
# observability callbacks. The same banned-param list applies — a value
# under ``metadata.langfuse_host`` redirects the same Langfuse client and
# leaks the same credentials as the root-level ``langfuse_host``, but the
# original check only walked the request-body root, so the metadata path
# was an unintentional bypass.
_NESTED_METADATA_KEYS: Tuple[str, ...] = ("metadata", "litellm_metadata")

# Banned request-body params. The same list applies to every entry in
# ``_NESTED_CONFIG_KEYS`` (dicts spread as ``**kwargs`` into outbound
# calls) and ``_NESTED_METADATA_KEYS`` (dicts read directly by integration
# callbacks), so a single banned name is enforced wherever the field can
# reach the call path from.
# Per-request observability params that are SAFE to accept from clients.
# These describe the request being logged (prompt version, sampling rate)
# without choosing the destination or the credentials, so they don't
# contribute to the data-exfil primitive that the rest of
# ``_supported_callback_params`` does.
_SAFE_CLIENT_CALLBACK_PARAMS: FrozenSet[str] = frozenset(
    {
        "langfuse_prompt_version",
        "langsmith_sampling_rate",
    }
)

# Observability fields that integrations read from the request body or
# metadata but that are not (yet) listed in ``_supported_callback_params``.
# Listed here so the proxy bans them today; the long-term cleanup is to
# fold these into the canonical allowlist so they share one source of
# truth with the rest.
_EXTRA_BANNED_OBSERVABILITY_PARAMS: FrozenSet[str] = frozenset(
    {
        "posthog_api_url",
        "phoenix_project_name",
        "wandb_api_key",
        "weave_project_id",
    }
)


def _build_banned_observability_params() -> FrozenSet[str]:
    """Derive the observability ban list from the canonical allowlist.

    ``_supported_callback_params`` and ``_request_blocked_callback_params`` in
    ``litellm/litellm_core_utils/initialize_dynamic_callback_params.py`` is
    the single place that enumerates every observability field integrations
    resolve from kwargs/metadata, plus fields that integration code explicitly
    blocks from request-supplied callback params. Subtract the small set of
    informational fields (``_SAFE_CLIENT_CALLBACK_PARAMS``) and union with the
    extras the canonical allowlist hasn't caught up to yet. New integrations
    added to the canonical allowlist are banned by default, which is the safe
    failure mode.
    """
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        _request_blocked_callback_params,
        _supported_callback_params,
    )

    return (
        (frozenset(_supported_callback_params) - _SAFE_CLIENT_CALLBACK_PARAMS)
        | frozenset(_request_blocked_callback_params)
        | _EXTRA_BANNED_OBSERVABILITY_PARAMS
    )


_BANNED_REQUEST_BODY_PARAMS: Tuple[str, ...] = (
    "api_base",
    "base_url",
    "user_config",
    "aws_sts_endpoint",
    "aws_web_identity_token",
    "aws_role_name",
    "vertex_credentials",
    # Azure managed-identity / federated-auth token. The Azure provider
    # transformer reads ``azure_ad_token`` (top-level or via
    # ``extra_body``) and resolves it through ``get_secret`` before
    # passing it as the bearer token to the Azure endpoint, so a
    # caller-supplied value is the same exfil shape as
    # ``aws_web_identity_token`` on the Bedrock path.
    "azure_ad_token",
    # Endpoint-targeting fields that retarget the outbound request or
    # an observability callback. An attacker-controlled value either
    # exfiltrates the request payload (incl. messages + admin-set
    # tokens) to the attacker's host, or coerces the proxy into
    # authenticating against the attacker's host with admin secrets.
    "aws_bedrock_runtime_endpoint",
    # Provider-specific endpoint overrides that flow into the outbound
    # request via ``optional_params``. Same threat as ``api_base``:
    # ``s3_endpoint_url`` redirects Bedrock file uploads to attacker
    # S3; ``sagemaker_base_url`` redirects all SageMaker traffic;
    # ``deployment_url`` redirects SAP deployments.
    "s3_endpoint_url",
    "sagemaker_base_url",
    "deployment_url",
    # Observability credentials, hosts, and project identifiers: derived
    # from the canonical ``_supported_callback_params`` allowlist so new
    # integrations are covered automatically. Sorted for stable iteration
    # order and reviewable diffs.
    *sorted(_build_banned_observability_params()),
)


def _check_banned_params(
    body: dict,
    general_settings: dict,
    llm_router: Optional[Router],
    model: str,
) -> None:
    """Raise ``ValueError`` if ``body`` carries a banned param without admin opt-in.

    Shared between the root-level check and the nested-config check so a
    new banned param only needs to be added in one place.
    """
    for param in _BANNED_REQUEST_BODY_PARAMS:
        if param not in body:
            continue
        if general_settings.get("allow_client_side_credentials") is True:
            # Proxy-wide opt-in: every banned param is permitted, exit
            # entirely so the rest of the loop doesn't waste work.
            return
        if (
            _allow_model_level_clientside_configurable_parameters(
                model=model,
                param=param,
                request_body_value=body[param],
                llm_router=llm_router,
            )
            is True
        ):
            # Per-param opt-in: only THIS param is permitted by the
            # deployment's ``configurable_clientside_auth_params``. Skip
            # to the next banned param so a body that pairs an allowed
            # ``api_base`` with an unallowed ``langfuse_host`` is still
            # rejected for the second field.
            continue
        raise ValueError(
            f"Rejected Request: {param} is not allowed in request body. "
            "Clientside passthrough requires explicit admin opt-in via "
            "either `general_settings.allow_client_side_credentials = true` "
            "(proxy-wide) or `configurable_clientside_auth_params` on the "
            "deployment in your proxy config.yaml. "
            "Relevant Issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997",
        )


def is_request_body_safe(
    request_body: dict, general_settings: dict, llm_router: Optional[Router], model: str
) -> bool:
    """
    Check if the request body is safe.

    A malicious user can set the ﻿api_base to their own domain and invoke POST /chat/completions to intercept and steal the OpenAI API key.
    Relevant issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997

    The blocklist is enforced unconditionally. Legitimate clientside
    credential / endpoint passthrough goes through one of the two
    explicit admin opt-ins (``general_settings.allow_client_side_credentials``
    proxy-wide or ``configurable_clientside_auth_params`` per deployment).
    Historically there was a third, *implicit*, *caller-controlled* path:
    ``check_complete_credentials`` returned True when the caller supplied
    any non-empty ``api_key``, which made the entire blocklist a no-op.
    That bypass turned every missing entry on the blocklist into an
    exploitable SSRF / credential-exfil hole — see GHSA-jh89-88fc-qrfp,
    GHSA-3frq-6r6h-7j64, and the chain of veria-admin findings (Dv_m860l,
    b_yRJeQ5, stN90yjP, LBlyOAc8, U2TD78kg). Removed: the blocklist now
    has a single, predictable failure mode for missing entries (a 400),
    not a credential leak.

    Iterative single-level descent into ``_NESTED_CONFIG_KEYS`` (rather
    than recursion) covers nested-config attacks like Milvus's
    ``litellm_embedding_config.api_base`` (VERIA-6) without exposing a
    recursion-depth DoS surface.
    """
    _check_banned_params(request_body, general_settings, llm_router, model)
    for nested_key in _NESTED_CONFIG_KEYS:
        nested = request_body.get(nested_key)
        if isinstance(nested, dict):
            _check_banned_params(nested, general_settings, llm_router, model)
    for metadata_key in _NESTED_METADATA_KEYS:
        metadata = _coerce_metadata_to_dict(request_body.get(metadata_key))
        if metadata is not None:
            _check_banned_params(metadata, general_settings, llm_router, model)
    return True


def _coerce_metadata_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``value`` as a dict, parsing it from JSON if delivered as a string.

    Multipart/form-data and ``extra_body`` callers send ``litellm_metadata``
    as a JSON-encoded string; the proxy parses it into a dict later in
    ``add_litellm_data_to_request``, but the auth-time bouncer runs first
    and would otherwise miss the banned-param check on a still-stringified
    metadata blob.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

        parsed = safe_json_loads(value)
        if isinstance(parsed, dict):
            return parsed
    return None


async def pre_db_read_auth_checks(
    request: Request,
    request_data: dict,
    route: str,
):
    """
    1. Checks if request size is under max_request_size_mb (if set)
    2. Check if request body is safe (example user has not set api_base in request body)
    3. Check if IP address is allowed (if set)
    4. Check if request route is an allowed route on the proxy (if set)

    Returns:
    - True

    Raises:
    - HTTPException if request fails initial auth checks
    """
    from litellm.proxy.proxy_server import general_settings, llm_router, premium_user

    # Check 1. request size
    await check_if_request_size_is_safe(request=request)

    # Check 2. Request body is safe
    is_request_body_safe(
        request_body=request_data,
        general_settings=general_settings,
        llm_router=llm_router,
        model=request_data.get(
            "model", ""
        ),  # [TODO] use model passed in url as well (azure openai routes)
    )

    # Check 3. Check if IP address is allowed
    is_valid_ip, passed_in_ip = _check_valid_ip(
        allowed_ips=general_settings.get("allowed_ips", None),
        use_x_forwarded_for=general_settings.get("use_x_forwarded_for", False),
        request=request,
    )

    if not is_valid_ip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access forbidden: IP address {passed_in_ip} not allowed.",
        )

    # Check 4. Check if request route is an allowed route on the proxy
    if "allowed_routes" in general_settings:
        _allowed_routes = general_settings["allowed_routes"]
        if premium_user is not True:
            verbose_proxy_logger.error(
                f"Trying to set allowed_routes. This is an Enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            )
        if route not in _allowed_routes:
            verbose_proxy_logger.error(
                f"Route {route} not in allowed_routes={_allowed_routes}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Route {route} not allowed",
            )


def route_in_additonal_public_routes(current_route: str):
    """
    Helper to check if the user defined public_routes on config.yaml

    Parameters:
    - current_route: str - the route the user is trying to call

    Returns:
    - bool - True if the route is defined in public_routes
    - bool - False if the route is not defined in public_routes

    Supports wildcard patterns (e.g., "/api/*" matches "/api/users", "/api/users/123")

    In order to use this the litellm config.yaml should have the following in general_settings:

    ```yaml
    general_settings:
        master_key: sk-1234
        public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate", "/api/*"]
    ```
    """
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.proxy_server import general_settings, premium_user

    try:
        if premium_user is not True:
            return False
        if general_settings is None:
            return False

        routes_defined = general_settings.get("public_routes", [])

        # Check exact match first
        if current_route in routes_defined:
            return True

        # Check wildcard patterns
        for route_pattern in routes_defined:
            if RouteChecks._route_matches_wildcard_pattern(
                route=current_route, pattern=route_pattern
            ):
                return True

        return False
    except Exception as e:
        verbose_proxy_logger.error(f"route_in_additonal_public_routes: {str(e)}")
        return False


def get_request_route(request: Request) -> str:
    """
    Helper to get the route from the request

    remove base url from path if set e.g. `/genai/chat/completions` -> `/chat/completions
    """
    try:
        if hasattr(request, "base_url") and request.url.path.startswith(
            request.base_url.path
        ):
            # remove base_url from path
            return request.url.path[len(request.base_url.path) - 1 :]
        else:
            return request.url.path
    except Exception as e:
        verbose_proxy_logger.debug(
            f"error on get_request_route: {str(e)}, defaulting to request.url.path={request.url.path}"
        )
        return request.url.path


@lru_cache(maxsize=256)
def normalize_request_route(route: str) -> str:
    """
    Normalize request routes by replacing dynamic path parameters with placeholders.

    This prevents high cardinality in Prometheus metrics by collapsing routes like:
    - /v1/responses/1234567890 -> /v1/responses/{response_id}
    - /v1/threads/thread_123 -> /v1/threads/{thread_id}

    Args:
        route: The request route path

    Returns:
        Normalized route with dynamic parameters replaced by placeholders

    Examples:
        >>> normalize_request_route("/v1/responses/abc123")
        '/v1/responses/{response_id}'
        >>> normalize_request_route("/v1/responses/abc123/cancel")
        '/v1/responses/{response_id}/cancel'
        >>> normalize_request_route("/chat/completions")
        '/chat/completions'
    """
    # Define patterns for routes with dynamic IDs
    # Format: (regex_pattern, replacement_template)
    patterns = [
        # Responses API - must come before generic patterns
        (r"^(/(?:openai/)?v1/responses)/([^/]+)(/input_items)$", r"\1/{response_id}\3"),
        (r"^(/(?:openai/)?v1/responses)/([^/]+)(/cancel)$", r"\1/{response_id}\3"),
        (r"^(/(?:openai/)?v1/responses)/([^/]+)$", r"\1/{response_id}"),
        (r"^(/responses)/([^/]+)(/input_items)$", r"\1/{response_id}\3"),
        (r"^(/responses)/([^/]+)(/cancel)$", r"\1/{response_id}\3"),
        (r"^(/responses)/([^/]+)$", r"\1/{response_id}"),
        # Threads API
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)/([^/]+)(/steps)/([^/]+)$",
            r"\1/{thread_id}\3/{run_id}\5/{step_id}",
        ),
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)/([^/]+)(/steps)$",
            r"\1/{thread_id}\3/{run_id}\5",
        ),
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)/([^/]+)(/cancel)$",
            r"\1/{thread_id}\3/{run_id}\5",
        ),
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)/([^/]+)(/submit_tool_outputs)$",
            r"\1/{thread_id}\3/{run_id}\5",
        ),
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)/([^/]+)$",
            r"\1/{thread_id}\3/{run_id}",
        ),
        (r"^(/(?:openai/)?v1/threads)/([^/]+)(/runs)$", r"\1/{thread_id}\3"),
        (
            r"^(/(?:openai/)?v1/threads)/([^/]+)(/messages)/([^/]+)$",
            r"\1/{thread_id}\3/{message_id}",
        ),
        (r"^(/(?:openai/)?v1/threads)/([^/]+)(/messages)$", r"\1/{thread_id}\3"),
        (r"^(/(?:openai/)?v1/threads)/([^/]+)$", r"\1/{thread_id}"),
        # Vector Stores API
        (
            r"^(/(?:openai/)?v1/vector_stores)/([^/]+)(/files)/([^/]+)$",
            r"\1/{vector_store_id}\3/{file_id}",
        ),
        (
            r"^(/(?:openai/)?v1/vector_stores)/([^/]+)(/files)$",
            r"\1/{vector_store_id}\3",
        ),
        (
            r"^(/(?:openai/)?v1/vector_stores)/([^/]+)(/file_batches)/([^/]+)$",
            r"\1/{vector_store_id}\3/{batch_id}",
        ),
        (
            r"^(/(?:openai/)?v1/vector_stores)/([^/]+)(/file_batches)$",
            r"\1/{vector_store_id}\3",
        ),
        (r"^(/(?:openai/)?v1/vector_stores)/([^/]+)$", r"\1/{vector_store_id}"),
        # Assistants API
        (r"^(/(?:openai/)?v1/assistants)/([^/]+)$", r"\1/{assistant_id}"),
        # Files API
        (r"^(/(?:openai/)?v1/files)/([^/]+)(/content)$", r"\1/{file_id}\3"),
        (r"^(/(?:openai/)?v1/files)/([^/]+)$", r"\1/{file_id}"),
        # Batches API
        (r"^(/(?:openai/)?v1/batches)/([^/]+)(/cancel)$", r"\1/{batch_id}\3"),
        (r"^(/(?:openai/)?v1/batches)/([^/]+)$", r"\1/{batch_id}"),
        # Fine-tuning API
        (
            r"^(/(?:openai/)?v1/fine_tuning/jobs)/([^/]+)(/events)$",
            r"\1/{fine_tuning_job_id}\3",
        ),
        (
            r"^(/(?:openai/)?v1/fine_tuning/jobs)/([^/]+)(/cancel)$",
            r"\1/{fine_tuning_job_id}\3",
        ),
        (
            r"^(/(?:openai/)?v1/fine_tuning/jobs)/([^/]+)(/checkpoints)$",
            r"\1/{fine_tuning_job_id}\3",
        ),
        (r"^(/(?:openai/)?v1/fine_tuning/jobs)/([^/]+)$", r"\1/{fine_tuning_job_id}"),
        # Models API
        (r"^(/(?:openai/)?v1/models)/([^/]+)$", r"\1/{model}"),
    ]

    # Apply patterns in order
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, route)
        if normalized != route:
            return normalized

    # Return original route if no pattern matched
    return route


async def check_if_request_size_is_safe(request: Request) -> bool:
    """
    Enterprise Only:
        - Checks if the request size is within the limit

    Args:
        request (Request): The incoming request.

    Returns:
        bool: True if the request size is within the limit

    Raises:
        ProxyException: If the request size is too large

    """
    from litellm.proxy.proxy_server import general_settings, premium_user

    max_request_size_mb = general_settings.get("max_request_size_mb", None)

    if max_request_size_mb is not None:
        # Check if premium user
        if premium_user is not True:
            verbose_proxy_logger.warning(
                f"using max_request_size_mb - not checking -  this is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
            )
            return True

        # Get the request body
        content_length = request.headers.get("content-length")

        if content_length:
            header_size = int(content_length)
            header_size_mb = bytes_to_mb(bytes_value=header_size)
            verbose_proxy_logger.debug(
                f"content_length request size in MB={header_size_mb}"
            )

            if header_size_mb > max_request_size_mb:
                raise ProxyException(
                    message=f"Request size is too large. Request size is {header_size_mb} MB. Max size is {max_request_size_mb} MB",
                    type=ProxyErrorTypes.bad_request_error.value,
                    code=400,
                    param="content-length",
                )
        else:
            # If Content-Length is not available, read the body
            body = await request.body()
            body_size = len(body)
            request_size_mb = bytes_to_mb(bytes_value=body_size)

            verbose_proxy_logger.debug(
                f"request body request size in MB={request_size_mb}"
            )
            if request_size_mb > max_request_size_mb:
                raise ProxyException(
                    message=f"Request size is too large. Request size is {request_size_mb} MB. Max size is {max_request_size_mb} MB",
                    type=ProxyErrorTypes.bad_request_error.value,
                    code=400,
                    param="content-length",
                )

    return True


async def check_response_size_is_safe(response: Any) -> bool:
    """
    Enterprise Only:
        - Checks if the response size is within the limit

    Args:
        response (Any): The response to check.

    Returns:
        bool: True if the response size is within the limit

    Raises:
        ProxyException: If the response size is too large

    """

    from litellm.proxy.proxy_server import general_settings, premium_user

    max_response_size_mb = general_settings.get("max_response_size_mb", None)
    if max_response_size_mb is not None:
        # Check if premium user
        if premium_user is not True:
            verbose_proxy_logger.warning(
                f"using max_response_size_mb - not checking -  this is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
            )
            return True

        response_size_mb = bytes_to_mb(bytes_value=sys.getsizeof(response))
        verbose_proxy_logger.debug(f"response size in MB={response_size_mb}")
        if response_size_mb > max_response_size_mb:
            raise ProxyException(
                message=f"Response size is too large. Response size is {response_size_mb} MB. Max size is {max_response_size_mb} MB",
                type=ProxyErrorTypes.bad_request_error.value,
                code=400,
                param="content-length",
            )

    return True


def bytes_to_mb(bytes_value: int):
    """
    Helper to convert bytes to MB
    """
    return bytes_value / (1024 * 1024)


# helpers used by parallel request limiter to handle model rpm/tpm limits for a given api key
def _get_deployment_default_limit(model_name: str, field: str) -> Optional[int]:
    """
    Return the minimum value of `field` across all deployments for model_name,
    or None if no deployment has the field set.

    When multiple deployments share the same model name, taking the minimum is
    the safest choice for load-balanced setups: it ensures no deployment is
    over-consumed regardless of which one actually serves a given request.
    """
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None
    deployments = llm_router.get_model_list(model_name=model_name)
    if not deployments:
        return None
    limits = []
    for deployment in deployments:
        raw = deployment.get("litellm_params", {}).get(field)
        if raw is not None:
            try:
                if isinstance(raw, (int, float, str, bytes, bytearray)):
                    limits.append(int(raw))
            except (ValueError, TypeError):
                pass
    return min(limits) if limits else None


def _get_deployment_default_rpm_limit(model_name: str) -> Optional[int]:
    return _get_deployment_default_limit(model_name, "default_api_key_rpm_limit")


def _get_deployment_default_tpm_limit(model_name: str) -> Optional[int]:
    return _get_deployment_default_limit(model_name, "default_api_key_tpm_limit")


def get_key_model_rpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, int]]:
    """
    Get the model rpm limit for a given api key.

    Priority order (returns first found):
    1. Key metadata (model_rpm_limit)
    2. Key model_max_budget (rpm_limit per model)
    3. Team metadata (model_rpm_limit)
    4. Deployment default_api_key_rpm_limit (when model_name is provided)
    """
    # 1. Check key metadata first (takes priority)
    if user_api_key_dict.metadata:
        result = user_api_key_dict.metadata.get("model_rpm_limit")
        if result:
            return result

    # 2. Check model_max_budget
    if user_api_key_dict.model_max_budget:
        model_rpm_limit: Dict[str, Any] = {}
        for model, budget in user_api_key_dict.model_max_budget.items():
            if isinstance(budget, dict) and budget.get("rpm_limit") is not None:
                model_rpm_limit[model] = budget["rpm_limit"]
        if model_rpm_limit:
            return model_rpm_limit

    # 3. Fallback to team metadata
    if user_api_key_dict.team_metadata:
        team_limit = user_api_key_dict.team_metadata.get("model_rpm_limit")
        if team_limit is not None:
            return team_limit

    # 4. Fallback to deployment default_api_key_rpm_limit
    if model_name is not None:
        default_limit = _get_deployment_default_rpm_limit(model_name)
        if default_limit is not None:
            return {model_name: default_limit}

    return None


def get_key_model_tpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, int]]:
    """
    Get the model tpm limit for a given api key.

    Priority order (returns first found):
    1. Key metadata (model_tpm_limit)
    2. Key model_max_budget (tpm_limit per model)
    3. Team metadata (model_tpm_limit)
    4. Deployment default_api_key_tpm_limit (when model_name is provided)
    """
    # 1. Check key metadata first (takes priority)
    if user_api_key_dict.metadata:
        result = user_api_key_dict.metadata.get("model_tpm_limit")
        if result:
            return result

    # 2. Check model_max_budget (iterate per-model like RPM does)
    if user_api_key_dict.model_max_budget:
        model_tpm_limit: Dict[str, Any] = {}
        for model, budget in user_api_key_dict.model_max_budget.items():
            if isinstance(budget, dict) and budget.get("tpm_limit") is not None:
                model_tpm_limit[model] = budget["tpm_limit"]
        if model_tpm_limit:
            return model_tpm_limit

    # 3. Fallback to team metadata
    if user_api_key_dict.team_metadata:
        team_limit = user_api_key_dict.team_metadata.get("model_tpm_limit")
        if team_limit is not None:
            return team_limit

    # 4. Fallback to deployment default_api_key_tpm_limit
    if model_name is not None:
        default_limit = _get_deployment_default_tpm_limit(model_name)
        if default_limit is not None:
            return {model_name: default_limit}

    return None


def get_model_rate_limit_from_metadata(
    user_api_key_dict: UserAPIKeyAuth,
    metadata_accessor_key: Literal[
        "team_metadata", "organization_metadata", "project_metadata"
    ],
    rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
) -> Optional[Dict[str, int]]:
    if getattr(user_api_key_dict, metadata_accessor_key):
        return getattr(user_api_key_dict, metadata_accessor_key).get(rate_limit_key)
    return None


def get_team_model_rpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.team_metadata:
        return user_api_key_dict.team_metadata.get("model_rpm_limit")
    return None


def get_team_model_tpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.team_metadata:
        return user_api_key_dict.team_metadata.get("model_tpm_limit")
    return None


def get_project_model_rpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.project_metadata:
        return user_api_key_dict.project_metadata.get("model_rpm_limit")
    return None


def get_project_model_tpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.project_metadata:
        return user_api_key_dict.project_metadata.get("model_tpm_limit")
    return None


def is_pass_through_provider_route(route: str) -> bool:
    PROVIDER_SPECIFIC_PASS_THROUGH_ROUTES = [
        "vertex-ai",
    ]

    # check if any of the prefixes are in the route
    for prefix in PROVIDER_SPECIFIC_PASS_THROUGH_ROUTES:
        if prefix in route:
            return True

    return False


def _has_user_setup_sso():
    """
    Check if the user has set up single sign-on (SSO) by verifying the presence of Microsoft client ID, Google client ID or generic client ID and UI username environment variables.
    Returns a boolean indicating whether SSO has been set up.
    """
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

    sso_setup = (
        (microsoft_client_id is not None)
        or (google_client_id is not None)
        or (generic_client_id is not None)
    )

    return sso_setup


def get_customer_user_header_from_mapping(user_id_mapping) -> Optional[list]:
    """Return the header_name mapped to CUSTOMER role, if any (dict-based)."""
    if not user_id_mapping:
        return None
    items = user_id_mapping if isinstance(user_id_mapping, list) else [user_id_mapping]
    customer_headers_mappings = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("litellm_user_role")
        header_name = item.get("header_name")
        if role is None or not header_name:
            continue
        if str(role).lower() == str(LitellmUserRoles.CUSTOMER).lower():
            customer_headers_mappings.append(header_name.lower())

    if customer_headers_mappings:
        return customer_headers_mappings

    return None


def _get_customer_id_from_standard_headers(
    request_headers: Optional[dict],
) -> Optional[str]:
    """
    Check standard customer ID headers for a customer/end-user ID.

    This enables tools like Claude Code to pass customer IDs via ANTHROPIC_CUSTOM_HEADERS.
    No configuration required - these headers are always checked.

    Args:
        request_headers: The request headers dict

    Returns:
        The customer ID if found in standard headers, None otherwise
    """
    if request_headers is None:
        return None

    for standard_header in STANDARD_CUSTOMER_ID_HEADERS:
        for header_name, header_value in request_headers.items():
            if header_name.lower() == standard_header.lower():
                user_id_str = str(header_value) if header_value is not None else ""
                if user_id_str.strip():
                    return user_id_str
    return None


def get_end_user_id_from_request_body(
    request_body: dict, request_headers: Optional[dict] = None
) -> Optional[str]:
    # Import general_settings here to avoid potential circular import issues at module level
    # and to ensure it's fetched at runtime.
    from litellm.proxy.proxy_server import general_settings

    # Check 1: Standard customer ID headers (always checked, no configuration required)
    customer_id = _get_customer_id_from_standard_headers(
        request_headers=request_headers
    )
    if customer_id is not None:
        return customer_id

    # Check 2: Follow the user header mappings feature, if not found, then check for deprecated user_header_name (only if request_headers is provided)
    # User query: "system not respecting user_header_name property"
    # This implies the key in general_settings is 'user_header_name'.
    if request_headers is not None:
        custom_header_name_to_check: Optional[Union[list, str]] = None

        # Prefer user mappings (new behavior)
        user_id_mapping = general_settings.get("user_header_mappings", None)
        if user_id_mapping:
            custom_header_name_to_check = get_customer_user_header_from_mapping(
                user_id_mapping
            )

        # Fallback to deprecated user_header_name if mapping did not specify
        if not custom_header_name_to_check:
            user_id_header_config_key = "user_header_name"
            value = general_settings.get(user_id_header_config_key)
            if isinstance(value, str) and value.strip() != "":
                custom_header_name_to_check = value

        # If we have a header name to check, try to read it from request headers
        if isinstance(custom_header_name_to_check, list):
            headers_lower = {k.lower(): v for k, v in request_headers.items()}
            for expected_header in custom_header_name_to_check:
                header_value = headers_lower.get(expected_header)
                if header_value is not None:
                    user_id_str = str(header_value)
                    if user_id_str.strip():
                        return user_id_str

        elif isinstance(custom_header_name_to_check, str):
            for header_name, header_value in request_headers.items():
                if header_name.lower() == custom_header_name_to_check.lower():
                    user_id_str = str(header_value) if header_value is not None else ""
                    if user_id_str.strip():
                        return user_id_str

    # Check 3: 'user' field in request_body (commonly OpenAI)
    if "user" in request_body and request_body["user"] is not None:
        user_from_body_user_field = request_body["user"]
        return str(user_from_body_user_field)

    def _as_dict(value: Any) -> dict:
        # metadata / litellm_metadata can arrive as JSON strings from
        # multipart/form-data or extra_body; coerce so string-encoded
        # payloads can't evade end-user attribution.
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

            parsed = safe_json_loads(value)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # Check 4: 'litellm_metadata.user' in request_body (commonly Anthropic)
    litellm_metadata = _as_dict(request_body.get("litellm_metadata"))
    user_from_litellm_metadata = litellm_metadata.get("user")
    if user_from_litellm_metadata is not None:
        return str(user_from_litellm_metadata)

    # Check 5: 'metadata.user_id' in request_body (another common pattern)
    metadata_dict = _as_dict(request_body.get("metadata"))
    user_id_from_metadata_field = metadata_dict.get("user_id")
    if user_id_from_metadata_field is not None:
        return str(user_id_from_metadata_field)

    # Check 6: 'safety_identifier' in request body (OpenAI Responses API parameter)
    # SECURITY NOTE: safety_identifier can be set by any caller in the request body.
    # Only use this for end-user identification in trusted environments where you control
    # the calling application. For untrusted callers, prefer using headers or server-side
    # middleware to set the end_user_id to prevent impersonation.
    if request_body.get("safety_identifier") is not None:
        user_from_body_user_field = request_body["safety_identifier"]
        return str(user_from_body_user_field)

    return None


MODEL_ROUTING_HEADER_NAME = "x-litellm-model"
_MODEL_ROUTING_ROUTE_MARKERS = (
    "/files",
    "/batches",
    "/vector_stores",
    "/skills",
    "/evals",
    "/fine_tuning",
    "/videos",
)
_MODEL_ROUTING_HEADER_OR_QUERY_ROUTE_MARKERS = (
    "/files",
    "/batches",
    "/skills",
    "/evals",
)
_MODEL_ROUTING_QUERY_TARGET_MODEL_ROUTE_MARKERS = (
    "/files",
    "/batches",
    "/fine_tuning",
)
_MODEL_ROUTING_BODY_TARGET_MODEL_ROUTE_MARKERS = (
    "/files",
    "/batches",
    "/vector_stores",
)
_MODEL_ROUTING_COMPLETION_MODEL_ROUTE_MARKERS = ("/evals",)
_MODEL_ROUTING_ID_FIELDS = (
    "file_id",
    "input_file_id",
    "output_file_id",
    "error_file_id",
    "batch_id",
    "fine_tuning_job_id",
    "training_file",
    "validation_file",
    "vector_store_id",
    "video_id",
    "character_id",
)


def _append_model_candidates(candidates: List[str], value: Any) -> None:
    if value is None:
        return

    values = value if isinstance(value, (list, tuple, set)) else [value]
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            model_names = [model.strip() for model in item.split(",")]
        else:
            model_names = [str(item).strip()]
        candidates.extend(model for model in model_names if model)


def _dedupe_model_candidates(candidates: List[str]) -> List[str]:
    deduped: List[str] = []
    for model in candidates:
        if model not in deduped:
            deduped.append(model)
    return deduped


def _get_case_insensitive_mapping_value(
    mapping: Optional[Mapping[str, Any]], key: str
) -> Any:
    if not mapping:
        return None
    if key in mapping:
        return mapping[key]
    key_lower = key.lower()
    for mapping_key, value in mapping.items():
        if str(mapping_key).lower() == key_lower:
            return value
    return None


def _route_matches_any_marker(route: str, markers: Tuple[str, ...]) -> bool:
    normalized_route = route.lower()
    return any(marker in normalized_route for marker in markers)


def _route_uses_model_routing_sources(route: str) -> bool:
    return _route_matches_any_marker(route=route, markers=_MODEL_ROUTING_ROUTE_MARKERS)


def _extract_models_from_managed_resource_id(
    resource_id: Any, resource_id_field: Optional[str] = None
) -> List[str]:
    if not isinstance(resource_id, str) or not resource_id:
        return []

    candidates: List[str] = []

    try:
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            decode_model_from_file_id,
            get_model_id_from_unified_batch_id,
            get_models_from_unified_file_id,
        )

        _append_model_candidates(
            candidates=candidates, value=decode_model_from_file_id(resource_id)
        )
        unified_file_id = _is_base64_encoded_unified_file_id(resource_id)
        if unified_file_id:
            _append_model_candidates(
                candidates=candidates,
                value=get_models_from_unified_file_id(unified_file_id),
            )
            _append_model_candidates(
                candidates=candidates,
                value=get_model_id_from_unified_batch_id(unified_file_id),
            )
    except Exception as e:
        verbose_proxy_logger.debug(
            "Unable to extract model from managed file/batch ID: %s", str(e)
        )

    try:
        from litellm.llms.base_llm.managed_resources.utils import parse_unified_id

        parsed_id = parse_unified_id(resource_id)
        if parsed_id:
            _append_model_candidates(
                candidates=candidates, value=parsed_id.get("model_id")
            )
            _append_model_candidates(
                candidates=candidates, value=parsed_id.get("target_model_names")
            )
    except Exception as e:
        verbose_proxy_logger.debug(
            "Unable to extract model from unified managed resource ID: %s", str(e)
        )

    if resource_id_field in ("video_id", "character_id"):
        try:
            from litellm.types.videos.utils import (
                decode_character_id_with_provider,
                decode_video_id_with_provider,
            )

            if resource_id_field == "video_id":
                _append_model_candidates(
                    candidates=candidates,
                    value=decode_video_id_with_provider(resource_id).get("model_id"),
                )
            else:
                _append_model_candidates(
                    candidates=candidates,
                    value=decode_character_id_with_provider(resource_id).get(
                        "model_id"
                    ),
                )
        except Exception as e:
            verbose_proxy_logger.debug(
                "Unable to extract model from managed video/character ID: %s", str(e)
            )

    return _dedupe_model_candidates(candidates)


def _extract_model_candidates_from_request(
    request_data: dict,
    route: str,
    request_headers: Optional[Mapping[str, Any]] = None,
    request_query_params: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    candidates: List[str] = []
    uses_model_routing_sources = _route_uses_model_routing_sources(route=route)
    uses_header_or_query_model_sources = _route_matches_any_marker(
        route=route, markers=_MODEL_ROUTING_HEADER_OR_QUERY_ROUTE_MARKERS
    )
    uses_query_target_model_sources = _route_matches_any_marker(
        route=route, markers=_MODEL_ROUTING_QUERY_TARGET_MODEL_ROUTE_MARKERS
    )
    uses_body_target_model_sources = _route_matches_any_marker(
        route=route, markers=_MODEL_ROUTING_BODY_TARGET_MODEL_ROUTE_MARKERS
    )
    uses_completion_model_sources = _route_matches_any_marker(
        route=route, markers=_MODEL_ROUTING_COMPLETION_MODEL_ROUTE_MARKERS
    )

    body_model = request_data.get("model")
    _append_model_candidates(candidates, body_model)
    if uses_body_target_model_sources or not body_model:
        _append_model_candidates(candidates, request_data.get("target_model_names"))
    if uses_completion_model_sources and isinstance(
        request_data.get("completion"), dict
    ):
        _append_model_candidates(candidates, request_data["completion"].get("model"))

    if uses_model_routing_sources:
        if uses_header_or_query_model_sources:
            _append_model_candidates(
                candidates,
                _get_case_insensitive_mapping_value(request_query_params, "model"),
            )
            _append_model_candidates(
                candidates,
                _get_case_insensitive_mapping_value(
                    request_headers, MODEL_ROUTING_HEADER_NAME
                ),
            )
        if uses_query_target_model_sources:
            _append_model_candidates(
                candidates,
                _get_case_insensitive_mapping_value(
                    request_query_params, "target_model_names"
                ),
            )

        for field in _MODEL_ROUTING_ID_FIELDS:
            _append_model_candidates(
                candidates,
                _extract_models_from_managed_resource_id(
                    request_data.get(field), resource_id_field=field
                ),
            )

    return _dedupe_model_candidates(candidates)


def _format_model_candidates(
    candidates: List[str],
) -> Optional[Union[str, List[str]]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return candidates


def get_model_from_request(
    request_data: dict,
    route: str,
    request_headers: Optional[Mapping[str, Any]] = None,
    request_query_params: Optional[Mapping[str, Any]] = None,
) -> Optional[Union[str, List[str]]]:
    candidates = _extract_model_candidates_from_request(
        request_data=request_data,
        route=route,
        request_headers=request_headers,
        request_query_params=request_query_params,
    )
    model = _format_model_candidates(candidates)

    # If no explicit model was found, try to extract from route
    if model is None:
        # Parse model from route that follows the pattern /openai/deployments/{model}/*
        match = re.match(r"/openai/deployments/([^/]+)", route)
        if match:
            model = match.group(1)

    # If still not found, extract model from Google generateContent-style routes.
    # These routes put the model in the path and allow "/" inside the model id.
    # Examples:
    # - /v1beta/models/gemini-2.0-flash:generateContent
    # - /v1beta/models/bedrock/claude-sonnet-3.7:generateContent
    # - /models/custom/ns/model:streamGenerateContent
    if model is None and not route.lower().startswith("/vertex"):
        google_match = re.search(r"/(?:v1beta|beta)/models/([^:]+):", route)
        if google_match:
            model = google_match.group(1)

    if model is None and not route.lower().startswith("/vertex"):
        google_match = re.search(r"^/models/([^:]+):", route)
        if google_match:
            model = google_match.group(1)

    # If still not found, extract from Vertex AI passthrough route
    # Pattern: /vertex_ai/.../models/{model_id}:*
    # Example: /vertex_ai/v1/.../models/gemini-1.5-pro:generateContent
    if model is None and route.lower().startswith("/vertex"):
        vertex_match = re.search(r"/models/([^:]+)", route)
        if vertex_match:
            model = vertex_match.group(1)

    return model


def abbreviate_api_key(api_key: str) -> str:
    return f"sk-...{api_key[-4:]}"
