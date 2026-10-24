from collections.abc import Mapping
from typing import Final

import litellm
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_str
from litellm.types.tokenize import TokenizeResponse

_PROJECTED_FIELDS: Final = frozenset(
    {
        "model",
        "prompt",
        "timeout",
        "base_url",
        "extra_headers",
        "include",
        "add_special_tokens",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "user",
        "metadata",
        "proxy_server_request",
        "litellm_logging_obj",
        "_tokenize_context",
    }
)
_OPERATIONAL_FIELDS: Final = frozenset(
    {
        "client",
        "shared_session",
        "request_timeout",
        "max_retries",
        "num_retries",
        "fallbacks",
        "retry_policy",
        "retry_strategy",
        "caching",
        "cache",
        "cache_key",
        "preset_cache_key",
        "model_info",
        "rpm",
        "tpm",
        "max_parallel_requests",
        "weight",
        "order",
        "litellm_call_id",
        "completion_call_id",
        "litellm_trace_id",
        "litellm_session_id",
        "litellm_metadata",
        "litellm_params",
        "logger_fn",
        "verbose",
        "no-log",
        "tags",
        "success_callback",
        "failure_callback",
        "async_success_callback",
        "async_failure_callback",
        "litellm_credential_name",
        "configurable_clientside_auth_params",
        "secret_fields",
    }
)


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    return {
        name: value
        for name, value in request.items()
        if name not in _PROJECTED_FIELDS and name not in _OPERATIONAL_FIELDS
    }


def deployment_params(request: Mapping[str, object]) -> Mapping[str, object]:
    from litellm.tokenize.routing import deployment_params as routing_deployment_params

    return routing_deployment_params(request)


def source_for(request: Mapping[str, object], name: str) -> str:
    from litellm.tokenize.routing import source_for as routing_source_for

    return routing_source_for(request, name)


def response(value: Mapping[str, object]) -> TokenizeResponse:
    response: Final = TokenizeResponse.model_validate(value)
    object.__setattr__(
        response,
        "_hidden_params",
        {
            "response_cost": 0,
            "x-litellm-rust": True,
            "additional_headers": {
                "x-litellm-response-cost": "0",
                "x-litellm-rust": "true",
            },
        },
    )
    return response


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "tokenize"
    if isinstance(error, (TypeError, ValueError)):
        return litellm.BadRequestError(message=str(error), model=str(request["model"]), llm_provider=provider)
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")),
    )
