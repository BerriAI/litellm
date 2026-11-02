"""
Fallback configuration checks ``RustCore`` runs on the host: ``validate_fallbacks``, with the
Python router's error texts. router-core walks the fallbacks; this module only refuses a list it
could not walk the way the Python router would, before the core is built.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.router_utils.fallback_event_handlers import check_non_standard_fallback_format


def refuse_malformed_fallbacks(fallback_param: Sequence[object] | None) -> None:
    """``Router.validate_fallbacks``: every entry is a one-key dict whose targets are a list (or a
    client-side dict target)."""
    if fallback_param is None:
        return
    for fallback_dict in fallback_param:
        if not isinstance(fallback_dict, dict):
            raise ValueError(f"Item '{fallback_dict}' is not a dictionary.")  # noqa: TRY004  # the Python router's error class
        if len(fallback_dict) != 1:  # pyright: ignore[reportUnknownArgumentType]  # only the size is read
            raise ValueError(
                f"Dictionary '{fallback_dict}' must have exactly one key, but has {len(fallback_dict)} keys."  # pyright: ignore[reportUnknownArgumentType]  # see above
            )
        ((model_group, targets),) = fallback_dict.items()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # an entry as configured
        entry: list[object] = [fallback_dict]  # mutable-ok: the one-entry list the check reads
        if isinstance(targets, list) or check_non_standard_fallback_format(fallbacks=entry):
            continue
        raise TypeError(
            f"Fallbacks for '{model_group}' must be a list of model names, got '{targets}'. "
            f"Use {{'{model_group}': ['{targets}']}}"
        )


def rejected_fallback_target(request: Mapping[str, object]) -> Exception | None:
    """The error ``run_async_fallback`` raises before its walk when a client-side request list
    carries a dict target the request may not set (a server-owned federation field, an ``oidc/``
    api key): every target of that walk is refused."""
    from litellm.litellm_core_utils.credential_accessor import reject_api_key_secret_reference
    from litellm.types.router import reject_server_owned_wif_params

    fallbacks: Final = request.get("fallbacks")
    if not isinstance(fallbacks, list) or not check_non_standard_fallback_format(fallbacks=fallbacks):  # pyright: ignore[reportUnknownArgumentType]  # a request list as given
        return None
    for target in fallbacks:  # pyright: ignore[reportUnknownVariableType]  # see above
        if isinstance(target, dict):
            try:
                reject_server_owned_wif_params(target)  # pyright: ignore[reportUnknownArgumentType]  # see above
                reject_api_key_secret_reference(target)  # pyright: ignore[reportUnknownArgumentType]  # see above
            except Exception as error:  # noqa: BLE001  # the walk's own refusal, reported as its failure
                return error
    return None


# The `mock_testing_*` flags the router reads, and the mock failure each asks router-core for.
MOCK_FLAGS: Final = MappingProxyType(
    {
        "mock_testing_fallbacks": "fallbacks",
        "mock_testing_context_fallbacks": "context_window",
        "mock_testing_content_policy_fallbacks": "content_policy",
        "mock_testing_rate_limit_error": "rate_limit",
    }
)


def mock_failure(request: Mapping[str, object]) -> str | None:
    """The mock failure a request asks for (``MockRouterTestingParams.from_kwargs``, then
    ``_handle_mock_testing_rate_limit_error``): the first fallback flag that is ``True`` (a string
    is read as a bool), else the rate limit flag when it is ``True`` itself."""
    from litellm.secret_managers.main import str_to_bool

    for flag in ("mock_testing_fallbacks", "mock_testing_context_fallbacks", "mock_testing_content_policy_fallbacks"):
        value = request.get(flag)
        if (str_to_bool(value) if isinstance(value, str) else value) is True:
            return MOCK_FLAGS[flag]
    return "rate_limit" if request.get("mock_testing_rate_limit_error") is True else None


def mock_error(
    mock: str, model: str, num_retries: object, request: Mapping[str, object], configured: Mapping[str, object]
) -> Exception:
    """The exception the Python router raises for a mock failure; ``configured`` holds the
    router's fallback lists, which the request's own replace."""
    import litellm.exceptions as litellm_exceptions

    def listed(name: str) -> object:
        return request.get(name, configured.get(name))

    if mock == "rate_limit":
        return litellm_exceptions.RateLimitError(
            model=model,
            llm_provider="",
            message=f"This is a mock exception for model={model}, to trigger a rate limit error.",
            num_retries=num_retries if isinstance(num_retries, int) else None,
        )
    if mock == "context_window":
        return litellm_exceptions.ContextWindowExceededError(
            model=model,
            llm_provider="",
            message=f"This is a mock exception for model={model}, to trigger a fallback. \
                    Context_Window_Fallbacks={listed('context_window_fallbacks')}",
        )
    if mock == "content_policy":
        return litellm_exceptions.ContentPolicyViolationError(
            model=model,
            llm_provider="",
            message=f"This is a mock exception for model={model}, to trigger a fallback. \
                    Context_Policy_Fallbacks={listed('content_policy_fallbacks')}",
        )
    return litellm_exceptions.InternalServerError(
        model=model,
        llm_provider="",
        message=f"This is a mock exception for model={model}, to trigger a fallback. Fallbacks={listed('fallbacks')}",
    )


class RustFallbacks:
    """Fallback members served on the host."""

    def validate_fallbacks(self, fallback_param: Sequence[object] | None) -> None:
        refuse_malformed_fallbacks(fallback_param)


__all__ = [
    "MOCK_FLAGS",
    "RustFallbacks",
    "mock_error",
    "mock_failure",
    "refuse_malformed_fallbacks",
    "rejected_fallback_target",
]
