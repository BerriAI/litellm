"""Resolve how many retries a RetryPolicy grants for a given exception."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.types.router import RetryPolicy

_RETRIES_BY_EXCEPTION_TYPE: Final[Mapping[type, Callable[[RetryPolicy], int | None]]] = MappingProxyType(
    {
        AuthenticationError: lambda policy: policy.AuthenticationErrorRetries,
        Timeout: lambda policy: policy.TimeoutErrorRetries,
        RateLimitError: lambda policy: policy.RateLimitErrorRetries,
        ContentPolicyViolationError: lambda policy: policy.ContentPolicyViolationErrorRetries,
        BadRequestError: lambda policy: policy.BadRequestErrorRetries,
        ServiceUnavailableError: lambda policy: policy.ServiceUnavailableErrorRetries,
        InternalServerError: lambda policy: policy.InternalServerErrorRetries,
    }
)


def _resolve_policy(
    retry_policy: RetryPolicy | Mapping[str, int | None] | None,
    model_group: str | None,
    model_group_retry_policy: Mapping[str, RetryPolicy | Mapping[str, int | None]] | None,
) -> RetryPolicy | None:
    selected: Final = (
        model_group_retry_policy[model_group]
        if model_group_retry_policy is not None and model_group is not None and model_group in model_group_retry_policy
        else retry_policy
    )
    if isinstance(selected, Mapping):
        return RetryPolicy(**selected)
    return selected


def get_num_retries_from_retry_policy(
    exception: Exception,
    retry_policy: RetryPolicy | Mapping[str, int | None] | None = None,
    model_group: str | None = None,
    model_group_retry_policy: Mapping[str, RetryPolicy | Mapping[str, int | None]] | None = None,
) -> int | None:
    """Walk the exception's MRO, most specific class first, and return the first configured retry count."""
    policy: Final = _resolve_policy(retry_policy, model_group, model_group_retry_policy)
    if policy is None:
        return None
    configured: Final = (
        _RETRIES_BY_EXCEPTION_TYPE[cls](policy) for cls in type(exception).__mro__ if cls in _RETRIES_BY_EXCEPTION_TYPE
    )
    return next((retries for retries in configured if retries is not None), policy.DefaultRetries)


def reset_retry_policy() -> RetryPolicy:
    return RetryPolicy()
