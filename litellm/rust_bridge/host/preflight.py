"""The SDK request policy the native driver runs before a route's host projects.

These are the `@client` prologue steps after `function_setup` and the deployment hook:
credential-name inheritance and the budget and retry-count limits. Rust owns the
inheritance itself; it borrows only the globals below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import CredentialItem


def credential_list() -> list[CredentialItem]:
    from litellm import credential_list as credentials

    return credentials


def warn_unknown_credential(name: str, loaded: int) -> None:
    from litellm._logging import verbose_logger

    verbose_logger.warning(
        "litellm_credential_name=%s matched none of the %d loaded credentials; the request runs without it",
        name,
        loaded,
    )


def check_limits(kwargs: Mapping[str, object]) -> None:
    from litellm import (
        BudgetExceededError,
        _current_cost,  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
        max_budget,
        num_retries_per_request,
    )
    from litellm.litellm_core_utils.core_helpers import max_retries_per_request_hit

    if max_budget and _current_cost > max_budget:
        raise BudgetExceededError(current_cost=_current_cost, max_budget=max_budget)
    if max_retries_per_request_hit(kwargs, num_retries_per_request):
        raise RuntimeError("Max retries per request hit!")
