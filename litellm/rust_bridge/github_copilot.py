"""The GitHub Copilot session a native call sends with: Python owns every Copilot credential.

Rust receives only the resolved ``(token, api_base)`` pair and never sees which mode produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from litellm.exceptions import AuthenticationError, BadRequestError


class SharedLogin(Protocol):
    def get_api_key(self) -> str: ...

    def get_api_base(self) -> str | None: ...


def _provider(model: str, custom_llm_provider: str | None) -> str | None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    try:
        return get_llm_provider(model, custom_llm_provider)[1]
    except BadRequestError:
        return custom_llm_provider


def resolve(arguments: Mapping[str, object], shared_login: SharedLogin) -> tuple[str, str]:
    from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAPIKeyError
    from litellm.llms.github_copilot.per_user_auth import require_github_copilot_user_session

    per_user: Final = require_github_copilot_user_session(arguments)
    if per_user is not None:
        return per_user.token, per_user.api_base
    try:
        token: Final = shared_login.get_api_key()
    except GetAPIKeyError as e:
        raise AuthenticationError(message=str(e), llm_provider="github_copilot", model="") from e
    return token, (shared_login.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE).rstrip("/")


def session(
    model: str,
    custom_llm_provider: str | None,
    arguments: dict[str, object],  # mutable-ok: the per-user attach writes the session into the call's kwargs
) -> tuple[str, str] | None:
    if _provider(model, custom_llm_provider) != "github_copilot":
        return None
    from litellm.llms.github_copilot.authenticator import Authenticator
    from litellm.llms.github_copilot.per_user_auth import attach_github_copilot_user_session

    attach_github_copilot_user_session(arguments)
    return resolve(arguments, Authenticator())
