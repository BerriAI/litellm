"""
1Password secure sharing for LiteLLM virtual keys.

Writes a virtual key into a 1Password vault as an API Credential item and returns
a 1Password secure-share link (https://share.1password.com/...) that a proxy admin
can hand to an external user or vendor.

Authentication uses a 1Password Service Account token read from the environment
(OP_SERVICE_ACCOUNT_TOKEN); the target vault is OP_VAULT_ID. No proxy config is
required. The official `onepassword-sdk` package must be installed.

All interaction with the (untyped) 1Password SDK is confined to
`_onepassword_sdk_adapter`; this module stays fully typed.
"""

import os
from functools import lru_cache
from typing import Awaitable, List, Optional, Protocol

from pydantic import BaseModel

ONEPASSWORD_SHARE_DURATIONS = (
    "OneHour",
    "OneDay",
    "SevenDays",
    "FourteenDays",
    "ThirtyDays",
)


class OnePasswordShareError(Exception):
    """Raised when a 1Password share operation cannot be completed."""


class OnePasswordConfig(BaseModel):
    service_account_token: str
    vault_id: str


class OnePasswordShareResult(BaseModel):
    share_link: str
    item_id: str
    item_title: str
    expire_after: Optional[str] = None
    one_time_only: bool = False


class ShareBackend(Protocol):
    def __call__(
        self,
        config: OnePasswordConfig,
        title: str,
        secret_value: str,
        recipients: Optional[List[str]],
        expire_after: Optional[str],
        one_time_only: bool,
    ) -> Awaitable[OnePasswordShareResult]: ...


def _default_backend() -> ShareBackend:
    from litellm.integrations._onepassword_sdk_adapter import share_via_sdk

    return share_via_sdk


def get_onepassword_config() -> OnePasswordConfig:
    """Read the 1Password service account token and vault id from the environment."""
    token = os.getenv("OP_SERVICE_ACCOUNT_TOKEN")
    vault_id = os.getenv("OP_VAULT_ID") or os.getenv("OP_VAULT")
    missing = [
        name
        for name, value in (
            ("OP_SERVICE_ACCOUNT_TOKEN", token),
            ("OP_VAULT_ID", vault_id),
        )
        if not value
    ]
    if missing:
        raise OnePasswordShareError(
            "1Password sharing is not configured. Set the following environment "
            f"variable(s) on the proxy: {', '.join(missing)}. "
            "OP_SERVICE_ACCOUNT_TOKEN is a 1Password service account token and "
            "OP_VAULT_ID is the vault the shared key is written to."
        )
    return OnePasswordConfig(service_account_token=str(token), vault_id=str(vault_id))


class OnePasswordClient:
    """
    Writes a secret into 1Password and generates a secure-share link.

    The SDK-backed sharing implementation is injected as ``backend`` so it can be
    replaced with a fake in tests; it defaults to the real 1Password SDK adapter.
    """

    def __init__(self, config: OnePasswordConfig, backend: Optional[ShareBackend] = None):
        self._config = config
        self._backend = backend

    async def share_secret(
        self,
        title: str,
        secret_value: str,
        recipients: Optional[List[str]] = None,
        expire_after: Optional[str] = None,
        one_time_only: bool = False,
    ) -> OnePasswordShareResult:
        """
        Create an API Credential item holding ``secret_value`` and return a
        secure-share link for it.

        Args:
            title: Item title shown in 1Password (e.g. the key alias).
            secret_value: The virtual key value to store.
            recipients: Optional emails/domains to lock the share to. When omitted,
                anyone with the link can view it (subject to account policy).
            expire_after: One of ONEPASSWORD_SHARE_DURATIONS. Defaults to the
                account policy default when omitted.
            one_time_only: Whether the link may be viewed only once per recipient.
        """
        if expire_after is not None and expire_after not in ONEPASSWORD_SHARE_DURATIONS:
            raise OnePasswordShareError(
                f"Invalid expire_after {expire_after!r}. Must be one of {ONEPASSWORD_SHARE_DURATIONS}."
            )

        backend = self._backend if self._backend is not None else _default_backend()
        return await backend(
            self._config,
            title,
            secret_value,
            recipients,
            expire_after,
            one_time_only,
        )


@lru_cache(maxsize=1)
def _cached_client_for(service_account_token: str, vault_id: str) -> OnePasswordClient:
    return OnePasswordClient(OnePasswordConfig(service_account_token=service_account_token, vault_id=vault_id))


def get_cached_onepassword_client() -> OnePasswordClient:
    """
    Return a process-wide cached OnePasswordClient built from the environment.

    A change to the token or vault id transparently rebuilds it.
    """
    config = get_onepassword_config()
    return _cached_client_for(config.service_account_token, config.vault_id)
