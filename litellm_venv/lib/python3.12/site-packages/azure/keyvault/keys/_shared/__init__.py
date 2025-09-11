# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from typing import Optional
from urllib import parse

from .challenge_auth_policy import ChallengeAuthPolicy
from .client_base import KeyVaultClientBase
from .http_challenge import HttpChallenge
from . import http_challenge_cache

HttpChallengeCache = http_challenge_cache  # to avoid aliasing pylint error (C4745)


__all__ = [
    "ChallengeAuthPolicy",
    "HttpChallenge",
    "HttpChallengeCache",
    "KeyVaultClientBase",
]


class KeyVaultResourceId:
    """Represents a Key Vault identifier and its parsed contents.

    :param str source_id: The complete identifier received from Key Vault
    :param str vault_url: The vault URL
    :param str name: The name extracted from the ID
    :param str version: The version extracted from the ID
    """

    def __init__(
        self,
        source_id: str,
        vault_url: str,
        name: str,
        version: "Optional[str]" = None,
    ) -> None:
        self.source_id = source_id
        self.vault_url = vault_url
        self.name = name
        self.version = version


def parse_key_vault_id(source_id: str) -> KeyVaultResourceId:
    try:
        parsed_uri = parse.urlparse(source_id)
    except Exception as exc:
        raise ValueError(f"'{source_id}' is not a valid ID") from exc
    if not (parsed_uri.scheme and parsed_uri.hostname):
        raise ValueError(f"'{source_id}' is not a valid ID")

    path = list(filter(None, parsed_uri.path.split("/")))

    if len(path) < 2 or len(path) > 3:
        raise ValueError(f"'{source_id}' is not a valid ID")

    vault_url = f"{parsed_uri.scheme}://{parsed_uri.hostname}"
    if parsed_uri.port:
        vault_url += f":{parsed_uri.port}"

    return KeyVaultResourceId(
        source_id=source_id,
        vault_url=vault_url,
        name=path[1],
        version=path[2] if len(path) == 3 else None,
    )


try:
    # pylint:disable=unused-import
    from .async_challenge_auth_policy import AsyncChallengeAuthPolicy
    from .async_client_base import AsyncKeyVaultClientBase

    __all__.extend(["AsyncChallengeAuthPolicy", "AsyncKeyVaultClientBase"])
except (SyntaxError, ImportError):
    pass
