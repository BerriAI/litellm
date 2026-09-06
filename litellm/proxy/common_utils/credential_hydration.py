"""Shared helper for resolving a named Credential's values server-side.

Memory first (``litellm.credential_list``, already decrypted -- matching
``CredentialAccessor.get_credential_values``), then a DB decrypt fallback for a pod whose
in-memory list has not yet picked up a credential another pod just wrote or updated.
"""

import asyncio
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

import litellm
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.utils import PrismaClient
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.router_utils.clientside_credential_handler import clientside_credential_keys
from litellm.types.router import (
    GenericLiteLLMParams,
    server_owned_wif_fields_named,
    server_owned_wif_fields_present,
)
from litellm.types.utils import CredentialItem, LlmProviders, server_owned_wif_litellm_params

_LITELLM_PROVIDER_IDS: Final = frozenset(provider.value for provider in LlmProviders)

_FEDERATION_SURFACE_FIELDS: Final = frozenset(
    (
        *clientside_credential_keys,
        "configurable_clientside_auth_params",
        "litellm_credential_name",
        *server_owned_wif_litellm_params,
    )
)


def write_touches_federation_surface(incoming: Mapping[str, object] | None) -> bool:
    """Whether this write can move or re-scope the token a federated deployment mints.

    Three groups of fields can. The federation parameters choose which server-side secret is read
    and what the minted token is scoped to. ``litellm_credential_name`` resolves to those same
    parameters by reference. ``api_key``, ``api_base``, ``base_url``, and the
    ``configurable_clientside_auth_params`` that let a caller override them decide where the
    resulting token is sent. A write setting none of them leaves the federation configuration
    exactly as the proxy admin left it, so renaming a federated deployment or changing its rpm
    stays an ordinary team-admin edit.
    """
    return incoming is not None and not _FEDERATION_SURFACE_FIELDS.isdisjoint(incoming.keys())


def stored_credential_provider(credential_provider: object) -> str | None:
    """The dashboard stores its display casing (``Anthropic``) on credentials it creates, so the
    provider a credential names is the lowercased value when that is a litellm provider id."""
    if not isinstance(credential_provider, str):
        return None
    lowered: Final = credential_provider.lower()
    return lowered if lowered in _LITELLM_PROVIDER_IDS else None


def decrypted_or_stored(key: str, value: str) -> str:
    """The stored value decrypted, or as stored when it was never encrypted (a config.yaml value)."""
    decrypted: Final = decrypt_value_helper(value=value, key=key)
    return value if decrypted is None else decrypted


def _decrypted(db_credential: CredentialItem) -> CredentialItem:
    """The stored credential with every value decrypted, leaving already-plaintext values alone."""
    decrypted_values: Final = MappingProxyType(
        {key: decrypted_or_stored(key, value) for key, value in db_credential.credential_values.items()}
    )
    return CredentialItem(
        credential_name=db_credential.credential_name,
        credential_values=decrypted_values,  # pyright: ignore[reportArgumentType]  # declared dict[str, str], and pydantic copies this mapping into one on validation; LIT002 rules out building that dict here
        credential_info=db_credential.credential_info,
    )


async def hydrate_named_credential_authoritative(
    credential_name: str,
    prisma_client: PrismaClient | None,
) -> CredentialItem | None:
    """The stored credential, preferring the row over this pod's in-memory copy.

    ``hydrate_named_credential`` reads memory first, which is right when serving a request. A
    management operation cannot: on a pod whose in-memory copy predates another pod's update, it
    would export the superseded JWKS, or discover models against superseded values. Same reason
    ``named_credential_wif_fields`` reads both.
    """
    if prisma_client is None:
        return await hydrate_named_credential(credential_name, prisma_client)
    db_credential: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
    if db_credential is None:
        return await hydrate_named_credential(credential_name, prisma_client)
    return _decrypted(db_credential)


async def hydrate_named_credential(
    credential_name: str,
    prisma_client: PrismaClient | None,
) -> CredentialItem | None:
    for credential in litellm.credential_list:
        if credential.credential_name == credential_name:
            return credential
    if prisma_client is None:
        return None
    db_credential: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
    if db_credential is None:
        return None
    return _decrypted(db_credential)


async def named_credential_wif_fields(
    credential_name: str,
    prisma_client: PrismaClient | None,
) -> tuple[str, ...]:
    """Federation field names a write to ``credential_name`` would touch, from memory AND the row.

    Resolution reads memory first and stops there, which is right when serving a request. An
    authorization decision cannot: a pod whose in-memory copy predates an admin adding federation
    fields would see none and allow the write. This reads both and returns the union, so the gate
    refuses whenever either side says the credential is server-owned.
    """
    in_memory: Final = tuple(
        name
        for credential in litellm.credential_list
        if credential.credential_name == credential_name
        for name in server_owned_wif_fields_named(credential.credential_values)
    )
    if prisma_client is None:
        return in_memory
    db_credential: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
    stored: Final = () if db_credential is None else server_owned_wif_fields_named(db_credential.credential_values)
    return tuple(dict.fromkeys(in_memory + stored))


def submitted_litellm_params(params: GenericLiteLLMParams | None) -> Mapping[str, object] | None:
    """The fields a pydantic write actually set, as the mapping the federation gate reads.

    Only the set fields belong here: ``GenericLiteLLMParams`` declares every federation field, so
    the whole model would report every write as touching all of them.
    """
    if params is None:
        return None
    return MappingProxyType({name: getattr(params, name, None) for name in params.model_fields_set})


async def effective_server_owned_wif_fields(
    stored: Mapping[str, object] | None,
    incoming: Mapping[str, object] | None,
    prisma_client: PrismaClient | None,
) -> tuple[str, ...]:
    """Federation field names the deployment would carry AFTER this write.

    Authorization has to read the resulting deployment, not the submitted payload. A patch that
    names no federation field still lands on a deployment that has them, and a patch that only
    attaches ``litellm_credential_name`` inherits whatever that credential holds.

    The two sides are matched differently on purpose. ``stored`` is matched by VALUE, since it is
    a full deployment and a declared-but-unset field is not a federation field it carries.
    ``incoming`` holds only the keys the write actually set, so an explicit null still counts as
    touching the field.
    """
    from_stored: Final = () if stored is None else server_owned_wif_fields_present(stored)
    from_incoming: Final = () if incoming is None else server_owned_wif_fields_named(incoming.keys())
    from_credential: Final = tuple(
        chain.from_iterable(
            await asyncio.gather(
                *(
                    named_credential_wif_fields(credential_name, prisma_client)
                    for credential_name in _effective_credential_names(stored, incoming)
                )
            )
        )
    )
    return tuple(dict.fromkeys(from_stored + from_incoming + from_credential))


def _effective_credential_names(
    stored: Mapping[str, object] | None,
    incoming: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Both the credential the deployment already carries and the one this write names.

    Taking only the incoming name would let a write clear its way out: detaching a federated
    credential, by sending ``litellm_credential_name: null`` alongside an api_key or api_base of
    the caller's choosing, would leave nothing federated to find and the write would be allowed.
    Detaching an administrator's federated credential is itself an administrator's action, so the
    stored name counts whatever the write says.
    """
    from_stored: Final = None if stored is None else stored.get("litellm_credential_name")
    from_incoming: Final = None if incoming is None else incoming.get("litellm_credential_name")
    return tuple(dict.fromkeys(name for name in (from_stored, from_incoming) if isinstance(name, str)))
