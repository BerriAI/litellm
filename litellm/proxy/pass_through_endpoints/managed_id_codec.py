"""
Codec for LiteLLM passthrough-managed object IDs.

Plaintext format (before urlsafe-base64 encoding):
    litellm_proxy:passthrough;provider:{p};unified_id,{u};raw_id,{r}

Uses the same base64.urlsafe_b64encode / padding-restore convention as
``_is_base64_encoded_unified_file_id`` in
``openai_files_endpoints/common_utils.py``.

The ``passthrough;`` discriminator distinguishes these rows from
unified-endpoint rows that share the same LiteLLM_ManagedFileTable /
LiteLLM_ManagedObjectTable.  ``_resolve_one`` in the rewriter module rejects
any row whose decoded plaintext lacks this discriminator, making cross-system
replay safe.
"""

from __future__ import annotations

import base64
import uuid as _uuid_mod
from dataclasses import dataclass
from typing import Optional

from litellm.types.utils import SpecialEnums

_PREFIX = SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value  # "litellm_proxy"
_DISCRIMINATOR = "passthrough"


@dataclass(frozen=True)
class ManagedIdPayload:
    """Decoded contents of a passthrough managed ID."""

    provider: str
    unified_uuid: str
    raw_provider_id: str


def encode(provider: str, unified_uuid: str, raw_provider_id: str) -> str:
    """Return a urlsafe-base64 managed ID string (trailing ``=`` stripped)."""
    plaintext = SpecialEnums.LITELLM_PASSTHROUGH_MANAGED_ID_COMPLETE_STR.value.format(
        provider, unified_uuid, raw_provider_id
    )
    return base64.urlsafe_b64encode(plaintext.encode()).decode().rstrip("=")


def decode(managed_id: str) -> Optional[ManagedIdPayload]:
    """
    Decode *managed_id*.

    Returns ``None`` for anything that is not a passthrough managed ID — raw
    OpenAI IDs, unified-endpoint IDs, garbage, wrong types.  Never raises.
    """
    if not isinstance(managed_id, str):
        return None
    # Restore stripped padding before decoding
    padded = managed_id + "=" * (-len(managed_id) % 4)
    try:
        plaintext = base64.urlsafe_b64decode(padded).decode()
    except Exception:
        return None

    # Must start with "litellm_proxy:passthrough;"
    expected_head = f"{_PREFIX}:{_DISCRIMINATOR};"
    if not plaintext.startswith(expected_head):
        return None

    rest = plaintext[len(expected_head) :]
    try:
        # Split only on first two ';' so a raw_id containing ';' cannot
        # break parsing (OpenAI IDs don't use ';', but defensive).
        provider_part, rest2 = rest.split(";", 1)
        unified_part, raw_id_part = rest2.split(";", 1)
        if not (
            provider_part.startswith("provider:")
            and unified_part.startswith("unified_id,")
            and raw_id_part.startswith("raw_id,")
        ):
            return None
        return ManagedIdPayload(
            provider=provider_part[len("provider:") :],
            unified_uuid=unified_part[len("unified_id,") :],
            raw_provider_id=raw_id_part[len("raw_id,") :],
        )
    except Exception:
        return None


def is_managed(value: str) -> bool:
    """Return ``True`` iff *value* decodes to a passthrough managed ID."""
    return decode(value) is not None


def new_managed_id(provider: str, raw_provider_id: str) -> str:
    """Mint a fresh managed ID for a given raw provider ID."""
    return encode(provider, str(_uuid_mod.uuid4()), raw_provider_id)
