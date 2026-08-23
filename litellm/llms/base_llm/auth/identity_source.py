"""Tagged-union identity-source configs for Anthropic workload identity federation, beyond the
existing token_file/env resolver in ``litellm/llms/anthropic/wif.py``.

Each variant only ever carries secret *pointer names* (``signing_key_ref``, ``client_secret_ref``),
never a resolved secret value, so ``identity_source_ref`` can safely hash a variant into the short,
content-derived ``oidc/<kind>/<hash>`` string used elsewhere as a get_secret ref, a token-exchange
cache-key discriminator, and an operator-facing error pointer.
"""

import hashlib
from enum import Enum
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_REF_HASH_HEX_LENGTH: Final = 16
_MAX_TTL_SECONDS: Final = 3600
_DEFAULT_TTL_SECONDS: Final = 300


class AnthropicIdentitySourceKind(str, Enum):
    internal_issuer = "internal_issuer"
    keycloak = "keycloak"


class InternalIssuerSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    kind: Literal[AnthropicIdentitySourceKind.internal_issuer] = AnthropicIdentitySourceKind.internal_issuer
    issuer_url: str
    subject: str
    audience: str | None = None
    ttl_seconds: Annotated[int, Field(gt=0, le=_MAX_TTL_SECONDS)] = _DEFAULT_TTL_SECONDS
    signing_key_ref: str


class KeycloakSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    kind: Literal[AnthropicIdentitySourceKind.keycloak] = AnthropicIdentitySourceKind.keycloak
    token_url: str
    client_id: str
    auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic"
    client_secret_ref: str
    scope: str | None = None


AnthropicIdentitySourceConfig: TypeAlias = Annotated[InternalIssuerSource | KeycloakSource, Field(discriminator="kind")]
identity_source_config_adapter: Final = TypeAdapter[AnthropicIdentitySourceConfig](AnthropicIdentitySourceConfig)


def identity_source_ref(config: AnthropicIdentitySourceConfig) -> str:
    """``oidc/<kind>/<hash>``: a short, secret-free pointer, stable for identical config and rolling
    whenever any field does, including a ``*_ref`` pointer NAME (never the secret it points to)."""
    digest: Final = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:_REF_HASH_HEX_LENGTH]
    return f"oidc/{config.kind.value}/{digest}"
