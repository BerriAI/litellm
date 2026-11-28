from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from nacl.exceptions import CryptoError
from pydantic import JsonValue, SecretStr, TypeAdapter
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import single_log_line
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with, encrypt_value_helper
from litellm.proxy.mcp_server.auth_profiles import InvalidAuthProfiles, parse_auth_profiles, profile_credentials_error
from litellm.types.mcp_server.auth_profile import (
    ClientCredentialsAuthProfile,
    MCPAuthProfile,
    StaticAuthProfile,
    profile_id_key,
)

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])


@dataclass(frozen=True, slots=True)
class MissingSigningKey:
    pass


@dataclass(frozen=True, slots=True)
class ProfileEncryptionFailed:
    pass


def _seal(value: str, signing_key: str) -> str:
    return encrypt_value_helper(value, new_encryption_key=signing_key)


def _secrets(profile: MCPAuthProfile) -> Mapping[str, SecretStr | None]:
    match profile:
        case ClientCredentialsAuthProfile():
            return {"client_id": profile.client_id, "client_secret": profile.client_secret}
        case StaticAuthProfile():
            return {"authentication_token": profile.authentication_token}
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(profile)  # pragma: no cover


def _can_inherit(profile: MCPAuthProfile, previous: MCPAuthProfile) -> bool:
    if isinstance(profile, ClientCredentialsAuthProfile) and isinstance(previous, ClientCredentialsAuthProfile):
        return "token_url" not in profile.model_fields_set or profile.token_url == previous.token_url
    return profile.auth_type == previous.auth_type


def _is_text(value: str) -> bool:
    try:
        value.encode()
    except UnicodeEncodeError:
        return False
    return True


def _has_unencodable_secret(profile: MCPAuthProfile) -> bool:
    return any(not _is_text(secret.get_secret_value()) for secret in _secrets(profile).values() if secret)


def _encrypted_secrets(
    profile: MCPAuthProfile, signing_key: str, encrypt: Callable[[str, str], str]
) -> Mapping[str, str]:
    return {
        field: encrypt(secret.get_secret_value(), signing_key) for field, secret in _secrets(profile).items() if secret
    }


def _readable_ciphertext(previous: MCPAuthProfile, signing_key: str) -> Mapping[str, str]:
    return {
        field: secret.get_secret_value()
        for field, secret in _secrets(previous).items()
        if secret and decrypt_if_encrypted_with(secret.get_secret_value(), signing_key) is not None
    }


def _stored_entry(
    profile: MCPAuthProfile, previous: MCPAuthProfile | None, ciphertext: Mapping[str, str], signing_key: str
) -> Mapping[str, JsonValue]:
    submitted: Final = _JSON_OBJECT.validate_python(profile.model_dump(mode="json"))
    if previous is None or not _can_inherit(profile, previous):
        return {**submitted, "id": profile.id if previous is None else previous.id, **ciphertext}
    return {
        **_JSON_OBJECT.validate_python(previous.model_dump(mode="json")),
        **submitted,
        "id": previous.id,
        **_readable_ciphertext(previous, signing_key),
        **ciphertext,
    }


def encrypt_auth_profiles(
    server: str,
    submitted: object,
    stored: object,
    *,
    signing_key: str,
    encrypt: Callable[[str, str], str] = _seal,
) -> tuple[Mapping[str, JsonValue], ...] | InvalidAuthProfiles | MissingSigningKey | ProfileEncryptionFailed:
    requested: Final = parse_auth_profiles(server, submitted)
    if isinstance(requested, InvalidAuthProfiles):
        return requested
    if requested and not signing_key:
        return MissingSigningKey()
    existing: Final = parse_auth_profiles(server, () if stored is None else stored)
    previous: Final = {} if isinstance(existing, InvalidAuthProfiles) else {profile_id_key(p.id): p for p in existing}
    for profile in requested:
        if _has_unencodable_secret(profile):
            return InvalidAuthProfiles(
                f"Profile '{profile.id}' of MCP server '{server}' has a credential that is not valid text"
            )
    try:
        ciphertexts: Final = tuple(_encrypted_secrets(profile, signing_key, encrypt) for profile in requested)
    except (CryptoError, OSError, ValueError) as exc:
        verbose_proxy_logger.warning(
            "MCP server %s: profile credentials were not encrypted: %s", single_log_line(server), type(exc).__name__
        )
        return ProfileEncryptionFailed()
    rows: Final = tuple(
        _stored_entry(profile, previous.get(profile_id_key(profile.id)), ciphertext, signing_key)
        for profile, ciphertext in zip(requested, ciphertexts, strict=True)
    )
    merged: Final = parse_auth_profiles(server, rows)
    if isinstance(merged, InvalidAuthProfiles):
        return merged
    return profile_credentials_error(server, merged) or rows


def _decrypted_secrets(profile: MCPAuthProfile, signing_key: str) -> Mapping[str, str | None]:
    return {
        field: decrypt_if_encrypted_with(secret.get_secret_value(), signing_key) if secret and signing_key else None
        for field, secret in _secrets(profile).items()
    }


def decrypt_auth_profiles(
    server: str, stored: object, *, signing_key: str
) -> tuple[MCPAuthProfile, ...] | InvalidAuthProfiles:
    profiles: Final = parse_auth_profiles(server, () if stored is None else stored)
    if isinstance(profiles, InvalidAuthProfiles):
        return profiles
    return parse_auth_profiles(
        server,
        tuple(
            {
                **_JSON_OBJECT.validate_python(profile.model_dump(mode="json")),
                **_decrypted_secrets(profile, signing_key),
            }
            for profile in profiles
        ),
    )
