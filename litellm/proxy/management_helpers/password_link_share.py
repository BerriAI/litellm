import base64
import json
import secrets
import string
from typing import Awaitable, Callable, Literal, Protocol, Union

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, ValidationError

_PASSWORD_PART_LENGTH = 18
_PBKDF2_ITERATIONS = 10000
_DERIVED_KEY_BYTES = 32
_GCM_IV_BYTES = 16
_GCM_SALT_BYTES = 16
_GCM_TAG_BYTES = 16
_KEY_SIZE_BITS = 256
_PART_ALPHABET = string.ascii_letters + string.digits
_DEFAULT_API_BASE = "https://password.link"
_CREATED_STATUS = 201


class PasswordLinkShare(BaseModel):
    status: Literal["ok"] = "ok"
    share_link: str
    secret_id: str


class PasswordLinkError(BaseModel):
    status: Literal["error"] = "error"
    message: str


PasswordLinkResult = Union[PasswordLinkShare, PasswordLinkError]


class _SecretData(BaseModel):
    id: str


class _CreateSecretResponse(BaseModel):
    data: _SecretData


class HttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> object: ...


Poster = Callable[[str, dict[str, str], dict[str, object]], Awaitable[HttpResponse]]


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _random_part() -> str:
    return "".join(secrets.choice(_PART_ALPHABET) for _ in range(_PASSWORD_PART_LENGTH))


def _sjcl_gcm_ciphertext(passphrase: str, plaintext: str) -> str:
    salt = secrets.token_bytes(_GCM_SALT_BYTES)
    iv = secrets.token_bytes(_GCM_IV_BYTES)
    derived = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_DERIVED_KEY_BYTES,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    ).derive(passphrase.encode("utf-8"))
    encryptor = Cipher(algorithms.AES(derived), modes.GCM(iv)).encryptor()
    encryptor.authenticate_additional_data(b"")
    body = encryptor.update(plaintext.encode("utf-8")) + encryptor.finalize()
    ciphertext = body + encryptor.tag[:_GCM_TAG_BYTES]
    payload = {
        "iv": _b64(iv),
        "v": 1,
        "iter": _PBKDF2_ITERATIONS,
        "ks": _KEY_SIZE_BITS,
        "ts": _GCM_TAG_BYTES * 8,
        "mode": "gcm",
        "adata": "",
        "cipher": "aes",
        "salt": _b64(salt),
        "ct": _b64(ciphertext),
    }
    return json.dumps(payload, separators=(",", ":"))


async def create_password_link_secret(
    *,
    secret: str,
    api_key: str,
    poster: Poster,
    api_base: str = _DEFAULT_API_BASE,
    expiration_hours: int = 24,
    max_views: int = 1,
) -> PasswordLinkResult:
    private_part = _random_part()
    public_part = _random_part()
    ciphertext = _b64(_sjcl_gcm_ciphertext(private_part + public_part, secret).encode("utf-8"))
    request_body: dict[str, object] = {
        "ciphertext": ciphertext,
        "password_part_private": _b64(private_part.encode("utf-8")),
        "expiration": expiration_hours,
        "max_views": max_views,
    }
    base = api_base.rstrip("/")
    headers = {"Authorization": f"ApiKey {api_key}", "Content-Type": "application/json"}
    try:
        response = await poster(f"{base}/api/secrets", headers, request_body)
    except httpx.HTTPError as exc:
        return PasswordLinkError(message=f"Failed to reach password.link: {exc}")

    if response.status_code != _CREATED_STATUS:
        return PasswordLinkError(message=f"password.link returned status {response.status_code}")

    try:
        parsed = _CreateSecretResponse.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        return PasswordLinkError(message=f"Unexpected password.link response: {exc}")

    public_b64 = _b64(public_part.encode("utf-8"))
    share_link = f"{base}/?{parsed.data.id}#{public_b64}"
    return PasswordLinkShare(share_link=share_link, secret_id=parsed.data.id)
