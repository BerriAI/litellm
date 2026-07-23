import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from litellm.proxy.management_helpers.password_link_share import (
    PasswordLinkError,
    PasswordLinkShare,
    create_password_link_secret,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self._status_code = status_code
        self._payload = payload

    @property
    def status_code(self) -> int:
        return self._status_code

    def json(self) -> object:
        return self._payload


def _decrypt_sjcl(passphrase: str, sjcl_json: str) -> str:
    payload = json.loads(sjcl_json)
    salt = base64.b64decode(payload["salt"])
    iv = base64.b64decode(payload["iv"])
    tag_bytes = payload["ts"] // 8
    ct_and_tag = base64.b64decode(payload["ct"])
    ciphertext, tag = ct_and_tag[:-tag_bytes], ct_and_tag[-tag_bytes:]
    derived = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=payload["iter"],
    ).derive(passphrase.encode("utf-8"))
    decryptor = Cipher(algorithms.AES(derived), modes.GCM(iv, tag, min_tag_length=tag_bytes)).decryptor()
    decryptor.authenticate_additional_data(b"")
    return (decryptor.update(ciphertext) + decryptor.finalize()).decode("utf-8")


@pytest.mark.asyncio
async def test_creates_decryptable_one_time_link() -> None:
    secret = "sk-super-secret-value-1234567890"
    captured: dict[str, object] = {}

    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return _FakeResponse(201, {"data": {"id": "abc123"}})

    result = await create_password_link_secret(
        secret=secret,
        api_key="private_key_test",
        poster=poster,
        expiration_hours=12,
        max_views=1,
    )

    assert isinstance(result, PasswordLinkShare)
    assert result.secret_id == "abc123"
    assert captured["url"] == "https://password.link/api/secrets"

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "ApiKey private_key_test"

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["expiration"] == 12
    assert body["max_views"] == 1

    assert result.share_link.startswith("https://password.link/?abc123#")

    public_part = base64.b64decode(result.share_link.split("#", 1)[1]).decode("utf-8")
    private_part = base64.b64decode(str(body["password_part_private"])).decode("utf-8")
    sjcl_json = base64.b64decode(str(body["ciphertext"])).decode("utf-8")

    assert _decrypt_sjcl(private_part + public_part, sjcl_json) == secret

    parsed = json.loads(sjcl_json)
    assert parsed["mode"] == "gcm"
    assert parsed["ks"] == 256
    assert parsed["iter"] == 10000
    assert parsed["ts"] == 128


@pytest.mark.asyncio
async def test_ciphertext_does_not_leak_plaintext() -> None:
    secret = "sk-leak-canary-value"
    captured: dict[str, object] = {}

    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        captured["body"] = body
        return _FakeResponse(201, {"data": {"id": "id1"}})

    await create_password_link_secret(secret=secret, api_key="k", poster=poster)

    body = captured["body"]
    assert isinstance(body, dict)
    assert secret not in json.dumps(body)


@pytest.mark.asyncio
async def test_link_uses_configured_api_base() -> None:
    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        return _FakeResponse(201, {"data": {"id": "xyz"}})

    result = await create_password_link_secret(
        secret="sk-value",
        api_key="k",
        poster=poster,
        api_base="https://vault.example.com/",
    )

    assert isinstance(result, PasswordLinkShare)
    assert result.share_link.startswith("https://vault.example.com/?xyz#")


@pytest.mark.asyncio
async def test_non_created_status_returns_error() -> None:
    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        return _FakeResponse(403, {"error": {"message": "invalid key"}})

    result = await create_password_link_secret(secret="sk-value", api_key="bad", poster=poster)

    assert isinstance(result, PasswordLinkError)
    assert "403" in result.message


@pytest.mark.asyncio
async def test_network_failure_returns_error() -> None:
    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    result = await create_password_link_secret(secret="sk-value", api_key="k", poster=poster)

    assert isinstance(result, PasswordLinkError)
    assert "connection refused" in result.message


@pytest.mark.asyncio
async def test_each_link_uses_fresh_password_parts() -> None:
    bodies: list[dict[str, object]] = []

    async def poster(url: str, headers: dict[str, str], body: dict[str, object]) -> _FakeResponse:
        bodies.append(body)
        return _FakeResponse(201, {"data": {"id": "id"}})

    await create_password_link_secret(secret="sk-value", api_key="k", poster=poster)
    await create_password_link_secret(secret="sk-value", api_key="k", poster=poster)

    assert bodies[0]["password_part_private"] != bodies[1]["password_part_private"]
    assert bodies[0]["ciphertext"] != bodies[1]["ciphertext"]
