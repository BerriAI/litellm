import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote, urlsplit

import httpx
import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually, object_value
from integration._support.wire import Reply, Request
from pydantic import JsonValue
from redis import Redis

RECORDS: Final = "max_batch_file_records"
UPLOADS: Final = "max_batch_file_uploads_per_day"
DOWNLOADS: Final = "max_file_downloads_per_minute"
DAY_SECONDS: Final = 24 * 60 * 60
MINUTE_SECONDS: Final = 60
PROVIDER_KEY: Final = "integration-provider-key"
ROUTED_MODEL: Final = "batch-file-caps-routed"
PROVIDER_REJECTS: Final = "provider-rejects-this-upload"
MISSING_FILE: Final = "file-missing-"
IN_KEY: Final = "in this key's metadata"
IN_TEAM: Final = "in this team's metadata"
IN_GENERAL_SETTINGS: Final = "in general_settings"
THIS_KEY: Final = "this key"
JWT_KEY_ID: Final = "batch-file-caps-signing-key"
COMPLETION_TEXT: Final = "batch file caps completion"

_MARKER: Final = re.compile(rb"caps[0-9a-f]{32}")
_PURPOSE: Final = re.compile(rb'name="purpose"\r\n\r\n([A-Za-z_-]+)')
_CONTENT_PATH: Final = re.compile(r"/v1/files/(.+)/content")
_FILE_PATH: Final = re.compile(r"/v1/files/(.+)")


def marker() -> str:
    return "caps" + uuid.uuid4().hex


def batch_line(mark: str, index: int) -> bytes:
    return json.dumps(
        {
            "custom_id": f"{mark}-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "batch file caps"}]},
        }
    ).encode()


def batch_file(mark: str, records: int, separator: bytes = b"\n", ending: bytes = b"\n") -> bytes:
    return separator.join(batch_line(mark, index) for index in range(records)) + ending


def file_content(file_id: str) -> bytes:
    return json.dumps({"id": "batch_req_1", "custom_id": file_id, "response": {"status_code": 200}}).encode() + b"\n"


def file_object(file_id: str, purpose: str) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": 128,
        "created_at": 1700000000,
        "filename": "batch.jsonl",
        "purpose": purpose,
        "status": "processed",
    }


def _provider_error(status: int, message: str) -> Reply:
    return Reply(
        status=status,
        body=json.dumps(
            {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}
        ).encode(),
    )


def _uploaded(request: Request) -> Reply:
    if PROVIDER_REJECTS.encode() in request.body:
        return _provider_error(400, "The provider rejected this batch file.")
    mark: Final = _MARKER.search(request.body)
    purpose: Final = _PURPOSE.search(request.body)
    return Reply(
        body=json.dumps(
            file_object(
                f"file-{mark.group().decode()}" if mark is not None else f"file-{uuid.uuid4().hex}",
                purpose.group(1).decode() if purpose is not None else "batch",
            )
        ).encode()
    )


def _downloaded(file_id: str) -> Reply:
    if file_id.startswith(MISSING_FILE):
        return _provider_error(404, f"No such File object: {file_id}")
    return Reply(body=file_content(file_id), content_type="application/octet-stream")


def _completion() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": COMPLETION_TEXT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }
        ).encode()
    )


def provider(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method == "POST" and path == "/v1/files":
        return _uploaded(request)
    if request.method == "POST" and path == "/v1/chat/completions":
        return _completion()
    content: Final = _CONTENT_PATH.fullmatch(path)
    if request.method == "GET" and content is not None:
        return _downloaded(unquote(content.group(1)))
    described: Final = _FILE_PATH.fullmatch(path)
    if request.method == "GET" and described is not None:
        return Reply(body=json.dumps(file_object(unquote(described.group(1)), "batch")).encode())
    return _provider_error(404, f"No scripted reply for {request.method} {path}")


def seen(requests: tuple[Request, ...], mark: str) -> tuple[Request, ...]:
    return tuple(request for request in requests if mark in request.target or mark.encode() in request.body)


def uploads_seen(requests: tuple[Request, ...], mark: str) -> tuple[Request, ...]:
    return tuple(
        request for request in seen(requests, mark) if (request.method, request.target) == ("POST", "/v1/files")
    )


def downloads_seen(requests: tuple[Request, ...], file_id: str) -> tuple[Request, ...]:
    return tuple(
        request for request in requests if (request.method, request.target) == ("GET", f"/v1/files/{file_id}/content")
    )


def assert_forwarded_upload(request: Request, content: bytes) -> None:
    assert request.headers["authorization"] == f"Bearer {PROVIDER_KEY}", request.headers
    assert content in request.body, request.body
    assert b'name="purpose"\r\n\r\nbatch' in request.body, request.body


AUTH_CACHE_TTL_SECONDS: Final = 5


def caps_config(directory: Path, provider_url: str, general_settings: Mapping[str, JsonValue]) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = directory / f"batch_file_caps_{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [
                    {
                        "model_name": ROUTED_MODEL,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": PROVIDER_KEY,
                            "api_base": f"{provider_url}/v1",
                        },
                    }
                ],
                "general_settings": {
                    **base["general_settings"],
                    "user_api_key_cache_ttl": AUTH_CACHE_TTL_SECONDS,
                    **general_settings,
                },
                "files_settings": [
                    {"custom_llm_provider": "openai", "api_base": f"{provider_url}/v1", "api_key": PROVIDER_KEY}
                ],
            }
        )
    )
    return path


def provider_environment(provider_url: str) -> dict[str, str]:
    return {"OPENAI_BASE_URL": f"{provider_url}/v1", "OPENAI_API_KEY": PROVIDER_KEY}


def upload(
    candidate: Gateway,
    key: str,
    content: bytes,
    *,
    path: str = "/v1/files",
    purpose: str = "batch",
    filename: str = "batch.jsonl",
    fields: Mapping[str, str] = MappingProxyType({}),
) -> httpx.Response:
    return candidate.request_multipart(
        path, {"purpose": purpose, **fields}, {"file": (filename, content, "application/jsonl")}, key=key
    )


def download(
    candidate: Gateway,
    key: str,
    file_id: str,
    *,
    route: str = "/v1/files/{}/content",
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    return candidate.request("GET", route.format(file_id), key=key, params=params)


def window_end(window_seconds: int, room_seconds: int) -> float:
    started: Final = eventually(
        time.time,
        lambda now: window_seconds - now % window_seconds >= room_seconds,
        seconds=room_seconds + 5,
    )
    return (started // window_seconds + 1) * window_seconds


@dataclass(frozen=True, slots=True)
class Timed:
    response: httpx.Response
    before: float
    after: float


def timed(send: Callable[[], httpx.Response]) -> Timed:
    before: Final = time.time()
    response: Final = send()
    return Timed(response, before, time.time())


def _assert_rate_limited(observed: Timed, ends: float, what: str, held: str, reset: str) -> None:
    response: Final = observed.response
    assert response.status_code == 429, response.text
    assert observed.after < ends, f"The counted sequence ran past its window: {observed.after} >= {ends}"
    retry_after: Final = int(response.headers["retry-after"])
    earliest: Final = max(1, math.ceil(ends - observed.after))
    latest: Final = max(1, math.ceil(ends - observed.before))
    assert earliest <= retry_after <= latest, (retry_after, earliest, latest)
    assert response.json() == {
        "error": {
            "message": f"{what}: {held}. {reset.format(retry_after)}",
            "type": "rate_limit_error",
            "param": None,
            "code": "429",
        }
    }, response.text


def assert_upload_limited(observed: Timed, day_ends: float, limit: int, holder: str, source: str) -> None:
    _assert_rate_limited(
        observed,
        day_ends,
        "Batch file upload limit reached, the file was not forwarded to the provider",
        f"{UPLOADS} is {limit} for {holder} (set {source})",
        "The count resets at 00:00 UTC, in {} seconds.",
    )


def assert_download_limited(
    observed: Timed, minute_ends: float, file_id: str, limit: int, holder: str, source: str
) -> None:
    _assert_rate_limited(
        observed,
        minute_ends,
        f"Download limit reached for file {file_id}",
        f"{DOWNLOADS} is {limit} for {holder} (set {source})",
        "Retry in {} seconds.",
    )


def assert_too_many_records(response: httpx.Response, limit: int, source: str) -> None:
    assert response.status_code == 413, response.text
    assert response.json() == {
        "error": {
            "message": (
                f"Batch input file has more than {limit} records, which exceeds the {RECORDS} of {limit} "
                f"set {source}. The file was not forwarded to the provider."
            ),
            "type": "invalid_request_error",
            "param": "file",
            "code": "413",
        }
    }, response.text


@dataclass(frozen=True, slots=True)
class Counter:
    name: str
    count: int
    ttl: int


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def counters(cache: Redis, setting: str, holder: str) -> tuple[Counter, ...]:
    return tuple(
        Counter(name.decode(), int(cache.get(name) or 0), cache.ttl(name))
        for name in sorted(cache.scan_iter(match=f"*litellm:file_usage:{setting}:*{holder}*", count=1000))
    )


def config_entry(gateway: Gateway, setting: str) -> Mapping[str, JsonValue]:
    response: Final = gateway.request("GET", "/config/list", params={"config_type": "general_settings"})
    assert response.status_code == 200, response.text
    (entry,) = (object_value(field) for field in response.json() if field["field_name"] == setting)
    return entry


@dataclass(frozen=True, slots=True)
class Signer:
    private_key: rsa.RSAPrivateKey
    jwks: bytes


def signer() -> Signer:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())
    return Signer(private_key, json.dumps({"keys": [{**json.loads(public_jwk), "kid": JWT_KEY_ID}]}).encode())


def signed_token(identity: Signer, subject: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": subject, "iat": now, "exp": now + 300, "jti": uuid.uuid4().hex},
        identity.private_key,
        algorithm="RS256",
        headers={"kid": JWT_KEY_ID},
    )
