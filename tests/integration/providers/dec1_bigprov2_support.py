"""Shared rig for the December 1 big provider contracts, round 2 (`test_dec1_bigprov2_*`).

Round 1 (`test_dec1_bedrock_*`, `test_dec1_vertex_*`, `test_dec1_gemini_*`, `test_dec1_azure_*`,
`test_dec1_anthropic_*`, `test_dec1_passthru_*`, `test_dec1_respapi_*`) listed behaviours it found and did
not write. This module holds what round 2 needs on top of the sibling helpers, which it imports and never
changes:

- A Google "workload identity federation" peer (`wif_peer`). One owned TCP peer plays four Google hosts at
  once, told apart by path: the subject token URL a `credential_source.url` names (`GET /subject`), the STS
  token exchange (`POST /sts/token`), the IAM credentials `generateAccessToken` call for service account
  impersonation, and the Vertex model host. `external_account` credential JSON built here points its
  `token_url` and `service_account_impersonation_url` at that peer, so google-auth never reaches
  `sts.googleapis.com` or `iamcredentials.googleapis.com`.
- Replies that hold the provider back (`withheld_json`, `stalled_stream`), for provider level timeouts.
  They gate on an `Event` nobody sets, so the peer only ever releases them when the wire server closes.

The STS exchange shape (form fields, `subject_token_type`, the AWS signed GetCallerIdentity subject token)
is google-auth's own wire format (`google/auth/external_account.py`, `google/auth/oauth2/sts.py`,
`google/auth/aws.py`). google-auth is not installed in the container these tests were written in, so those
literals come from the library's documented STS contract and are marked UNVERIFIED where it matters.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

SUBJECT_PATH: Final = "/subject"
STS_PATH: Final = "/sts/token"
IMPERSONATED_ACCOUNT: Final = "dec1-wif@dec1-wif-project.iam.gserviceaccount.com"
IMPERSONATION_PATH: Final = f"/v1/projects/-/serviceAccounts/{IMPERSONATED_ACCOUNT}:generateAccessToken"
AUDIENCE: Final = (
    "//iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/dec1-pool/providers/dec1-provider"
)
TOKEN_EXCHANGE_GRANT: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:access_token"
JWT_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:jwt"
AWS_TOKEN_TYPE: Final = "urn:ietf:params:aws:token-type:aws4_request"
CLOUD_PLATFORM: Final = "https://www.googleapis.com/auth/cloud-platform"
SUBJECT_TOKEN: Final = "dec1-oidc-subject-token"
AWS_ACCESS_KEY: Final = "AKIADEC1WIFSYNTHETIC"
AWS_SECRET_KEY: Final = "dec1-wif-synthetic-secret"
AWS_REGION: Final = "us-east-1"
PROJECT: Final = "dec1-wif-project"
LOCATION: Final = "us-central1"
GEMINI: Final = "gemini-2.5-flash"


def marker(prefix: str = "dec1-bigprov2") -> str:
    """A unique prompt suffix, so the shared proxy's response cache never answers a test."""
    return f"{prefix}-{uuid.uuid4().hex}"


def path_of(request: Request) -> str:
    return unquote(urlsplit(request.target).path)


def query_of(request: Request) -> dict[str, str]:
    return {name: values[-1] for name, values in parse_qs(urlsplit(request.target).query).items()}


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def form_of(request: Request) -> dict[str, str]:
    """An `application/x-www-form-urlencoded` body as a flat dict (first value wins)."""
    return {name: values[0] for name, values in parse_qs(request.body.decode()).items()}


def json_reply(payload: Mapping[str, JsonValue], *, status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def gemini_answer(text: str = "pong") -> Reply:
    return json_reply(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
            "modelVersion": GEMINI,
        }
    )


def models_path(model: str = GEMINI, *, project: str = PROJECT, location: str = LOCATION) -> str:
    return f"/v1/projects/{project}/locations/{location}/publishers/google/models/{model}"


# ---------------------------------------------------------------- the federation peer


@dataclass
class FederationPlan:
    """How the peer answers the subject, STS and impersonation calls. Tokens are numbered per kind."""

    subject: str = SUBJECT_TOKEN
    subject_json_field: str | None = None
    sts_error: Reply | None = None
    sts_issued: list[str] = field(default_factory=list)
    impersonated_issued: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def subject_reply(self) -> Reply:
        if self.subject_json_field is None:
            return Reply(body=self.subject.encode(), content_type="text/plain")
        return json_reply({self.subject_json_field: self.subject})

    def sts_reply(self) -> Reply:
        if self.sts_error is not None:
            return self.sts_error
        with self._lock:
            token: Final = f"sts-token-{len(self.sts_issued) + 1}"
            self.sts_issued.append(token)
        return json_reply(
            {"access_token": token, "issued_token_type": ACCESS_TOKEN_TYPE, "token_type": "Bearer", "expires_in": 3600}
        )

    def impersonation_reply(self) -> Reply:
        with self._lock:
            token: Final = f"impersonated-token-{len(self.impersonated_issued) + 1}"
            self.impersonated_issued.append(token)
        return json_reply({"accessToken": token, "expireTime": "2099-01-01T00:00:00Z"})


@dataclass(frozen=True, slots=True)
class FederationPeer:
    wire: Wire
    plan: FederationPlan
    seen: list[Request]

    @property
    def url(self) -> str:
        return self.wire.url.rstrip("/")

    def requests(self) -> tuple[Request, ...]:
        self.seen.extend(self.wire.drain())
        return tuple(self.seen)

    def subject_calls(self) -> tuple[Request, ...]:
        return tuple(request for request in self.requests() if path_of(request) == SUBJECT_PATH)

    def sts_calls(self) -> tuple[Request, ...]:
        return tuple(request for request in self.requests() if path_of(request) == STS_PATH)

    def impersonation_calls(self) -> tuple[Request, ...]:
        return tuple(request for request in self.requests() if path_of(request) == IMPERSONATION_PATH)

    def model_calls(self) -> tuple[Request, ...]:
        return tuple(
            request
            for request in self.requests()
            if request.method == "POST" and path_of(request) not in (STS_PATH, IMPERSONATION_PATH)
        )


@contextmanager
def wif_peer(
    plan: FederationPlan | None = None, respond: Callable[[Request], Reply] | None = None
) -> Iterator[FederationPeer]:
    """An owned peer that is the subject URL, the STS endpoint, IAM credentials and the Vertex host."""
    federation: Final = plan if plan is not None else FederationPlan()
    model: Final = respond if respond is not None else (lambda _request: gemini_answer())

    def answer(request: Request) -> Reply:
        path: Final = path_of(request)
        if path == SUBJECT_PATH:
            return federation.subject_reply()
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        if path == STS_PATH:
            return federation.sts_reply()
        if path == IMPERSONATION_PATH:
            return federation.impersonation_reply()
        return model(request)

    with wire_server(answer) as wire:
        yield FederationPeer(wire, federation, [])


def identity_pool_file(base: str, subject_file: str, **extra: JsonValue) -> dict[str, JsonValue]:
    """A file sourced `external_account` (OIDC token read from disk) whose STS endpoint is the peer."""
    return {
        "type": "external_account",
        "audience": AUDIENCE,
        "subject_token_type": JWT_TOKEN_TYPE,
        "token_url": f"{base}{STS_PATH}",
        "credential_source": {"file": subject_file},
        **extra,
    }


def identity_pool_url(base: str, *, json_field: str | None = None, **extra: JsonValue) -> dict[str, JsonValue]:
    """A URL sourced `external_account`: google-auth GETs the subject token from `<peer>/subject`."""
    source: Final[dict[str, JsonValue]] = {"url": f"{base}{SUBJECT_PATH}", "headers": {"Metadata-Flavor": "dec1"}}
    if json_field is not None:
        source["format"] = {"type": "json", "subject_token_field_name": json_field}
    return {
        "type": "external_account",
        "audience": AUDIENCE,
        "subject_token_type": JWT_TOKEN_TYPE,
        "token_url": f"{base}{STS_PATH}",
        "credential_source": source,
        **extra,
    }


def impersonation(base: str) -> dict[str, JsonValue]:
    return {"service_account_impersonation_url": f"{base}{IMPERSONATION_PATH}"}


def aws_external_account(base: str, **aws: JsonValue) -> dict[str, JsonValue]:
    """An AWS `external_account` with LiteLLM's explicit `aws_*` keys, which bypass the EC2 metadata service
    (`litellm/llms/vertex_ai/vertex_ai_aws_wif.py`). `aws` overrides or removes (value `None`) those keys."""
    keys: Final[dict[str, JsonValue]] = {
        "aws_access_key_id": AWS_ACCESS_KEY,
        "aws_secret_access_key": AWS_SECRET_KEY,
        "aws_region_name": AWS_REGION,
        **aws,
    }
    return {
        "type": "external_account",
        "audience": AUDIENCE,
        "subject_token_type": AWS_TOKEN_TYPE,
        "token_url": f"{base}{STS_PATH}",
        "credential_source": {
            "environment_id": "aws1",
            "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
        },
        **{name: value for name, value in keys.items() if value is not None},
    }


def aws_subject(sts_call: Request) -> dict[str, JsonValue]:
    """The AWS subject token google-auth sends: a URL quoted JSON description of a signed GetCallerIdentity."""
    return JSON_OBJECT.validate_json(unquote(form_of(sts_call)["subject_token"]))


def aws_subject_headers(sts_call: Request) -> dict[str, str]:
    headers: Final = aws_subject(sts_call)["headers"]
    assert isinstance(headers, list), headers
    pairs: Final = (JSON_OBJECT.validate_python(item) for item in headers)
    return {str(pair["key"]).lower(): str(pair["value"]) for pair in pairs}


def register_vertex(
    scenario: Scenario,
    peer: FederationPeer,
    credentials: Mapping[str, JsonValue],
    *,
    backend: str = f"vertex_ai/{GEMINI}",
    **params: JsonValue,
) -> str:
    """`backend` (by default `vertex_ai/gemini-2.5-flash`) at the peer, with these `vertex_credentials` sent
    as JSON text. `params` override `vertex_project` / `vertex_location` or add litellm params."""
    settings: Final[dict[str, JsonValue]] = {
        "api_key": None,
        "api_base": peer.url,
        "vertex_project": PROJECT,
        "vertex_location": LOCATION,
        "vertex_credentials": json.dumps(dict(credentials)),
        **params,
    }
    return scenario.model(model=backend, **settings)


def chat(gateway: Gateway, model: str, **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker()}], **fields},
    )


def bearer_of(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


# ---------------------------------------------------------------- held back replies


def withheld_json(payload: Mapping[str, JsonValue] | None = None) -> Reply:
    """Response headers go out at once and the JSON body never does (until the peer shuts down), so the
    proxy's read timeout is what ends the call."""
    never: Final = threading.Event()
    return Reply(
        content_type="application/json",
        chunks=(json.dumps(dict(payload or {})).encode(),),
        gate_before_first=never,
    )


def stalled_stream(chunks: tuple[bytes, ...], content_type: str = "text/event-stream") -> Reply:
    """The first chunk goes out and the stream then stalls with the connection open."""
    never: Final = threading.Event()
    return Reply(content_type=content_type, chunks=chunks, gate_after_first=never)


def silent_stream(chunks: tuple[bytes, ...], content_type: str = "text/event-stream") -> Reply:
    """Stream headers go out and no event ever follows."""
    never: Final = threading.Event()
    return Reply(content_type=content_type, chunks=chunks, gate_before_first=never)


def sse_data(text: str) -> list[dict[str, JsonValue]]:
    """Every JSON `data:` payload of an SSE body, without `[DONE]`."""
    lines: Final = (line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))
    return [JSON_OBJECT.validate_json(line) for line in lines if line.strip() != "[DONE]"]


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    payload: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = payload.get("error", payload)
    return JSON_OBJECT.validate_python(error) if isinstance(error, dict) else {"message": str(error)}


def error_message(response: httpx.Response) -> str:
    return str(error_of(response).get("message", ""))
