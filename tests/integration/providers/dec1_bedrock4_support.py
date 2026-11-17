"""Shared rig for the fourth round of December 1 AWS Bedrock contracts (`test_dec1_bedrock4_*`).

Round 4 covers what rounds 1 to 3 listed as not covered: Bedrock image generation for every config
class (Titan, Nova Canvas task types, Stability 1, Stability 3 and Ultra), Bedrock image edit (Stability
edit models and Nova Canvas edit), STS credential chains on an owned proxy (AssumeRole with an external
id and session tags, AssumeRoleWithWebIdentity), InvokeAgent trace handling beyond `preProcessingTrace`,
and the knowledge base `retrievalConfiguration` merge.

Every helper points the proxy at an owned TCP peer (`wire_server`): deployments carry `api_base` or
`aws_bedrock_runtime_endpoint`, and the STS tests run an owned proxy whose `AWS_ENDPOINT_URL_STS` and
deployment `aws_sts_endpoint` both name the peer, so no request can reach AWS. The round 1 and round 3
helpers (`dec1_bedrock_support.py`, `dec1_bedrock3_support.py`) are imported unchanged.
"""

from __future__ import annotations

import base64
import json
import struct
import uuid
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qs, unquote

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_bedrock_support import TOKEN, register_model
from pydantic import JsonValue

GENERATIONS: Final = "/v1/images/generations"
EDITS: Final = "/v1/images/edits"

TITAN_IMAGE_V2: Final = "amazon.titan-image-generator-v2:0"
NOVA_CANVAS: Final = "amazon.nova-canvas-v1:0"
"""Has `supports_nova_canvas_image_edit: true` in `model_prices_and_context_window.json:552`."""
SD3_LARGE: Final = "stability.sd3-5-large-v1:0"
STABLE_IMAGE_ULTRA: Final = "stability.stable-image-ultra-v1:1"
SDXL_V1: Final = "stability.stable-diffusion-xl-v1"
INPAINT: Final = "stability.stable-image-inpaint-v1:0"
OUTPAINT: Final = "stability.stable-outpaint-v1:0"
STYLE_TRANSFER: Final = "stability.stable-style-transfer-v1:0"
REMOVE_BACKGROUND: Final = "stability.stable-image-remove-background-v1:0"
CONTROL_SKETCH: Final = "stability.stable-image-control-sketch-v1:0"

IMAGE_ONE: Final = "Zmlyc3QtaW1hZ2U="
"""base64 of `first-image`, the bytes a scripted Bedrock image model returns."""
IMAGE_TWO: Final = "c2Vjb25kLWltYWdl"
"""base64 of `second-image`."""

STS_ACCESS_KEY: Final = "AKIADEC1BEDROCK4SRC1"
STS_SECRET_KEY: Final = "synthetic-source-secret-key-dec1-bedrock4"
ASSUMED_ACCESS_KEY: Final = "ASIADEC1BEDROCK4ROLE"
ASSUMED_TOKEN: Final = "synthetic-assumed-session-token-dec1-bedrock4"
WEB_IDENTITY_ACCESS_KEY: Final = "ASIADEC1BEDROCK4WEBI"
WEB_IDENTITY_TOKEN_ENV: Final = "DEC1_BEDROCK4_OIDC_TOKEN"
WEB_IDENTITY_TOKEN: Final = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJkZWMxLWJlZHJvY2s0In0."
"""An unsigned synthetic JWT; the STS peer never validates it."""
CONVERSE_MODEL: Final = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
CONVERSE_TARGET: Final = "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"


def prompt(prefix: str = "dec1-bedrock4") -> str:
    """A fresh prompt, so the shared proxy's response cache can never answer a test."""
    return f"{prefix} {uuid.uuid4().hex}"


def json_body(request: Request) -> dict[str, JsonValue]:
    decoded: Final = json.loads(request.body)
    assert isinstance(decoded, dict), request.body
    return decoded


def json_reply(payload: Mapping[str, JsonValue], *, status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def path_of(request: Request) -> str:
    """The decoded request path without its query string (`%3A` reads back as `:`)."""
    return unquote(request.target.split("?", 1)[0])


def b64(content: bytes) -> str:
    return base64.b64encode(content).decode()


def png(rgba: tuple[int, int, int, int]) -> bytes:
    """A valid 1x1 RGBA PNG, unique per colour, so each upload has distinct bytes."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header: Final = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels: Final = zlib.compress(b"\x00" + bytes(rgba))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def error_text(response: httpx.Response) -> str:
    payload: Final = response.json()
    error: Final = payload.get("error", payload)
    return str(error.get("message", "")) if isinstance(error, dict) else str(error)


@dataclass(frozen=True, slots=True)
class ImageRig:
    """One owned Bedrock image peer plus the model group name that routes to it."""

    gateway: Gateway
    scenario: Scenario
    wire: Wire
    model: str

    def posts(self) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if request.method == "POST")

    def one_post(self) -> Request:
        posts: Final = self.posts()
        assert len(posts) == 1, [(request.method, request.target) for request in posts]
        return posts[0]

    def generate(self, text: str, **fields: JsonValue) -> httpx.Response:
        return self.gateway.request("POST", GENERATIONS, {"model": self.model, "prompt": text, **fields})

    def edit(
        self,
        fields: Mapping[str, str],
        files: Sequence[tuple[str, tuple[str, bytes, str]]],
    ) -> httpx.Response:
        """A multipart `POST /v1/images/edits` with `model` set to this rig's group."""
        return self.gateway.client.post(
            EDITS,
            data={"model": self.model, **fields},
            files=list(files),
            headers={"Authorization": f"Bearer {self.gateway.key}"},
        )


def _posts_only(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        return respond(request) if request.method == "POST" else Reply(body=b"{}")

    return answer


@contextmanager
def image_model(
    gateway: Gateway, respond: Callable[[Request], Reply], *, backend: str, **params: JsonValue
) -> Iterator[ImageRig]:
    """Register `bedrock/<backend>` with `api_base` at a new owned peer and a Bedrock API key (bearer).

    Bearer auth keeps the image tests independent of which `aws_*` keys reach the image handler's
    `optional_params`; SigV4 on image routes is left to the chat suites. `None` drops a parameter.
    """
    with wire_server(_posts_only(respond)) as wire, gateway.scenario() as scenario:
        merged: Final[dict[str, JsonValue]] = {
            "api_base": wire.url,
            "api_key": TOKEN,
            "aws_region_name": "us-east-1",
            **params,
        }
        cleaned: Final = {key: value for key, value in merged.items() if value is not None}
        name: Final = register_model(scenario, model=f"bedrock/{backend}", **cleaned)
        yield ImageRig(gateway, scenario, wire, name)


# ------------------------------------------------------------------ STS


def sts_action(request: Request) -> dict[str, list[str]]:
    """The form fields of one STS query API call (`Action=AssumeRole&RoleArn=...`)."""
    return parse_qs(request.body.decode())


def _credentials_xml(access_key: str, token: str) -> str:
    return (
        f"<Credentials><AccessKeyId>{access_key}</AccessKeyId>"
        "<SecretAccessKey>synthetic-assumed-secret-key-dec1-bedrock4</SecretAccessKey>"
        f"<SessionToken>{token}</SessionToken><Expiration>2035-01-01T00:00:00Z</Expiration></Credentials>"
    )


def sts_reply(action: str, *, access_key: str = ASSUMED_ACCESS_KEY, token: str = ASSUMED_TOKEN) -> Reply:
    """A successful STS XML answer for `GetCallerIdentity`, `AssumeRole` or `AssumeRoleWithWebIdentity`.

    `GetCallerIdentity` names an IAM user, never a role, so `_is_already_running_as_role` is false and
    the proxy goes on to assume the role (`litellm/llms/bedrock/base_aws_llm.py:1244`).
    """
    if action == "GetCallerIdentity":
        result = (
            "<GetCallerIdentityResult><Arn>arn:aws:iam::123456789012:user/dec1-source</Arn>"
            "<UserId>dec1-source</UserId><Account>123456789012</Account></GetCallerIdentityResult>"
        )
    elif action == "AssumeRoleWithWebIdentity":
        result = (
            f"<AssumeRoleWithWebIdentityResult>{_credentials_xml(access_key, token)}"
            "<SubjectFromWebIdentityToken>dec1-bedrock4</SubjectFromWebIdentityToken>"
            "<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/dec1/web</Arn>"
            "<AssumedRoleId>dec1:web</AssumedRoleId></AssumedRoleUser></AssumeRoleWithWebIdentityResult>"
        )
    else:
        result = (
            f"<AssumeRoleResult>{_credentials_xml(access_key, token)}"
            "<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/dec1/session</Arn>"
            "<AssumedRoleId>dec1:session</AssumedRoleId></AssumedRoleUser>"
            "<PackedPolicySize>0</PackedPolicySize></AssumeRoleResult>"
        )
    return Reply(
        content_type="text/xml",
        body=(
            f'<{action}Response xmlns="https://sts.amazonaws.com/doc/2011-06-15/">{result}'
            f"<ResponseMetadata><RequestId>dec1-bedrock4-sts</RequestId></ResponseMetadata></{action}Response>"
        ).encode(),
    )


def role_arn() -> str:
    """A fresh role ARN per test, so the proxy's credential cache never serves another test's session."""
    return f"arn:aws:iam::123456789012:role/dec1-bedrock4-{uuid.uuid4().hex[:12]}"


def calls_for(requests: Sequence[Request], role: str, action: str) -> tuple[dict[str, list[str]], ...]:
    """The STS calls of one `action` that name `role` as `RoleArn`."""
    return tuple(
        fields
        for fields in (sts_action(request) for request in requests)
        if fields.get("Action") == [action] and fields.get("RoleArn") == [role]
    )
