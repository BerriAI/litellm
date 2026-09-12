import hashlib
import hmac
import os
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Final

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.client import BaseClient
from botocore.credentials import AssumeRoleCredentialFetcher, Credentials, DeferredRefreshableCredentials
from botocore.session import get_session

TEST_ACCESS_KEY: Final = "e2e-bedrock-replay"
TEST_SECRET_KEY: Final = "e2e-bedrock-replay-secret"
_MOUNT: Final = re.compile(r"bedrock-(us-(?:east|west)-[12])")
_AUTH: Final = re.compile(
    r"AWS4-HMAC-SHA256 Credential=([^/]+)/([0-9]{8})/([^/]+)/bedrock/aws4_request, "
    r"SignedHeaders=([a-z0-9;-]+), Signature=([0-9a-f]{64})"
)


def bedrock_region(mount: str) -> str | None:
    match: Final = _MOUNT.fullmatch(mount)
    return match.group(1) if match else None


def bedrock_upstream(mount: str) -> str | None:
    region: Final = bedrock_region(mount)
    return f"https://bedrock-runtime.{region}.amazonaws.com" if region else None


def valid_bedrock_signature(
    method: str, raw_path: str, headers: Mapping[str, str], body: bytes | None, region: str
) -> bool:
    normalized: Final = {key.lower(): value for key, value in headers.items()}
    auth: Final = _AUTH.fullmatch(normalized.get("authorization", ""))
    if auth is None or auth.group(1) != TEST_ACCESS_KEY or auth.group(3) != region:
        return False
    timestamp: Final = normalized.get("x-amz-date", "")
    signed: Final = frozenset(auth.group(4).split(";"))
    if not {"host", "x-amz-date"} <= signed or not signed <= normalized.keys() or auth.group(2) != timestamp[:8]:
        return False
    payload_hash: Final = normalized.get("x-amz-content-sha256")
    if payload_hash is not None and payload_hash != hashlib.sha256(body or b"").hexdigest():
        return False
    request: Final = AWSRequest(
        method=method,
        url=f"http://{normalized['host']}{raw_path}",
        data=body,
        headers={key: normalized[key] for key in signed},
    )
    request.context["timestamp"] = timestamp
    signer: Final = SigV4Auth(Credentials(TEST_ACCESS_KEY, TEST_SECRET_KEY), "bedrock", region)
    canonical: Final = signer.canonical_request(request)
    expected: Final = signer.signature(signer.string_to_sign(request, canonical), request)
    return hmac.compare_digest(expected, auth.group(5))


def recording_credentials_from(environ: Mapping[str, str], client_creator: Callable[..., BaseClient]) -> Credentials:
    access_key: Final = environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key: Final = environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise ValueError("Bedrock recording requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the runner")
    source: Final = Credentials(access_key, secret_key, environ.get("AWS_SESSION_TOKEN") or None)
    role: Final = environ.get("AWS_ROLE_NAME")
    if not role:
        return source
    extra_args: Final = {
        key: value
        for key, value in (
            ("RoleSessionName", environ.get("AWS_SESSION_NAME", "e2e-bedrock-record")),
            ("ExternalId", environ.get("AWS_EXTERNAL_ID")),
        )
        if value
    }
    fetcher: Final = AssumeRoleCredentialFetcher(
        client_creator=client_creator,
        source_credentials=source,
        role_arn=role,
        extra_args=extra_args,
    )
    return DeferredRefreshableCredentials(refresh_using=fetcher.fetch_credentials, method="assume-role")


@lru_cache(maxsize=1)
def recording_credentials() -> Credentials:
    return recording_credentials_from(os.environ, get_session().create_client)


def sign_bedrock_forward(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, region: str, credentials: Credentials
) -> dict[str, str]:
    forwarded: Final = {
        key: value
        for key, value in headers.items()
        if key.lower() not in {"authorization", "host", "content-length", "connection", "accept-encoding"}
        and not key.lower().startswith("x-amz-")
    }
    request: Final = AWSRequest(method=method, url=url, data=body, headers=forwarded)
    SigV4Auth(credentials.get_frozen_credentials(), "bedrock", region).add_auth(request)
    return dict(request.headers.items())
