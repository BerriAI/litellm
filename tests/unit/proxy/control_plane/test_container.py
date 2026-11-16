import os
from dataclasses import dataclass, field, replace
from typing import Final, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.control_plane.audit.log import build_audit_log
from litellm.proxy.control_plane.audit.pipeline import Delivered, Undelivered
from litellm.proxy.control_plane.audit.s3 import HttpxObjectStore, S3ObjectLockSink
from litellm.proxy.control_plane.container import (
    HandlerTransport,
    _environment_region,
    _shared_http_handler,
    build_audit_pipeline,
)
from litellm.proxy.control_plane.settings import ControlPlaneSettings

_JSON_COUNT: Final = TypeAdapter(dict[str, int])


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["ExpiredToken", "ExpiredTokenException", "RequestExpired", "TokenRefreshRequired"])
async def test_expired_s3_web_identity_credentials_are_refetched_without_flushing_other_sinks(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN", "oidc/env/CONTROL_PLANE_REFRESH_TEST_TOKEN")
    monkeypatch.setenv("CONTROL_PLANE_REFRESH_TEST_TOKEN", "test-token")
    monkeypatch.setenv("AWS_ROLE_NAME", "arn:aws:iam::123456789012:role/audit-refresh-test")
    monkeypatch.setenv("AWS_SESSION_NAME", code)
    sts: Final = MagicMock()
    sts.assume_role_with_web_identity.side_effect = (
        {
            "Credentials": {"AccessKeyId": key, "SecretAccessKey": "test-secret", "SessionToken": key},
            "PackedPolicySize": 1,
        }
        for key in ("expired-key", "other-sink-key", "fresh-key")
    )
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "01234567890123456789012345678901",
            "audit_sinks": ("s3",),
            "audit_s3_bucket": "audit",
        }
    )
    sink: Final = build_audit_pipeline(settings).workers[0].sink
    other: Final = build_audit_pipeline(settings).workers[0].sink
    assert isinstance(sink, S3ObjectLockSink) and isinstance(other, S3ObjectLockSink)
    sent: Final[list[httpx.Request]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(403, text=f"<Error><Code>{code}</Code></Error>")
        return httpx.Response(200)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(reply))
    audit: Final = build_audit_log()
    audit.record("operator", None, "control_plane", "PUT", "/roles/new", "allowed", 200)
    batch: Final = audit.recent_records(1)
    configured: Final = replace(sink, store=HttpxObjectStore(handler))
    try:
        with patch("boto3.client", return_value=sts):
            expired: Final = configured.sign("https://audit.s3.test/event", b"{}", {})
            isolated: Final = other.sign("https://audit.s3.test/event", b"{}", {})
            assert "expired-key/" in expired["Authorization"]
            assert "other-sink-key/" in isolated["Authorization"]
            assert isinstance(await configured.deliver(batch), Undelivered)
            assert await configured.deliver(batch) == Delivered(1)
            assert "other-sink-key/" in other.sign("https://audit.s3.test/event", b"{}", {})["Authorization"]
        assert len(sent) == 2 and sent[0].url == sent[1].url and sent[0].content == sent[1].content
        assert "expired-key/" in sent[0].headers["Authorization"]
        assert "fresh-key/" in sent[1].headers["Authorization"]
        assert sts.assume_role_with_web_identity.call_count == 3
    finally:
        await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock_mode", [None, "COMPLIANCE", "GOVERNANCE"])
@pytest.mark.parametrize(
    ("bucket", "prefix", "region", "resolved_region", "partition", "dns_suffix", "scoped_prefix"),
    [
        ("audit", "/cp/", "eu-west-1", "eu-west-1", "aws", "amazonaws.com", "cp/"),
        ("other-audit", "/separate/", "eu-west-1", "eu-west-1", "aws", "amazonaws.com", "separate/"),
        ("audit", "", None, "cn-north-1", "aws-cn", "amazonaws.com.cn", ""),
        ("audit", "/cp/sub/", "us-gov-west-1", "us-gov-west-1", "aws-us-gov", "amazonaws.com", "cp/sub/"),
    ],
)
async def test_s3_web_identity_signing_allows_uploads_and_object_retention(
    monkeypatch: pytest.MonkeyPatch,
    lock_mode: str | None,
    bucket: str,
    prefix: str,
    region: str | None,
    resolved_region: str,
    partition: str,
    dns_suffix: str,
    scoped_prefix: str,
) -> None:
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN", "oidc/env/CONTROL_PLANE_AUDIT_TEST_WEB_IDENTITY")
    monkeypatch.setenv("CONTROL_PLANE_AUDIT_TEST_WEB_IDENTITY", "audit-test-token")
    monkeypatch.setenv("AWS_ROLE_NAME", f"arn:{partition}:iam::123456789012:role/control-plane-audit")
    monkeypatch.setenv("AWS_SESSION_NAME", f"audit-{lock_mode}")
    monkeypatch.setenv("AWS_REGION", resolved_region)
    sts: Final = MagicMock()
    sts.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": "audit-access-key",
            "SecretAccessKey": "audit-secret-key",
            "SessionToken": "audit-session-token",
        },
        "PackedPolicySize": 1,
    }
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "01234567890123456789012345678901",
            "audit_sinks": ("s3",),
            "audit_s3_bucket": bucket,
            "audit_s3_prefix": prefix,
            "audit_s3_region": region,
            "audit_s3_object_lock_mode": lock_mode,
        }
    )
    pipeline: Final = build_audit_pipeline(settings)
    sink: Final = pipeline.workers[0].sink
    assert isinstance(sink, S3ObjectLockSink)
    with patch("boto3.client", return_value=sts) as client:
        signed: Final = sink.sign(f"https://{bucket}.s3.{resolved_region}.{dns_suffix}/event", b"{}", {})
    assert "audit-access-key/" in signed["Authorization"]
    assert f"/{resolved_region}/s3/aws4_request" in signed["Authorization"]
    assert signed["X-Amz-Security-Token"] == "audit-session-token"
    assert client.call_args.args == ("sts",)
    assert client.call_args.kwargs["region_name"] == resolved_region
    request: Final = sts.assume_role_with_web_identity.call_args.kwargs
    assert request["WebIdentityToken"] == "audit-test-token"
    assert request["RoleArn"] == os.environ["AWS_ROLE_NAME"]
    assert request["RoleSessionName"] == os.environ["AWS_SESSION_NAME"]
    policy: Final = TypeAdapter(dict[str, object]).validate_json(request["Policy"])
    statements: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_python(policy["Statement"])
    actions: Final = tuple(TypeAdapter(tuple[str, ...]).validate_python(item["Action"]) for item in statements)
    assert frozenset().union(*actions) == {
        "s3:PutObject",
        "s3:PutObjectRetention",
        "kms:GenerateDataKey",
        "kms:Decrypt",
    }
    bucket_arn: Final = f"arn:{partition}:s3:::{bucket}"
    objects: Final = f"{bucket_arn}/{scoped_prefix}*"
    assert tuple(item["Resource"] for item in statements) == (
        objects,
        f"arn:{partition}:kms:{resolved_region}:*:key/*",
    )
    # AWS KMS ViaService uses amazonaws.com in every partition, verified 2026-11-15.
    # https://docs.aws.amazon.com/kms/latest/developerguide/conditions-kms.html#conditions-kms-via-service
    assert tuple(item["Condition"] for item in statements) == (
        {"Bool": {"aws:SecureTransport": "true"}},
        {
            "Bool": {"aws:SecureTransport": "true"},
            "StringEquals": {"kms:ViaService": f"s3.{resolved_region}.amazonaws.com"},
            "StringLike": {"kms:EncryptionContext:aws:s3:arn": [objects, bucket_arn]},
        },
    )
    await pipeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("exporter", "endpoint", "remote"),
    [
        (None, None, False),
        ("console", None, False),
        ("in_memory", None, False),
        ("unknown", None, False),
        ("otlp_http", None, True),
        ("otlp_grpc", None, True),
        (None, "http://collector.test:4318", True),
    ],
)
async def test_otlp_auditing_refuses_local_exporter_fallbacks(
    monkeypatch: pytest.MonkeyPatch, strict: bool, exporter: str | None, endpoint: str | None, remote: bool
) -> None:
    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)
    if exporter is not None:
        monkeypatch.setenv("OTEL_EXPORTER", exporter)
    if endpoint is not None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
    settings: Final = ControlPlaneSettings.model_validate(
        {"LITELLM_MASTER_KEY": "01234567890123456789012345678901", "audit_sinks": ("otlp",), "audit_strict": strict}
    )
    if not remote:
        with pytest.raises(TypeError, match="requires an OTLP HTTP or gRPC log exporter"):
            build_audit_pipeline(settings)
        return
    pipeline: Final = build_audit_pipeline(settings)
    assert tuple(worker.sink.name for worker in pipeline.workers) == ("otlp",)
    await pipeline.close()


def test_shared_http_handler_does_not_follow_redirects() -> None:
    handler: Final = _shared_http_handler(
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "01234567890123456789012345678901"})
    )
    assert handler.follow_redirects is False


@dataclass
class _RecyclingHandler:  # mutable-ok: counts how often the pool is re-read
    """Stands in for AsyncHTTPHandler, whose client property hands out a fresh pool after cache eviction."""

    pools: list[httpx.AsyncClient] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        pool: Final = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"n": len(self.pools)}))
        )
        self.pools.append(pool)
        return pool


@pytest.mark.asyncio
async def test_transport_rereads_the_handlers_pool_on_every_request() -> None:
    handler: Final = _RecyclingHandler()
    transport: Final = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    async with transport.stream("GET", "https://dp.test/health", timeout=1.0) as first_response:
        first: Final = _JSON_COUNT.validate_json(await first_response.aread())
    async with transport.stream("GET", "https://dp.test/health", timeout=1.0) as second_response:
        second: Final = _JSON_COUNT.validate_json(await second_response.aread())
    assert (first, second) == ({"n": 1}, {"n": 2}), "a pool closed by eviction must not be reused"


@dataclass
class _RedirectingHandler:  # mutable-ok: counts the hops a following pool would take
    hops: list[str] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        def route(request: httpx.Request) -> httpx.Response:
            self.hops.append(str(request.url))
            if request.url.host == "dp.test":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:9/secret"})
            return httpx.Response(200, json={"leaked": True})

        return httpx.AsyncClient(transport=httpx.MockTransport(route), follow_redirects=True)


@pytest.mark.asyncio
async def test_transport_never_follows_a_dataplane_redirect_even_on_a_following_pool() -> None:
    handler: Final = _RedirectingHandler()
    transport: Final = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    async with transport.stream("GET", "https://dp.test/v1/mcp/server/oauth/x/authorize", timeout=1.0) as response:
        status: Final = response.status_code
    assert status == 302 and handler.hops == ["https://dp.test/v1/mcp/server/oauth/x/authorize"]


def test_the_s3_region_follows_the_aws_sdk_order_aws_region_then_default_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    assert _environment_region(BaseAWSLLM()) == "us-east-1"
    monkeypatch.delenv("AWS_REGION")
    assert _environment_region(BaseAWSLLM()) == "eu-west-1"
