"""Unit tests for the S3/R2 cassette backend.

The backend is meant to replace the Redis cassette cache without
changing any test-side behavior, so these tests cover the operations
the persister actually exercises: get-with-TTL, put, NoSuchKey ⇒
cache miss, transient ClientError ⇒ raise (so the persister surfaces
it as a load/save failure), and selection precedence.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    VCRCassetteCacheWarning,
    _S3Backend,
    _select_remote_backend,
    cassette_cache_health,
    make_persister,
    redis_key_for,
    reset_cassette_cache_health,
)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class _FakeS3Client:
    """In-process stand-in for the boto3 S3 client.

    Stores objects with a fake ``LastModified`` so we can drive the
    backend's TTL check without sleeping. Mirrors the boto3 surface
    that ``_S3Backend`` actually depends on (``get_object``,
    ``put_object``).
    """

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.put_calls = 0
        self.get_calls = 0

    def put_object(self, *, Bucket, Key, Body):
        self.put_calls += 1
        if isinstance(Body, (bytes, bytearray)):
            data = bytes(Body)
        else:  # pragma: no cover - defensive
            data = Body.read()
        self.objects[(Bucket, Key)] = {
            "Body": data,
            "LastModified": _now(),
        }

    def get_object(self, *, Bucket, Key):
        self.get_calls += 1
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
                "GetObject",
            )
        return {
            "Body": BytesIO(obj["Body"]),
            "LastModified": obj["LastModified"],
        }


def _sample_cassette() -> dict:
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","messages":[]}',
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": b'{"id":"msg_1","type":"message"}'},
    }
    return {"requests": [request], "responses": [response]}


def test_s3_round_trip_via_persister():
    fake = _FakeS3Client()
    persister = make_persister(backend=_S3Backend(client=fake, bucket="cassettes"))
    cassette_id = "tests/llm_translation/test_x/test_s3_round_trip"

    persister.save_cassette(cassette_id, _sample_cassette(), yamlserializer)
    requests, responses = persister.load_cassette(cassette_id, yamlserializer)
    assert len(requests) == 1
    assert responses[0]["status"]["code"] == 200
    assert fake.put_calls == 1
    assert fake.get_calls == 1


def test_s3_no_such_key_is_treated_as_cassette_miss():
    fake = _FakeS3Client()
    persister = make_persister(backend=_S3Backend(client=fake, bucket="cassettes"))
    with pytest.raises(CassetteNotFoundError):
        persister.load_cassette("never/recorded", yamlserializer)


def test_s3_object_older_than_ttl_is_treated_as_miss():
    """If a bucket lifecycle policy hasn't reaped a stale object yet,
    the read-side age check still treats it as a cache miss so the
    test re-records — same semantics as the Redis TTL.
    """
    fake = _FakeS3Client()
    backend = _S3Backend(client=fake, bucket="cassettes")
    key = redis_key_for("tests/llm_translation/test_x/test_stale")
    fake.put_object(Bucket="cassettes", Key=key, Body=b"fake_payload")
    fake.objects[("cassettes", key)]["LastModified"] = _now() - dt.timedelta(
        seconds=CASSETTE_TTL_SECONDS + 60
    )

    assert backend.get(key) is None


def test_s3_5xx_propagates_as_save_failure_warning():
    """Unexpected ClientErrors must surface to the persister so the
    health counters tick and the session-end banner shows the failure.
    """

    class _BoomClient(_FakeS3Client):
        def put_object(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "boom"}},
                "PutObject",
            )

    reset_cassette_cache_health()
    persister = make_persister(
        backend=_S3Backend(client=_BoomClient(), bucket="cassettes")
    )
    cassette_id = "tests/llm_translation/test_x/test_5xx"

    with pytest.warns(VCRCassetteCacheWarning, match="InternalError"):
        persister.save_cassette(cassette_id, _sample_cassette(), yamlserializer)
    assert cassette_cache_health()["save_failures"] == 1


def test_select_remote_backend_prefers_s3_when_both_envs_set(monkeypatch):
    """If a maintainer accidentally configures both env vars during a
    Redis→R2 migration, S3 wins. The reverse precedence would silently
    keep paying for Redis bandwidth even after the bucket is wired up.
    """
    monkeypatch.setenv("CASSETTE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CASSETTE_S3_BUCKET", "test-bucket")
    monkeypatch.setenv(
        "CASSETTE_S3_ENDPOINT", "https://example.r2.cloudflarestorage.com"
    )
    backend = _select_remote_backend()
    assert backend.name == "s3"


def test_select_remote_backend_falls_back_to_redis(monkeypatch):
    monkeypatch.delenv("CASSETTE_S3_BUCKET", raising=False)
    monkeypatch.setenv("CASSETTE_REDIS_URL", "redis://localhost:6379/0")
    backend = _select_remote_backend()
    assert backend.name == "redis"


def test_select_remote_backend_raises_when_neither_configured(monkeypatch):
    monkeypatch.delenv("CASSETTE_REDIS_URL", raising=False)
    monkeypatch.delenv("CASSETTE_S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="No cassette backend configured"):
        _select_remote_backend()
