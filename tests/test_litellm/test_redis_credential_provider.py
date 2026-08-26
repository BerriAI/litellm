import asyncio
import builtins
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm._redis_credential_provider import ElastiCacheIAMCredentialProvider


class _FakeCredentials:
    def __init__(self, access_key: str) -> None:
        self.access_key = access_key
        self.secret_key = "synthetic-secret"
        self.token = "synthetic-session-token"

    def get_frozen_credentials(self):
        return self


class _RotatingFakeCredentials:
    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def get_frozen_credentials(self):
        self.calls += 1
        return SimpleNamespace(
            access_key=f"AKIA-SYNTHETIC-{self.calls}",
            secret_key="synthetic-secret",
            token="synthetic-session-token",
        )


class _FakeResolver:
    def __init__(self, credentials: _FakeCredentials | _RotatingFakeCredentials | None) -> None:
        self.credentials = credentials
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.credentials


def test_elasticache_provider_signs_expected_query():
    resolver = _FakeResolver(_FakeCredentials("AKIA-SYNTHETIC"))
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=resolver,
    )

    user_name, token = provider.get_credentials()
    parsed = urlsplit("https://" + token)
    query = parse_qs(parsed.query)

    assert user_name == "iam-user"
    assert parsed.netloc == "cache.example.com"
    assert query["Action"] == ["connect"]
    assert query["User"] == ["iam-user"]
    assert query["X-Amz-Expires"] == ["900"]
    assert "elasticache" in query["X-Amz-Credential"][0]
    assert query["X-Amz-Credential"][0].split("/")[2] == "us-east-1"
    assert not token.startswith("https://")


def test_elasticache_provider_resolves_credentials_once_but_refreshes_signature():
    rotating_credentials = _RotatingFakeCredentials()
    resolver = _FakeResolver(rotating_credentials)
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=resolver,
    )

    first = provider.get_credentials()
    second = provider.get_credentials()
    async_result = asyncio.run(provider.get_credentials_async())

    assert first[0] == second[0] == async_result[0] == "iam-user"
    assert first[1] != second[1]
    assert async_result[1] != second[1]
    assert resolver.calls == 1
    assert rotating_credentials.calls == 3


def test_elasticache_provider_uses_botocore_session_credentials(monkeypatch):
    credentials = _FakeCredentials("AKIA-SYNTHETIC")
    monkeypatch.setattr("botocore.session.get_session", lambda: SimpleNamespace(get_credentials=lambda: credentials))
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
    )

    user_name, token = provider.get_credentials()

    assert user_name == "iam-user"
    assert "AKIA-SYNTHETIC" in token


def test_elasticache_provider_reports_missing_botocore(monkeypatch):
    original_import = builtins.__import__

    def import_without_botocore(name, *args, **kwargs):
        if name == "botocore.session":
            raise ImportError("synthetic missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "botocore.session", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_botocore)
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
    )

    with pytest.raises(ImportError, match="pip install boto3"):
        provider.get_credentials()


def test_elasticache_provider_reports_missing_credentials():
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=_FakeResolver(None),
    )

    with pytest.raises(RuntimeError, match="Unable to resolve AWS credentials"):
        provider.get_credentials()


def test_elasticache_provider_reports_missing_frozen_credentials():
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=_FakeResolver(SimpleNamespace(get_frozen_credentials=lambda: None)),
    )

    with pytest.raises(RuntimeError, match="Unable to resolve AWS credentials"):
        provider.get_credentials()


def test_elasticache_provider_reports_missing_signing_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_without_botocore_auth(name, *args, **kwargs):
        if name == "botocore.auth":
            raise ImportError("synthetic missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "botocore.auth", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_botocore_auth)
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=_FakeResolver(_FakeCredentials("AKIA-SYNTHETIC")),
    )

    with pytest.raises(ImportError, match="pip install boto3"):
        provider.get_credentials()


def test_elasticache_provider_recovers_after_a_failed_resolution():
    resolver = _FakeResolver(None)
    provider = ElastiCacheIAMCredentialProvider(
        user_name="iam-user",
        cache_name="cache.example.com",
        region="us-east-1",
        credentials_resolver=resolver,
    )

    with pytest.raises(RuntimeError, match="Unable to resolve AWS credentials"):
        provider.get_credentials()

    resolver.credentials = _FakeCredentials("AKIA-SYNTHETIC")
    user_name, token = provider.get_credentials()

    assert user_name == "iam-user"
    assert token
    assert resolver.calls == 2
