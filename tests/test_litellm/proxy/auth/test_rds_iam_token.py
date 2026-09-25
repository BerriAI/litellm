import sys
import types
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.auth.rds_iam_token import (
    generate_iam_auth_token,  # pyright: ignore[reportUnknownVariableType]  # source signature params are untyped upstream
)

FAKE_DB_AUTH_TOKEN: Final = "db:5432/?Action=connect&X-Amz-Credential=a/b"

AWS_ENV_VARS: Final = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SESSION_NAME",
    "AWS_REGION_NAME",
    "AWS_REGION",
    "AWS_PROFILE_NAME",
    "AWS_ROLE_NAME",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_RDS_IAM_IGNORE_WEB_IDENTITY_TOKEN",
)


@dataclass(slots=True)
class _CallRecord:
    client_calls: list[tuple[str | None, dict[str, object]]] = field(default_factory=list)
    sts_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)


class _FakeStsClient:
    def __init__(self, record: _CallRecord) -> None:
        self._record = record

    def assume_role(self, **kwargs: object) -> dict[str, object]:
        self._record.sts_calls.append(("assume_role", kwargs))
        return {
            "Credentials": {
                "AccessKeyId": "ASIA-ASSUMED",
                "SecretAccessKey": "assumed-secret",
                "SessionToken": "assumed-session",
            }
        }

    def assume_role_with_web_identity(self, **kwargs: object) -> dict[str, object]:
        self._record.sts_calls.append(("assume_role_with_web_identity", kwargs))
        return {
            "Credentials": {
                "AccessKeyId": "ASIA-WEB-IDENTITY",
                "SecretAccessKey": "web-identity-secret",
                "SessionToken": "web-identity-session",
            }
        }


class _FakeRdsClient:
    def generate_db_auth_token(self, DBHostname: str, Port: int, DBUsername: str) -> str:
        return FAKE_DB_AUTH_TOKEN


class _FakeSessionModule(types.ModuleType):
    def Config(self, *args: object, **kwargs: object) -> object:
        return object()


class _FakeBoto3Module(types.ModuleType):
    def __init__(self, record: _CallRecord) -> None:
        super().__init__("boto3")
        self._record = record
        self._sts_client = _FakeStsClient(record)
        self.session = _FakeSessionModule("boto3.session")

    def client(self, *args: str, **kwargs: object) -> _FakeStsClient | _FakeRdsClient:
        service_name = kwargs.get("service_name", args[0] if args else None)
        assert isinstance(service_name, str) or service_name is None
        self._record.client_calls.append((service_name, kwargs))
        if service_name == "sts":
            return self._sts_client
        return _FakeRdsClient()

    def Session(self, *args: object, **kwargs: object) -> _FakeRdsClient:
        return _FakeRdsClient()


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in AWS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    return monkeypatch


@pytest.fixture
def boto3_stub(monkeypatch: pytest.MonkeyPatch) -> _CallRecord:
    record = _CallRecord()
    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3Module(record))
    return record


def _set_irsa_pod_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    token_file = tmp_path / "web_identity_token"
    token_file.write_text("oidc-token-contents")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:pod")
    monkeypatch.setenv("AWS_ROLE_NAME", "arn:rds")
    monkeypatch.setenv("AWS_SESSION_NAME", "sess")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-EXPLICIT")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "explicit-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "explicit-session")
    return "oidc-token-contents"


@pytest.mark.parametrize("flag_value", ["true", "1", "TRUE"])
def test_ignore_web_identity_token_uses_assume_role_with_session_token(
    aws_env: pytest.MonkeyPatch, boto3_stub: _CallRecord, tmp_path: Path, flag_value: str
) -> None:
    _set_irsa_pod_env(aws_env, tmp_path)
    aws_env.setenv("AWS_RDS_IAM_IGNORE_WEB_IDENTITY_TOKEN", flag_value)

    token = generate_iam_auth_token(db_host="db", db_port=5432, db_user="litellm")

    sts_methods = [name for name, _ in boto3_stub.sts_calls]
    assert sts_methods == ["assume_role"]
    assert boto3_stub.sts_calls[0][1]["RoleArn"] == "arn:rds"
    assert boto3_stub.sts_calls[0][1]["RoleSessionName"] == "sess"

    sts_client_calls = [kw for svc, kw in boto3_stub.client_calls if svc == "sts"]
    assert len(sts_client_calls) == 1
    assert sts_client_calls[0]["aws_access_key_id"] == "AKIA-EXPLICIT"
    assert sts_client_calls[0]["aws_secret_access_key"] == "explicit-secret"
    assert sts_client_calls[0]["aws_session_token"] == "explicit-session"

    rds_client_calls = [kw for svc, kw in boto3_stub.client_calls if svc == "rds"]
    assert len(rds_client_calls) == 1
    assert rds_client_calls[0]["aws_access_key_id"] == "ASIA-ASSUMED"
    assert rds_client_calls[0]["aws_secret_access_key"] == "assumed-secret"
    assert rds_client_calls[0]["aws_session_token"] == "assumed-session"

    assert token == urllib.parse.quote(FAKE_DB_AUTH_TOKEN, safe="")


def test_ignore_web_identity_token_explicit_keys_forward_session_token(
    aws_env: pytest.MonkeyPatch, boto3_stub: _CallRecord
) -> None:
    aws_env.setenv("AWS_RDS_IAM_IGNORE_WEB_IDENTITY_TOKEN", "true")
    aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIA-EXPLICIT")
    aws_env.setenv("AWS_SECRET_ACCESS_KEY", "explicit-secret")
    aws_env.setenv("AWS_SESSION_TOKEN", "explicit-session")

    token = generate_iam_auth_token(db_host="db", db_port=5432, db_user="litellm")

    sts_client_calls = [kw for svc, kw in boto3_stub.client_calls if svc == "sts"]
    assert sts_client_calls == []
    assert boto3_stub.sts_calls == []

    rds_client_calls = [kw for svc, kw in boto3_stub.client_calls if svc == "rds"]
    assert len(rds_client_calls) == 1
    assert rds_client_calls[0]["aws_access_key_id"] == "AKIA-EXPLICIT"
    assert rds_client_calls[0]["aws_secret_access_key"] == "explicit-secret"
    assert rds_client_calls[0]["aws_session_token"] == "explicit-session"

    assert token == urllib.parse.quote(FAKE_DB_AUTH_TOKEN, safe="")


@pytest.mark.parametrize("flag_value", [None, "false"])
def test_web_identity_token_used_when_flag_not_truthy(
    aws_env: pytest.MonkeyPatch, boto3_stub: _CallRecord, tmp_path: Path, flag_value: str | None
) -> None:
    oidc_contents = _set_irsa_pod_env(aws_env, tmp_path)
    if flag_value is not None:
        aws_env.setenv("AWS_RDS_IAM_IGNORE_WEB_IDENTITY_TOKEN", flag_value)

    token = generate_iam_auth_token(db_host="db", db_port=5432, db_user="litellm")

    sts_methods = [name for name, _ in boto3_stub.sts_calls]
    assert sts_methods == ["assume_role_with_web_identity"]
    assert boto3_stub.sts_calls[0][1]["RoleArn"] == "arn:rds"
    assert boto3_stub.sts_calls[0][1]["RoleSessionName"] == "sess"
    assert boto3_stub.sts_calls[0][1]["WebIdentityToken"] == oidc_contents

    assert token == urllib.parse.quote(FAKE_DB_AUTH_TOKEN, safe="")
