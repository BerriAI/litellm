from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import boto3
import pytest
from botocore.exceptions import ClientError

from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token, init_rds_client


@pytest.mark.parametrize(
    ("db_host", "expected_region"),
    [
        ("w.abc123.us-east-1.rds.amazonaws.com", "us-east-1"),
        ("lit-r.abc123xyz.ap-northeast-1.rds.amazonaws.com", "ap-northeast-1"),
        ("c.cluster-abc123.eu-central-1.rds.amazonaws.com", "eu-central-1"),
        ("c.cluster-ro-abc.eu-west-2.rds.amazonaws.com", "eu-west-2"),
        ("p.proxy-abc123.us-west-2.rds.amazonaws.com", "us-west-2"),
        ("ep1.endpoint.proxy-ab0cd1efghij.us-east-2.rds.amazonaws.com", "us-east-2"),
        ("d.abc123.cn-north-1.rds.amazonaws.com.cn", "cn-north-1"),
        ("d.abc123.us-gov-west-1.rds.amazonaws.com", "us-gov-west-1"),
        ("W.ABC123.US-EAST-1.RDS.AMAZONAWS.COM", "us-east-1"),
        ("lit.abc.us-east-1.rds.amazonaws.com.", "us-east-1"),
        ("writer.aurora.local", "ap-northeast-1"),
        ("db.internal.example.com", "ap-northeast-1"),
        ("10.0.0.5", "ap-northeast-1"),
        ("localhost", "ap-northeast-1"),
        ("us-east-1.rds.amazonaws.com.evil.example", "ap-northeast-1"),
    ],
)
def test_generate_iam_auth_token_signs_for_the_database_hostname_region(
    db_host: str, expected_region: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    client: Final = boto3.client(
        "rds",
        region_name="ap-northeast-1",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="x" * 40,
    )

    token: Final = unquote(
        generate_iam_auth_token(
            db_host=db_host,
            db_port="5432",
            db_user="litellm",
            client=client,
        )
    )
    credential: Final = parse_qs(urlsplit(token).query)["X-Amz-Credential"][0]

    assert token.split("?", maxsplit=1)[0] == f"{db_host}:5432/"
    assert credential.split("/")[2] == expected_region


POD_ROLE: Final = "arn:aws:iam::111111111111:role/litellm-pod-irsa"
DB_ROLE: Final = "arn:aws:iam::222222222222:role/litellm-db-connect"


def _creds(tag: str) -> dict[str, dict[str, str]]:
    return {
        "Credentials": {"AccessKeyId": f"ASIA{tag}", "SecretAccessKey": f"secret-{tag}", "SessionToken": f"token-{tag}"}
    }


def _refused(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "Not authorized"}}, operation)


class _FakeSts:
    def __init__(
        self, calls: list[tuple[str, str, str | None]], roles_trusting_oidc: frozenset[str], caller: str | None
    ) -> None:
        self._calls: Final = calls
        self._roles_trusting_oidc: Final = roles_trusting_oidc
        self._caller: Final = caller

    def assume_role_with_web_identity(
        self, *, RoleArn: str, RoleSessionName: str, WebIdentityToken: str, **_: int
    ) -> dict[str, dict[str, str]]:
        self._calls.append(("web_identity", RoleArn, WebIdentityToken))
        if RoleArn not in self._roles_trusting_oidc:
            raise _refused("AssumeRoleWithWebIdentity")
        return _creds(RoleArn.rsplit("/", maxsplit=1)[1])

    def assume_role(self, *, RoleArn: str, RoleSessionName: str) -> dict[str, dict[str, str]]:
        self._calls.append(("assume_role", RoleArn, self._caller))
        if self._caller != f"ASIA{POD_ROLE.rsplit('/', maxsplit=1)[1]}":
            raise _refused("AssumeRole")
        return _creds("chained")


def _boto3_client_factory(calls: list[tuple[str, str, str | None]], roles_trusting_oidc: frozenset[str]):
    rds_clients: Final[list[dict[str, str]]] = []

    def client(service_name: str, **kwargs: str) -> object:
        if service_name == "sts":
            return _FakeSts(calls, roles_trusting_oidc, kwargs.get("aws_access_key_id"))
        rds_clients.append({k: v for k, v in kwargs.items() if k.startswith("aws_")})
        return rds_clients[-1]

    return client, rds_clients


@pytest.fixture
def web_identity_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    token_file: Final = tmp_path / "token"
    token_file.write_text("eyJ.oidc.token")
    monkeypatch.setenv("AWS_ROLE_ARN", POD_ROLE)
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_file))
    return str(token_file)


def _init(
    monkeypatch: pytest.MonkeyPatch, token_file: str, roles_trusting_oidc: frozenset[str]
) -> tuple[list[tuple[str, str, str | None]], list[dict[str, str]]]:
    calls: Final[list[tuple[str, str, str | None]]] = []
    client, rds_clients = _boto3_client_factory(calls, roles_trusting_oidc)
    monkeypatch.setattr(boto3, "client", client)
    init_rds_client(
        aws_role_name=DB_ROLE,
        aws_session_name="litellm",
        aws_web_identity_token=token_file,
        aws_region_name="us-east-1",
    )
    return calls, rds_clients


def test_init_rds_client_chains_through_the_irsa_role_when_the_db_role_only_trusts_the_pod_role(
    web_identity_token_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, rds_clients = _init(monkeypatch, web_identity_token_file, roles_trusting_oidc=frozenset({POD_ROLE}))

    assert rds_clients == [
        {
            "aws_access_key_id": "ASIAchained",
            "aws_secret_access_key": "secret-chained",
            "aws_session_token": "token-chained",
        }
    ]
    assert calls[-2:] == [
        ("web_identity", POD_ROLE, "eyJ.oidc.token"),
        ("assume_role", DB_ROLE, "ASIAlitellm-pod-irsa"),
    ]


def test_init_rds_client_keeps_the_direct_web_identity_exchange_when_the_db_role_trusts_the_oidc_provider(
    web_identity_token_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, rds_clients = _init(monkeypatch, web_identity_token_file, roles_trusting_oidc=frozenset({POD_ROLE, DB_ROLE}))

    assert rds_clients == [
        {
            "aws_access_key_id": "ASIAlitellm-db-connect",
            "aws_secret_access_key": "secret-litellm-db-connect",
            "aws_session_token": "token-litellm-db-connect",
        }
    ]
    assert calls == [("web_identity", DB_ROLE, "eyJ.oidc.token")]


def test_init_rds_client_surfaces_the_sts_refusal_when_there_is_no_distinct_irsa_role_to_chain_through(
    web_identity_token_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ROLE_ARN", DB_ROLE)

    with pytest.raises(ClientError, match="AssumeRoleWithWebIdentity"):
        _init(monkeypatch, web_identity_token_file, roles_trusting_oidc=frozenset())
