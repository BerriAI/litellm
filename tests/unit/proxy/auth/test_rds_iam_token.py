from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import boto3
import pytest

from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token


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
