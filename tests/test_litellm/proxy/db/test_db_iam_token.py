"""Unit tests for the DB IAM provider dispatcher (litellm.proxy.db.db_iam_token).

The provider is auto-detected from the database host: Azure Database for
PostgreSQL hosts (``*.postgres.database.azure.com``) use the Azure Entra path,
everything else uses the AWS RDS IAM default.

Run:
    uv run pytest tests/test_litellm/proxy/db/test_db_iam_token.py -v
"""

from unittest.mock import patch

import pytest

from litellm.proxy.db import db_iam_token


@pytest.fixture(autouse=True)
def _clear_host_env(monkeypatch):
    monkeypatch.delenv("DATABASE_HOST", raising=False)
    yield


def test_default_provider_is_aws():
    assert db_iam_token.get_db_iam_auth_provider() == "aws"


@pytest.mark.parametrize(
    "host,expected",
    [
        ("myserver.postgres.database.azure.com", "azure"),
        ("MYSERVER.POSTGRES.DATABASE.AZURE.COM", "azure"),  # case-insensitive
        ("db.cluster-xyz.us-east-1.rds.amazonaws.com", "aws"),
        ("localhost", "aws"),
        ("", "aws"),
        (None, "aws"),
    ],
)
def test_provider_detected_from_host(host, expected):
    assert db_iam_token.get_db_iam_auth_provider(host) == expected


def test_provider_detected_from_database_host_env(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "x.postgres.database.azure.com")
    assert db_iam_token.get_db_iam_auth_provider() == "azure"


def test_dispatch_aws_for_rds_host():
    with patch(
        "litellm.proxy.auth.rds_iam_token.generate_iam_auth_token",
        return_value="aws-token",
    ) as aws_mock:
        token = db_iam_token.generate_db_iam_token(
            db_host="h.rds.amazonaws.com", db_port="5432", db_user="u"
        )
    assert token == "aws-token"
    aws_mock.assert_called_once_with(
        db_host="h.rds.amazonaws.com", db_port="5432", db_user="u"
    )


def test_dispatch_azure_for_azure_host():
    with patch(
        "litellm.proxy.auth.azure_entra_db_token.generate_azure_entra_db_token",
        return_value="azure-token",
    ) as az_mock:
        token = db_iam_token.generate_db_iam_token(
            db_host="h.postgres.database.azure.com", db_port="5432", db_user="mi-name"
        )
    assert token == "azure-token"
    az_mock.assert_called_once_with(
        db_host="h.postgres.database.azure.com", db_port="5432", db_user="mi-name"
    )
