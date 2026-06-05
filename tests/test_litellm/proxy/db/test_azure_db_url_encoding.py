"""DB-IAM URL assembly: principal/db-name/schema encoding and writer/reader parity.

Azure Entra **user** principals are UPNs containing '@', and managed-identity /
service-principal names can contain other reserved characters; embedding them
raw would corrupt the postgresql:// URL. Encoding is a no-op for simple AWS RDS
IAM usernames, so the AWS path is unaffected. The reader (parsed-URL) path must
produce the same byte-identical URL as the writer (env) path for the same
principal — the bug a single shared builder is meant to prevent.

Run:
    uv run pytest tests/test_litellm/proxy/db/test_azure_db_url_encoding.py -v
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.db import db_iam_token
from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.prisma_client import PrismaWrapper, parse_iam_endpoint_from_url

_UPN = "svc@athenir.com"
_ENC = "svc%40athenir.com"
_HOST = "myserver.postgres.database.azure.com"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    yield


def _patch_token(value="TOK"):
    return patch(
        "litellm.proxy.db.db_iam_token.generate_db_iam_token", return_value=value
    )


# --- principal encoding (writer / env paths) --------------------------------


def test_get_rds_iam_token_url_encodes_principal(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", _HOST)
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", _UPN)
    monkeypatch.setenv("DATABASE_NAME", "litellm")

    wrapper = PrismaWrapper(original_prisma=MagicMock(), iam_token_db_auth=True)
    with _patch_token("TOKEN123"):
        url = wrapper.get_rds_iam_token()

    assert url == f"postgresql://{_ENC}:TOKEN123@{_HOST}:5432/litellm"
    assert f"{_UPN}:" not in url  # raw '@' must not appear in the userinfo
    assert os.environ["DATABASE_URL"] == url


def test_build_writer_url_encodes_principal(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", _HOST)
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", _UPN)
    monkeypatch.setenv("DATABASE_NAME", "litellm")

    with _patch_token():
        url = DatabaseURLSettings.from_env().build_writer_url()

    assert url == f"postgresql://{_ENC}:TOK@{_HOST}:5432/litellm"


def test_simple_username_is_unchanged(monkeypatch):
    """Encoding is a no-op for simple AWS RDS IAM usernames (no regression)."""
    monkeypatch.setenv("DATABASE_HOST", "db.rds.amazonaws.com")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", "rds_iam_user")
    monkeypatch.setenv("DATABASE_NAME", "litellm")

    wrapper = PrismaWrapper(original_prisma=MagicMock(), iam_token_db_auth=True)
    with _patch_token("T"):
        url = wrapper.get_rds_iam_token()

    assert url == "postgresql://rds_iam_user:T@db.rds.amazonaws.com:5432/litellm"


# --- build_postgres_url encoding --------------------------------------------


def test_build_postgres_url_basic():
    url = db_iam_token.build_postgres_url(
        user="u", token="t", host="h", port="5432", name="litellm"
    )
    assert url == "postgresql://u:t@h:5432/litellm"


def test_build_postgres_url_encodes_schema():
    # A schema value containing a reserved char must be encoded, not injected
    # raw as additional query parameters.
    url = db_iam_token.build_postgres_url(
        user="u", token="t", host="h", port="5432", name="db", schema="a&b"
    )
    assert url == "postgresql://u:t@h:5432/db?schema=a%26b"


# --- reader (parsed-URL) path parity ----------------------------------------


@pytest.mark.parametrize("reader_principal", [_UPN, _ENC])
def test_reader_endpoint_round_trip_matches_writer(reader_principal):
    """A reader URL written with either a raw OR a percent-encoded UPN parses to
    the same decoded principal and rebuilds to the same encoded URL the writer
    would emit (regression guard for the writer/reader divergence)."""
    reader_url = (
        f"postgresql://{reader_principal}:placeholder@reader.host:5432/litellm"
    )
    endpoint = parse_iam_endpoint_from_url(reader_url)

    assert endpoint.user == _UPN  # stored DECODED, regardless of input form
    built = endpoint.build_url("FRESHTOKEN")
    assert built == f"postgresql://{_ENC}:FRESHTOKEN@reader.host:5432/litellm"
