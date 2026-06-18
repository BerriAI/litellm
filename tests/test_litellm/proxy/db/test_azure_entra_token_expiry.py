"""Tests for Azure Entra ID (JWT) token-expiry parsing in PrismaWrapper.

The proactive refresh loop schedules the next refresh from the token's parsed
expiry. AWS RDS presigned tokens carry expiry in query params; Azure Entra ID
tokens are JWTs carrying an ``exp`` claim. Both must parse; anything else falls
back to the fixed refresh interval.

Run:
    uv run pytest tests/test_litellm/proxy/db/test_azure_entra_token_expiry.py -v
"""

import base64
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from litellm.proxy.db.prisma_client import PrismaWrapper


def _b64url(data: dict) -> str:
    raw = json.dumps(data).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _make_jwt(exp_epoch: int) -> str:
    header = _b64url({"alg": "RS256", "typ": "JWT"})
    payload = _b64url(
        {"exp": exp_epoch, "aud": "https://ossrdbms-aad.database.windows.net"}
    )
    return f"{header}.{payload}.signature"


def _wrapper() -> PrismaWrapper:
    return PrismaWrapper(original_prisma=MagicMock(), iam_token_db_auth=True)


def test_parses_jwt_exp_claim():
    exp = int(time.time()) + 3600
    parsed = _wrapper()._parse_token_expiration(_make_jwt(exp))
    assert parsed == datetime.utcfromtimestamp(exp)


def test_aws_presigned_token_still_parsed():
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%dT%H%M%SZ")
    token = f"mock?X-Amz-Date={date_str}&X-Amz-Expires=900&X-Amz-Signature=abc"
    parsed = _wrapper()._parse_token_expiration(token)
    assert parsed is not None
    assert abs((parsed - (now + timedelta(seconds=900))).total_seconds()) < 2


def test_unparseable_tokens_return_none():
    w = _wrapper()
    assert w._parse_token_expiration(None) is None
    assert w._parse_token_expiration("not-a-token") is None
    # A 3-segment string whose payload has no exp claim.
    assert w._parse_token_expiration(_make_jwt_without_exp()) is None


def _make_jwt_without_exp() -> str:
    header = _b64url({"alg": "RS256", "typ": "JWT"})
    payload = _b64url({"aud": "https://ossrdbms-aad.database.windows.net"})
    return f"{header}.{payload}.signature"


def test_extract_jwt_token_from_db_url_then_parse():
    exp = int(time.time()) + 2700
    jwt = _make_jwt(exp)
    db_url = (
        f"postgresql://mi-name:{jwt}@host.postgres.database.azure.com:5432/litellm"
    )
    w = _wrapper()
    extracted = w._extract_token_from_db_url(db_url)
    assert extracted == jwt
    assert w._parse_token_expiration(extracted) == datetime.utcfromtimestamp(exp)
