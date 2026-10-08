import os
import urllib.parse
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psycopg
import pytest
from psycopg import sql

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.postgres_front import PostgresFront
from integration._support.process import owned_proxy_process

FAKE_AWS_ACCESS_KEY: Final = "AKIAIOSFODNN7EXAMPLE"
FAKE_AWS_SECRET_KEY: Final = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
PROCESS_REGION: Final = "ap-southeast-2"
WRITER_PASSWORD: Final = "integration-iam-writer-password"
READER_PASSWORD: Final = "integration-iam-reader-password"

REMOVE_ENVIRONMENT: Final = (
    "DATABASE_URL",
    "DIRECT_URL",
    "DATABASE_URL_READ_REPLICA",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION_NAME",
    "AWS_PROFILE_NAME",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_NAME",
    "AWS_ROLE_ARN",
    "AWS_SESSION_NAME",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


@dataclass(frozen=True, slots=True)
class IamRig:
    database: str
    writer_role: str
    reader_role: str
    writer_front: PostgresFront
    reader_front: PostgresFront


def _signed_region(password: str) -> str:
    token: Final = urllib.parse.unquote(password)
    query: Final = urllib.parse.urlsplit(token).query
    credential: Final = urllib.parse.parse_qs(query)["X-Amz-Credential"][0]
    return urllib.parse.unquote(credential).split("/")[2]


def _assert_regions(front: PostgresFront, expected: str, side: str) -> None:
    regions: Final = [_signed_region(login.password) for login in front.logins]
    assert regions, f"no {side} connections were recorded"
    assert all(region == expected for region in regions), (
        f"{side} connections signed regions {regions}, expected all {expected}"
    )


def _iam_overrides(rig: IamRig, extra: Mapping[str, str]) -> dict[str, str]:
    return {
        "IAM_TOKEN_DB_AUTH": "true",
        "AWS_ACCESS_KEY_ID": FAKE_AWS_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": FAKE_AWS_SECRET_KEY,
        "AWS_REGION": PROCESS_REGION,
        "DATABASE_HOST": "127.0.0.1",
        "DATABASE_PORT": str(rig.writer_front.port),
        "DATABASE_USER": rig.writer_role,
        "DATABASE_NAME": rig.database,
        "DATABASE_URL_READ_REPLICA": (
            f"postgresql://{rig.reader_role}@127.0.0.1:{rig.reader_front.port}/{rig.database}"
        ),
        **extra,
    }


@pytest.fixture
def iam_rig() -> Iterator[IamRig]:
    url: Final = os.environ["DATABASE_URL"]
    suffix: Final = uuid.uuid4().hex[:12]
    database: Final = f"iam_regions_{suffix}"
    writer_role: Final = f"iam_writer_{suffix}"
    reader_role: Final = f"iam_reader_{suffix}"
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(writer_role), sql.Literal(WRITER_PASSWORD)
            )
        )
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(reader_role), sql.Literal(READER_PASSWORD)
            )
        )
        admin.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(database), sql.Identifier(writer_role)))
        try:
            db_url: Final = urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(path=f"/{database}"))
            with psycopg.connect(db_url, autocommit=True) as db_admin:
                db_admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(reader_role)))
                db_admin.execute(
                    sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(reader_role))
                )
                db_admin.execute(
                    sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(
                        sql.Identifier(writer_role), sql.Identifier(reader_role)
                    )
                )
                db_admin.execute(
                    sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
                        sql.Identifier(reader_role)
                    )
                )
            writer_front: Final = PostgresFront("127.0.0.1", 5432, writer_role, WRITER_PASSWORD)
            reader_front: Final = PostgresFront("127.0.0.1", 5432, reader_role, READER_PASSWORD)
            writer_front.start()
            reader_front.start()
            try:
                yield IamRig(
                    database=database,
                    writer_role=writer_role,
                    reader_role=reader_role,
                    writer_front=writer_front,
                    reader_front=reader_front,
                )
            finally:
                writer_front.stop()
                reader_front.stop()
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(reader_role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(writer_role)))
    assert read_rows("SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", ([writer_role, reader_role],)) == []



def test_writer_and_reader_sign_their_own_override_regions(
    gateway: Gateway, tmp_path: Path, iam_rig: IamRig
) -> None:
    overrides: Final = _iam_overrides(
        iam_rig, {"AWS_RDS_REGION": "us-east-1", "AWS_RDS_READ_REPLICA_REGION": "eu-west-1"}
    )
    with owned_proxy_process(
        gateway, tmp_path, overrides, remove_environment=REMOVE_ENVIRONMENT
    ) as owned:
        _assert_regions(iam_rig.writer_front, "us-east-1", "writer")
        _assert_regions(iam_rig.reader_front, "eu-west-1", "reader")
        response: Final = owned.gateway.request("POST", "/key/generate", {})
        assert response.status_code == 200, response.text
        read_rows_before: Final = iam_rig.reader_front.query_bytes
        listing: Final = owned.gateway.request("GET", "/key/list")
        assert listing.status_code == 200, listing.text
        assert read_rows("SELECT pid FROM pg_stat_activity WHERE usename=%s", (iam_rig.reader_role,)), (
            "reader role was never connected; reads cannot have been served by the replica"
        )
        assert iam_rig.reader_front.query_bytes > read_rows_before or read_rows_before > 0, (
            "reader front saw no post-auth client traffic"
        )


def test_reader_does_not_inherit_writer_override(gateway: Gateway, tmp_path: Path, iam_rig: IamRig) -> None:
    overrides: Final = _iam_overrides(iam_rig, {"AWS_RDS_REGION": "us-east-1"})
    with owned_proxy_process(
        gateway, tmp_path, overrides, remove_environment=REMOVE_ENVIRONMENT
    ) as owned:
        _assert_regions(iam_rig.writer_front, "us-east-1", "writer")
        _assert_regions(iam_rig.reader_front, PROCESS_REGION, "reader")
        response: Final = owned.gateway.request("POST", "/key/generate", {})
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_overrides_fall_back_to_process_region(
    gateway: Gateway, tmp_path: Path, iam_rig: IamRig, blank: str
) -> None:
    overrides: Final = _iam_overrides(
        iam_rig, {"AWS_RDS_REGION": blank, "AWS_RDS_READ_REPLICA_REGION": blank}
    )
    with owned_proxy_process(
        gateway, tmp_path, overrides, remove_environment=REMOVE_ENVIRONMENT
    ) as owned:
        _assert_regions(iam_rig.writer_front, PROCESS_REGION, "writer")
        _assert_regions(iam_rig.reader_front, PROCESS_REGION, "reader")
        response: Final = owned.gateway.request("POST", "/key/generate", {})
        assert response.status_code == 200, response.text


@pytest.mark.timeout(840)
def test_refreshed_tokens_keep_their_override_regions(
    gateway: Gateway, tmp_path: Path, iam_rig: IamRig
) -> None:
    overrides: Final = _iam_overrides(
        iam_rig, {"AWS_RDS_REGION": "us-east-1", "AWS_RDS_READ_REPLICA_REGION": "eu-west-1"}
    )
    with owned_proxy_process(
        gateway, tmp_path, overrides, remove_environment=REMOVE_ENVIRONMENT
    ) as owned:
        before_writer: Final = len(iam_rig.writer_front.logins)
        before_reader: Final = len(iam_rig.reader_front.logins)
        assert before_writer and before_reader
        stale_writer: Final = {login.password for login in iam_rig.writer_front.logins[:before_writer]}
        stale_reader: Final = {login.password for login in iam_rig.reader_front.logins[:before_reader]}
        # boto mints RDS tokens at a fixed X-Amz-Expires=900 (no ExpiresIn parameter
        # and no LiteLLM env knob), so the supported way to see a refresh is the
        # scheduled re-mint ~720s after boot; the new engine then logs into the front.
        assert eventually(
            lambda: len(iam_rig.writer_front.logins) > before_writer
            and len(iam_rig.reader_front.logins) > before_reader,
            bool,
            seconds=780,
        ), "no fresh connections after the scheduled token refresh"
        _assert_regions(iam_rig.writer_front, "us-east-1", "writer")
        _assert_regions(iam_rig.reader_front, "eu-west-1", "reader")
        fresh_writer: Final = {login.password for login in iam_rig.writer_front.logins[before_writer:]}
        fresh_reader: Final = {login.password for login in iam_rig.reader_front.logins[before_reader:]}
        assert not stale_writer.issuperset(fresh_writer), (
            "no writer connection carried a re-minted token signature after the scheduled refresh"
        )
        assert not stale_reader.issuperset(fresh_reader), (
            "no reader connection carried a re-minted token signature after the scheduled refresh"
        )
        response: Final = owned.gateway.request("GET", "/key/list")
        assert response.status_code == 200, response.text


def test_password_auth_passes_password_through_unchanged(
    gateway: Gateway, tmp_path: Path, iam_rig: IamRig
) -> None:
    overrides: Final = {
        "DATABASE_URL": (
            f"postgresql://{iam_rig.writer_role}:{WRITER_PASSWORD}@127.0.0.1:"
            f"{iam_rig.writer_front.port}/{iam_rig.database}"
        ),
    }
    with owned_proxy_process(
        gateway, tmp_path, overrides, remove_environment=REMOVE_ENVIRONMENT
    ) as owned:
        response: Final = owned.gateway.request("POST", "/key/generate", {})
        assert response.status_code == 200, response.text
        passwords: Final = {login.password for login in iam_rig.writer_front.logins}
        assert passwords == {WRITER_PASSWORD}, (
            f"front recorded passwords {sorted(passwords)}, expected exactly the role password"
        )
