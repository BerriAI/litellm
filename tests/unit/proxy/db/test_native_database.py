import asyncio
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import BestAvailableEncryption, load_pem_private_key, pkcs12
from cryptography.x509.oid import NameOID
from pydantic import TypeAdapter

from litellm.proxy.db.native_database import (
    TokenRotation,
    keep_token_fresh,
    native_endpoint,
    needs_tls_probe,
    reader_token_rotation,
    server_offers_tls,
    writer_token_rotation,
)
from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR
from litellm.proxy.db.token_auth import IAMEndpoint, RdsIamTokenAuth
from litellm.rust_bridge.database import NativeDatabaseEndpoint, NativePoolLimits

NOW: Final = datetime(2026, 11, 3, 12, 0, 0)
_ADDRESS: Final = TypeAdapter(tuple[str, int])
PRISMA_DEFAULT_POOL: Final = NativePoolLimits(
    max_connections=2, acquire_timeout_seconds=10.0, idle_timeout_seconds=None, socket_timeout_seconds=None
)


def _rds_token(issued: datetime, lifetime_seconds: int) -> str:
    return (
        "db.internal:5432/?Action=connect&DBUser=litellm"
        f"&X-Amz-Date={issued:%Y%m%dT%H%M%SZ}&X-Amz-Expires={lifetime_seconds}&X-Amz-Signature=abc"
    )


def _endpoint(**overrides: object) -> NativeDatabaseEndpoint:
    return NativeDatabaseEndpoint(
        **{  # pyright: ignore[reportArgumentType]  # test overrides are checked by the equality assertion
            "host": "db",
            "port": 5432,
            "socket": None,
            "username": "u",
            "password": None,
            "database": "proxy",
            "ssl_mode": "prefer",
            "ssl_root_cert": None,
            "ssl_client": None,
            "application_name": "litellm-native",
            "options": (),
            "statement_cache_capacity": 100,
            "pool": PRISMA_DEFAULT_POOL,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "url, tls_offered, expected",
    [
        pytest.param(
            "postgresql://litellm:p%40ss%3Aword@db.internal:6543/proxy?connection_limit=12&pool_timeout=20"
            "&max_idle_connection_lifetime=60&socket_timeout=7&application_name=litellm-prisma",
            True,
            _endpoint(
                host="db.internal",
                port=6543,
                username="litellm",
                password="p@ss:word",
                pool=NativePoolLimits(
                    max_connections=2,
                    acquire_timeout_seconds=20.0,
                    idle_timeout_seconds=60.0,
                    socket_timeout_seconds=7.0,
                ),
            ),
            id="password_pool_params_and_native_share",
        ),
        pytest.param(
            "postgresql://svc%40corp@db/proxy?sslmode=require&sslcert=/certs/root.pem&sslaccept=strict"
            "&schema=Tenant&options=-c%20statement_timeout%3D30000%20-clock_timeout%3D5000&pgbouncer=true",
            True,
            _endpoint(
                username="svc@corp",
                ssl_mode="verify-full",
                ssl_root_cert="/certs/root.pem",
                options=(("search_path", '"Tenant"'), ("statement_timeout", "30000"), ("lock_timeout", "5000")),
                statement_cache_capacity=0,
            ),
            id="strict_tls_schema_timeouts_and_pgbouncer",
        ),
        pytest.param(
            "postgresql://u@db/proxy?statement_cache_size=0&application_name=billing-proxy",
            True,
            _endpoint(statement_cache_capacity=0, application_name="billing-proxy"),
            id="statement_cache_size_zero_and_an_operator_application_name_carry_over",
        ),
        pytest.param(
            "postgresql://u@db/proxy?statement_cache_size=250&pgbouncer=false",
            True,
            _endpoint(statement_cache_capacity=250),
            id="statement_cache_size_sets_the_capacity",
        ),
        pytest.param(
            "postgresql://u@db/proxy?statement_cache_size=250&pgbouncer=true",
            True,
            _endpoint(statement_cache_capacity=0),
            id="pgbouncer_disables_the_statement_cache_whatever_its_size",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslmode=require&sslcert=/certs/root.pem&schema=public",
            True,
            _endpoint(ssl_mode="require"),
            id="require_without_strict_never_verifies_the_operator_cert",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslcert=/certs/root.pem&sslaccept=strict",
            True,
            _endpoint(ssl_mode="verify-full", ssl_root_cert="/certs/root.pem"),
            id="prefer_strict_verifies_when_the_server_offers_tls",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslcert=/certs/root.pem&sslaccept=strict",
            False,
            _endpoint(ssl_mode="disable"),
            id="prefer_strict_falls_back_to_plaintext_without_server_tls",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslmode=disable&sslaccept=strict",
            True,
            _endpoint(ssl_mode="disable"),
            id="disable_wins_over_strict",
        ),
        pytest.param(
            "postgresql://u@localhost/proxy?host=/cloudsql/project:region:db&pool_timeout=0",
            True,
            _endpoint(
                host="localhost",
                socket="/cloudsql/project:region:db",
                pool=NativePoolLimits(
                    max_connections=2,
                    acquire_timeout_seconds=None,
                    idle_timeout_seconds=None,
                    socket_timeout_seconds=None,
                ),
            ),
            id="unix_socket_and_pool_timeout_zero_waits_forever",
        ),
    ],
)
def test_native_endpoint_reads_the_prisma_url(url: str, tls_offered: bool, expected: NativeDatabaseEndpoint) -> None:
    settings: Final = native_endpoint(url, tls_offered=tls_offered, max_connections=2)

    assert settings.endpoint == expected
    assert settings.ignored_options == ()


@pytest.mark.parametrize(
    "options, expected_options, expected_ignored",
    [
        pytest.param("-c statement_timeout=5 -B 64", (("statement_timeout", "5"),), ("-B", "64"), id="flags"),
        pytest.param("statement_timeout=5", (), ("statement_timeout=5",), id="bare_setting"),
        pytest.param("-c 'unterminated", (), ("-c 'unterminated",), id="unbalanced_quote"),
    ],
)
def test_native_endpoint_reports_options_it_cannot_send_instead_of_failing(
    options: str, expected_options: tuple[tuple[str, str], ...], expected_ignored: tuple[str, ...]
) -> None:
    settings: Final = native_endpoint(f"postgresql://u@db/proxy?options={options}", tls_offered=True, max_connections=2)

    assert settings.endpoint["options"] == expected_options
    assert settings.ignored_options == expected_ignored


def test_native_endpoint_rejects_a_url_without_a_user() -> None:
    with pytest.raises(ValueError, match="DATABASE_URL must name a host and a user"):
        native_endpoint("postgresql://db.internal/proxy", tls_offered=True, max_connections=2)


def _identity(password: bytes) -> tuple[bytes, ec.EllipticCurvePrivateKey, x509.Certificate]:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "litellm")])
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
        .not_valid_after(dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc))
        .sign(key, hashes.SHA256())
    )
    bundle: Final = pkcs12.serialize_key_and_certificates(
        b"litellm", key, cert, None, BestAvailableEncryption(password)
    )
    return bundle, key, cert


def test_native_endpoint_hands_over_the_pkcs12_identity_as_pem() -> None:
    bundle, key, cert = _identity(b"s3cret")
    read: Final = {Path("/certs/client.p12"): bundle}

    identity: Final = native_endpoint(
        "postgresql://u@db/proxy?sslmode=require&sslidentity=/certs/client.p12&sslpassword=s3cret",
        tls_offered=True,
        max_connections=2,
        read_file=read.__getitem__,
    ).endpoint["ssl_client"]

    assert identity is not None
    assert x509.load_pem_x509_certificate(identity["cert_pem"].encode()) == cert
    assert load_pem_private_key(identity["key_pem"].encode(), None).public_key() == key.public_key()


def test_native_endpoint_rejects_a_wrong_pkcs12_password() -> None:
    bundle, _, _ = _identity(b"s3cret")

    with pytest.raises(ValueError, match=r"Invalid password or PKCS12 data|Could not deserialize PKCS12 data"):
        native_endpoint(
            "postgresql://u@db/proxy?sslidentity=/certs/client.p12&sslpassword=wrong",
            tls_offered=True,
            max_connections=2,
            read_file=lambda _path: bundle,
        )


@pytest.mark.parametrize(
    "url, expected",
    [
        pytest.param("postgresql://u@db/proxy?sslaccept=strict", True, id="default_prefer_strict"),
        pytest.param("postgresql://u@db/proxy?sslmode=prefer&sslaccept=strict", True, id="prefer_strict"),
        pytest.param("postgresql://u@db/proxy?sslmode=require&sslaccept=strict", False, id="require_never_falls_back"),
        pytest.param("postgresql://u@db/proxy", False, id="prefer_unverified_falls_back_itself"),
        pytest.param("postgresql://u@localhost/proxy?sslaccept=strict&host=/run/pg", False, id="unix_socket"),
    ],
)
def test_only_prefer_strict_needs_a_tls_probe(url: str, expected: bool) -> None:
    assert needs_tls_probe(url) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, offered", [(b"S", True), (b"N", False)])
async def test_server_offers_tls_reads_the_ssl_request_answer(answer: bytes, offered: bool) -> None:
    received: Final = asyncio.Event()

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if await reader.readexactly(8) == b"\x00\x00\x00\x08\x04\xd2\x16\x2f":
            received.set()
        writer.write(answer)
        await writer.drain()
        writer.close()

    server: Final = await asyncio.start_server(respond, "127.0.0.1", 0)
    _, port = _ADDRESS.validate_python(server.sockets[0].getsockname())
    try:
        assert await server_offers_tls(f"postgresql://u@127.0.0.1:{port}/proxy") is offered
    finally:
        server.close()
        await server.wait_closed()
    assert received.is_set()


@pytest.mark.asyncio
async def test_unreachable_server_counts_as_offering_tls() -> None:
    assert await server_offers_tls("postgresql://u@127.0.0.1:1/proxy", timeout=1) is True


def test_writer_token_rotation_follows_token_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    url: Final = "postgresql://litellm:tok@db.internal:5432/proxy?schema=s"
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "false")
    assert writer_token_rotation(url) is None

    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AWS_RDS_REGION", "us-west-2")
    assert writer_token_rotation(url) == TokenRotation(
        auth=RdsIamTokenAuth(region="us-west-2"),
        endpoint=IAMEndpoint(host="db.internal", port="5432", user="litellm", name="proxy", schema="s"),
    )


def test_reader_token_rotation_uses_the_reader_region_even_when_the_writer_is_pooled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url: Final = "postgresql://reader:tok@replica.internal:5433/proxy"
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "true")
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "false")
    assert reader_token_rotation(url) is None

    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AWS_RDS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_RDS_READ_REPLICA_REGION", "eu-west-1")
    assert reader_token_rotation(url) == TokenRotation(
        auth=RdsIamTokenAuth(region="eu-west-1"),
        endpoint=IAMEndpoint(host="replica.internal", port="5433", user="reader", name="proxy"),
    )


def test_pooled_writer_needs_no_token_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "true")

    assert writer_token_rotation("postgresql://litellm:local@127.0.0.1:6432/proxy") is None


@pytest.mark.parametrize(
    "token, expected_seconds",
    [
        pytest.param(_rds_token(NOW, 900), 720.0, id="refreshes_buffer_before_expiry"),
        pytest.param(_rds_token(NOW - timedelta(seconds=890), 900), 30.0, id="never_sooner_than_retry_floor"),
        pytest.param("opaque", 600.0, id="unreadable_expiry_uses_fallback"),
    ],
)
def test_token_rotation_schedules_the_next_refresh(token: str, expected_seconds: float) -> None:
    rotation: Final = TokenRotation(
        auth=RdsIamTokenAuth(), endpoint=IAMEndpoint(host="db", port="5432", user="u", name="n")
    )

    assert rotation.seconds_until_refresh(token, NOW) == expected_seconds


class _Sink:
    def __init__(self, wanted: int) -> None:
        self.passwords: tuple[str, ...] = ()
        self._wanted: Final = wanted
        self.done: Final = asyncio.Event()

    def install(self, password: str) -> None:
        self.passwords = (*self.passwords, password)
        if len(self.passwords) >= self._wanted:
            self.done.set()


class _Schedule:
    def __init__(self) -> None:
        self.delays: tuple[float, ...] = ()

    async def sleep(self, seconds: float) -> None:
        self.delays = (*self.delays, seconds)
        await asyncio.sleep(0)


def _minting(*outcomes: str | Exception) -> TokenRotation:
    remaining: Final = iter(outcomes)

    def mint(_auth: object, _endpoint: object) -> str:
        outcome: Final = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return TokenRotation(
        auth=RdsIamTokenAuth(), endpoint=IAMEndpoint(host="db", port="5432", user="u", name="n"), mint_token=mint
    )


async def _run_until(sink: _Sink, rotation: TokenRotation, schedule: _Schedule) -> None:
    task: Final = asyncio.create_task(
        keep_token_fresh(sink.install, rotation, "opaque", now=lambda: NOW, sleep=schedule.sleep)
    )
    await asyncio.wait_for(sink.done.wait(), timeout=5)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_keep_token_fresh_installs_each_minted_token() -> None:
    sink: Final = _Sink(wanted=2)
    schedule: Final = _Schedule()

    await _run_until(sink, _minting("first%2Ftoken", _rds_token(NOW, 900)), schedule)

    assert sink.passwords == ("first/token", _rds_token(NOW, 900))
    assert schedule.delays[:2] == (600.0, 600.0)


@pytest.mark.asyncio
async def test_keep_token_fresh_retries_a_failed_mint() -> None:
    sink: Final = _Sink(wanted=1)
    schedule: Final = _Schedule()

    await _run_until(sink, _minting(RuntimeError("sts down"), "recovered"), schedule)

    assert sink.passwords == ("recovered",)
    assert schedule.delays[:2] == (600.0, 30.0)
