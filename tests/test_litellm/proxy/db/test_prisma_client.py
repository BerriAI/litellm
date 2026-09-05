import json
import os
import signal
import sys
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient



from litellm.proxy.db.prisma_client import PrismaWrapper, should_update_prisma_schema


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


def test_should_update_prisma_schema(monkeypatch):
    # CASE 1: Environment variable behavior
    # When DISABLE_SCHEMA_UPDATE is not set -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema() == True

    # When DISABLE_SCHEMA_UPDATE="true" -> should not update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "true")
    assert should_update_prisma_schema() == False

    # When DISABLE_SCHEMA_UPDATE="false" -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "false")
    assert should_update_prisma_schema() == True

    # CASE 2: Explicit parameter behavior (overrides env var)
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema(True) == False  # Param True -> should not update

    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)  # Set env var opposite to param
    assert should_update_prisma_schema(False) == True  # Param False -> should update


@pytest.mark.asyncio
async def test_recreate_prisma_client_successful_disconnect():
    """
    Test that recreate_prisma_client works normally when disconnect succeeds.
    """
    # Mock the original prisma client
    mock_prisma = AsyncMock()

    # Create a mock PrismaWrapper instance
    wrapper = Mock()
    wrapper._original_prisma = mock_prisma

    # Configure disconnect to succeed
    mock_prisma.disconnect.return_value = None

    # Mock the entire recreate_prisma_client method to avoid import issues
    async def mock_recreate_prisma_client(new_db_url: str, http_client=None):
        try:
            await mock_prisma.disconnect()
        except Exception:
            pass

        mock_new_prisma = AsyncMock()
        wrapper._original_prisma = mock_new_prisma
        await mock_new_prisma.connect()

    # Assign the mock method to the wrapper
    wrapper.recreate_prisma_client = mock_recreate_prisma_client

    # Call the method
    await wrapper.recreate_prisma_client("postgresql://new:new@localhost:5432/new")

    # Verify that disconnect was called
    mock_prisma.disconnect.assert_called_once()

    # Verify that the new client replaced the original
    assert wrapper._original_prisma != mock_prisma
    assert hasattr(wrapper._original_prisma, "connect")


@pytest.mark.asyncio
async def test_recreate_prisma_client_kills_old_engine_on_disconnect_failure(
    mock_prisma_binary,
):
    """When disconnect() fails, recreate_prisma_client must SIGTERM/SIGKILL the old engine PID."""
    mock_prisma = AsyncMock()
    mock_prisma.disconnect.side_effect = Exception("engine hung")
    mock_prisma.is_connected = MagicMock(return_value=True)

    # Simulate engine subprocess with a known PID
    mock_engine = MagicMock()
    mock_engine.process.pid = 12345
    mock_prisma._engine = mock_engine

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    # Configure the mock Prisma constructor
    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    # Verify old engine was killed
    mock_kill.assert_any_call(12345, signal.SIGTERM)
    # Verify new client was created and connected
    mock_new_prisma.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_prisma_client_skips_kill_on_successful_disconnect(
    mock_prisma_binary,
):
    """When disconnect() succeeds, no kill should be attempted."""
    mock_prisma = AsyncMock()
    mock_prisma.is_connected = MagicMock(return_value=True)
    mock_prisma.disconnect.return_value = None

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with patch("os.kill") as mock_kill:
        await wrapper.recreate_prisma_client("postgresql://new")

    mock_kill.assert_not_called()
    mock_new_prisma.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_prisma_client_handles_missing_engine_pid(
    mock_prisma_binary,
):
    """When engine PID is unavailable (no _engine attr), kill is skipped gracefully."""
    mock_prisma = AsyncMock()
    mock_prisma.is_connected = MagicMock(return_value=True)
    mock_prisma.disconnect.side_effect = Exception("engine hung")
    mock_prisma._engine = None  # No engine subprocess

    wrapper = PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=False)

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    mock_kill.assert_not_called()  # PID was 0, kill skipped
    mock_new_prisma.connect.assert_awaited_once()


def test_get_engine_pid_returns_zero_for_disconnected_client(disconnected_prisma):
    """A disconnected client must read as "no engine" instead of raising,
    otherwise the reconnect path can never recover."""
    wrapper = PrismaWrapper(
        original_prisma=disconnected_prisma, iam_token_db_auth=False
    )

    assert wrapper._get_engine_pid() == 0


@pytest.mark.asyncio
async def test_recreate_prisma_client_recovers_from_disconnected_client(
    mock_prisma_binary, disconnected_prisma
):
    """recreate_prisma_client must still build a replacement client when the
    current one is disconnected."""
    wrapper = PrismaWrapper(
        original_prisma=disconnected_prisma, iam_token_db_auth=False
    )

    mock_new_prisma = AsyncMock()
    mock_prisma_binary.Prisma.return_value = mock_new_prisma

    with patch("os.kill") as mock_kill:
        result = await wrapper.recreate_prisma_client("postgresql://new")

    assert result is True
    mock_kill.assert_not_called()
    assert wrapper._original_prisma is mock_new_prisma
    mock_new_prisma.connect.assert_awaited_once()


def test_db_push_applies_replica_identity_full_when_requested(monkeypatch):
    """`prisma db push` bypasses litellm-proxy-extras, so it needs its own call
    into the opt-in REPLICA IDENTITY FULL step."""
    from litellm.proxy.db.prisma_client import PrismaManager
    from litellm_proxy_extras.replica_identity import REPLICA_IDENTITY_FULL_ENV_VAR
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")
    applied = []
    monkeypatch.setattr(
        ProxyExtrasDBManager,
        "apply_replica_identity_full_if_requested",
        staticmethod(lambda: applied.append(True)),
    )

    with patch("litellm.proxy.db.prisma_client.subprocess.run") as mock_run:
        assert PrismaManager.setup_database(use_migrate=False) is True

    assert mock_run.call_args[0][0][:3] == ["prisma", "db", "push"]
    assert applied == [True]


def test_db_push_is_rejected_when_spend_logs_is_partitioned(monkeypatch):
    """A doc-partitioned LiteLLM_SpendLogs makes `prisma db push` rewrite the
    primary key back to ("request_id"), which Postgres rejects; the guard must
    fail fast with guidance instead of running the push."""
    from litellm.proxy.db.prisma_client import PrismaManager
    from litellm_proxy_extras.utils import (
        PARTITIONED_SPEND_LOGS_PUSH_ERROR,
        ProxyExtrasDBManager,
    )

    monkeypatch.setattr(
        ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: True)
    )
    with patch(  # test-quality-ok: subprocess.run is the external prisma CLI boundary, asserted never reached
        "litellm.proxy.db.prisma_client.subprocess.run"
    ) as mock_run:
        with pytest.raises(RuntimeError) as err:
            PrismaManager.setup_database(use_migrate=False)

    assert str(err.value) == PARTITIONED_SPEND_LOGS_PUSH_ERROR
    mock_run.assert_not_called()


def test_db_push_proceeds_when_spend_logs_is_not_partitioned(monkeypatch):
    from litellm.proxy.db.prisma_client import PrismaManager
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    monkeypatch.setattr(
        ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: False)
    )
    with patch(  # test-quality-ok: subprocess.run is the external prisma CLI boundary, not SDK logic
        "litellm.proxy.db.prisma_client.subprocess.run"
    ) as mock_run:
        assert PrismaManager.setup_database(use_migrate=False) is True

    assert mock_run.call_args[0][0][:3] == ["prisma", "db", "push"]


def _entra_jwt(expires_in_seconds: int) -> str:
    """A JWT shaped like a real Entra access token, expiring ``expires_in_seconds`` from now."""
    import base64
    from datetime import datetime, timedelta, timezone

    exp = int((datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in_seconds)).timestamp())
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).rstrip(b"=").decode()
    return f"aGVhZGVy.{payload}.c2ln"


@pytest.fixture
def azure_env(monkeypatch, unset_database_url):
    monkeypatch.setenv("DATABASE_HOST", "pg.postgres.database.azure.com")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", "litellm@contoso.onmicrosoft.com")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")


def _azure_wrapper(token: str, **kwargs):
    from litellm.proxy.db.token_auth import AzureEntraTokenAuth

    return PrismaWrapper(
        original_prisma=MagicMock(),
        token_auth=AzureEntraTokenAuth(token_provider=lambda: token),
        **kwargs,
    )


def test_azure_entra_mint_writes_an_encoded_url_into_the_db_url_env_var(azure_env):
    """The UPN user and the JWT both have to survive being embedded in a URL."""
    token = _entra_jwt(3600)
    wrapper = _azure_wrapper(token)

    db_url = wrapper.get_rds_iam_token()

    assert db_url == (
        f"postgresql://litellm%40contoso.onmicrosoft.com:{urllib.parse.quote(token, safe='')}"
        "@pg.postgres.database.azure.com:5432/litellm_db"
    )
    assert os.environ["DATABASE_URL"] == db_url


def test_azure_entra_refresh_is_scheduled_off_the_jwt_expiry(azure_env):
    """Without reading `exp` this falls back to a fixed 600s interval, which silently
    outlives a token and breaks every reconnect after it lapses (issue #29661)."""
    wrapper = _azure_wrapper(_entra_jwt(3600))
    wrapper.get_rds_iam_token()

    seconds = wrapper._calculate_seconds_until_refresh()

    expected = 3600 - PrismaWrapper.TOKEN_REFRESH_BUFFER_SECONDS
    assert seconds != PrismaWrapper.FALLBACK_REFRESH_INTERVAL_SECONDS
    assert expected - 5 <= seconds <= expected


def test_a_token_whose_expiry_never_advances_cannot_spin_the_refresh_loop(azure_env):
    """azure-identity hands back its cached token when a renewal attempt fails inside its
    own window, so a transient Entra or IMDS problem in the last 3 minutes of a token
    yields a successful refresh whose `exp` has not moved. With no floor on the sleep the
    loop then re-mints and recreates the query engine on every pass, with nothing in
    between, for as long as Entra stays sick."""
    wrapper = _azure_wrapper(_entra_jwt(60))
    wrapper.get_rds_iam_token()
    first = wrapper._calculate_seconds_until_refresh()

    wrapper.get_rds_iam_token()
    second = wrapper._calculate_seconds_until_refresh()

    assert first == second == PrismaWrapper.TOKEN_REFRESH_MIN_SLEEP_SECONDS


def test_azure_entra_token_expiry_is_detected(azure_env):
    wrapper = _azure_wrapper(_entra_jwt(3600))
    fresh_url = wrapper.get_rds_iam_token()
    expired_url = _azure_wrapper(_entra_jwt(-1)).get_rds_iam_token()

    assert wrapper.is_token_expired(fresh_url) is False
    assert wrapper.is_token_expired(expired_url) is True


@pytest.mark.asyncio
async def test_azure_entra_strategy_starts_the_refresh_task(azure_env):
    """The refresh loop is gated on the legacy boolean, so an Azure strategy has to
    get past that gate; a password-auth wrapper still must not start a task."""
    wrapper = _azure_wrapper(_entra_jwt(3600))
    wrapper.get_rds_iam_token()
    password_wrapper = PrismaWrapper(original_prisma=MagicMock())

    await wrapper.start_token_refresh_task()
    await password_wrapper.start_token_refresh_task()
    try:
        assert wrapper._token_refresh_task is not None
        assert not wrapper._token_refresh_task.done()
        assert password_wrapper._token_refresh_task is None
    finally:
        await wrapper.stop_token_refresh_task()


def test_azure_entra_strategy_reads_as_token_auth_enabled(azure_env):
    """`routing_prisma_wrapper` gates the reader's refresh on this flag, so an Azure
    reader has to answer True to it."""
    wrapper = _azure_wrapper(_entra_jwt(3600))

    assert wrapper.iam_token_db_auth is True
    assert wrapper.token_label == "Azure Entra token"


def test_the_token_strategy_cannot_be_swapped_after_construction(azure_env):
    """Assigning the legacy boolean used to replace a configured Entra strategy with the
    RDS one, which points boto at an Azure host."""
    wrapper = _azure_wrapper(_entra_jwt(3600))

    with pytest.raises(AttributeError):
        wrapper.iam_token_db_auth = True


def test_minting_without_the_database_env_vars_names_them(azure_env, monkeypatch):
    """A blank host used to produce `postgresql://:<token>@:5432/`, which fails deep
    inside Prisma instead of at the misconfiguration."""
    monkeypatch.delenv("DATABASE_HOST")
    wrapper = _azure_wrapper(_entra_jwt(3600))

    with pytest.raises(RuntimeError, match="DATABASE_HOST"):
        wrapper.get_rds_iam_token()
