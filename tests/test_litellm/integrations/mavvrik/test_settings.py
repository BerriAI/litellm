"""Unit tests for MavvrikSettings — config detection, persistence, encryption."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.settings import MavvrikSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SETTINGS_MODULE = "litellm.integrations.mavvrik.settings"


def _make_db_row(value: dict):
    """Return a mock DB row whose param_value is the JSON-serialised dict."""
    row = MagicMock()
    row.param_value = json.dumps(value)
    return row


def _mock_prisma(row=None, *, delete_ok: bool = True):
    """Return a mock prisma_client with pre-configured behaviour."""
    client = MagicMock()
    client.db.litellm_config.find_first = AsyncMock(return_value=row)
    client.db.litellm_config.upsert = AsyncMock()
    client.db.litellm_config.delete = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# is_setup()
# ---------------------------------------------------------------------------


class TestIsSetup:
    @pytest.mark.asyncio
    async def test_returns_true_via_env_vars(self):
        """is_setup() returns True when all three env vars are present."""
        s = MavvrikSettings()
        env = {
            "MAVVRIK_API_KEY": "mav_key",
            "MAVVRIK_API_ENDPOINT": "https://api.mavvrik.dev/acme",
            "MAVVRIK_CONNECTION_ID": "prod",
        }
        with patch.dict("os.environ", env):
            result = await s.is_setup()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_via_db(self):
        """is_setup() returns True when a DB row exists (no env vars)."""
        s = MavvrikSettings()
        mock_row = _make_db_row({"api_key": "enc", "api_endpoint": "https://e", "connection_id": "c"})
        mock_client = _mock_prisma(row=mock_row)

        env = {k: "" for k in ("MAVVRIK_API_KEY", "MAVVRIK_API_ENDPOINT", "MAVVRIK_CONNECTION_ID")}
        with patch.dict("os.environ", env), patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            result = await s.is_setup()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_neither_configured(self):
        """is_setup() returns False when env vars are missing and DB has no row."""
        s = MavvrikSettings()
        mock_client = _mock_prisma(row=None)

        env = {k: "" for k in ("MAVVRIK_API_KEY", "MAVVRIK_API_ENDPOINT", "MAVVRIK_CONNECTION_ID")}
        with patch.dict("os.environ", env), patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            result = await s.is_setup()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_prisma_client_is_none(self):
        """is_setup() returns False when no DB is connected and env vars absent."""
        s = MavvrikSettings()
        env = {k: "" for k in ("MAVVRIK_API_KEY", "MAVVRIK_API_ENDPOINT", "MAVVRIK_CONNECTION_ID")}
        with patch.dict("os.environ", env), patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: None)
        ):
            result = await s.is_setup()

        assert result is False


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    @pytest.mark.asyncio
    async def test_save_encrypts_api_key_and_persists(self):
        """save() encrypts the api_key before writing to the database."""
        s = MavvrikSettings()
        mock_client = _mock_prisma()

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ), patch.object(s, "encrypt_value_helper", return_value="encrypted_key") as mock_enc:
            await s.save(
                api_key="plaintext_key",
                api_endpoint="https://api.mavvrik.dev/acme",
                connection_id="prod",
                marker="2024-01-01",
            )

        mock_enc.assert_called_once_with("plaintext_key")
        mock_client.db.litellm_config.upsert.assert_called_once()
        call_data = mock_client.db.litellm_config.upsert.call_args[1]["data"]
        stored = json.loads(call_data["create"]["param_value"])
        assert stored["api_key"] == "encrypted_key"
        assert stored["api_endpoint"] == "https://api.mavvrik.dev/acme"
        assert stored["connection_id"] == "prod"
        assert stored["marker"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_save_omits_marker_when_not_provided(self):
        """save() does not include a marker key when marker=None."""
        s = MavvrikSettings()
        mock_client = _mock_prisma()

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ), patch.object(s, "encrypt_value_helper", return_value="enc"):
            await s.save(
                api_key="key",
                api_endpoint="https://api.mavvrik.dev/acme",
                connection_id="c",
            )

        call_data = mock_client.db.litellm_config.upsert.call_args[1]["data"]
        stored = json.loads(call_data["create"]["param_value"])
        assert "marker" not in stored


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_decrypts_api_key(self):
        """load() returns settings with api_key already decrypted."""
        s = MavvrikSettings()
        row = _make_db_row({
            "api_key": "encrypted_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
            "marker": "2024-01-14",
        })
        mock_client = _mock_prisma(row=row)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ), patch.object(s, "decrypt_value_helper", return_value="plaintext_key"):
            result = await s.load()

        assert result["api_key"] == "plaintext_key"
        assert result["api_endpoint"] == "https://api.mavvrik.dev/acme"
        assert result["marker"] == "2024-01-14"

    @pytest.mark.asyncio
    async def test_load_returns_empty_dict_when_no_row(self):
        """load() returns {} when no row exists in the database."""
        s = MavvrikSettings()
        mock_client = _mock_prisma(row=None)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            result = await s.load()

        assert result == {}


# ---------------------------------------------------------------------------
# advance_marker()
# ---------------------------------------------------------------------------


class TestAdvanceMarker:
    @pytest.mark.asyncio
    async def test_advance_marker_updates_only_marker_field(self):
        """advance_marker() preserves all other fields while updating marker."""
        s = MavvrikSettings()
        existing = {
            "api_key": "already_encrypted",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
            "marker": "2024-01-01T00:00:00+00:00",
        }
        row = _make_db_row(existing)
        mock_client = _mock_prisma(row=row)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            await s.advance_marker("2024-02-01T00:00:00+00:00")

        mock_client.db.litellm_config.upsert.assert_called_once()
        call_data = mock_client.db.litellm_config.upsert.call_args[1]["data"]
        stored = json.loads(call_data["create"]["param_value"])
        assert stored["marker"] == "2024-02-01T00:00:00+00:00"
        # Other fields must be preserved unchanged
        assert stored["api_key"] == "already_encrypted"
        assert stored["api_endpoint"] == "https://api.mavvrik.dev/acme"
        assert stored["connection_id"] == "prod"

    @pytest.mark.asyncio
    async def test_advance_marker_creates_settings_when_no_row(self):
        """advance_marker() creates a minimal row when no settings exist yet."""
        s = MavvrikSettings()
        mock_client = _mock_prisma(row=None)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            await s.advance_marker("2024-03-01T00:00:00+00:00")

        mock_client.db.litellm_config.upsert.assert_called_once()
        call_data = mock_client.db.litellm_config.upsert.call_args[1]["data"]
        stored = json.loads(call_data["create"]["param_value"])
        assert stored["marker"] == "2024-03-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_config_row(self):
        """delete() calls prisma delete when the row exists."""
        s = MavvrikSettings()
        row = _make_db_row({"api_key": "enc", "api_endpoint": "https://e", "connection_id": "c"})
        mock_client = _mock_prisma(row=row)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            await s.delete()

        mock_client.db.litellm_config.delete.assert_called_once_with(
            where={"param_name": "mavvrik_settings"}
        )

    @pytest.mark.asyncio
    async def test_delete_raises_lookup_error_when_not_configured(self):
        """delete() raises LookupError when no settings row exists."""
        s = MavvrikSettings()
        mock_client = _mock_prisma(row=None)

        with patch.object(
            type(s), "_prisma_client", new_callable=lambda: property(lambda self: mock_client)
        ):
            with pytest.raises(LookupError):
                await s.delete()

        mock_client.db.litellm_config.delete.assert_not_called()
