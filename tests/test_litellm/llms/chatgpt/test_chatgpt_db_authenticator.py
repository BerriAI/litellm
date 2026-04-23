import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.chatgpt.db_authenticator import (
    CREDENTIAL_TYPE,
    DBAuthenticator,
    _pack_auth_record,
    _unpack_auth_record,
    persist_credential_to_db,
)
from litellm.types.utils import CredentialItem


class TestAuthRecordPacking:
    def test_pack_stringifies_all_fields(self):
        packed = _pack_auth_record(
            {
                "access_token": "a",
                "refresh_token": "r",
                "id_token": "i",
                "account_id": "acct-1",
                "expires_at": 1700000000,
            }
        )
        assert packed == {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
            "account_id": "acct-1",
            "expires_at": "1700000000",
        }

    def test_pack_omits_none(self):
        packed = _pack_auth_record({"access_token": "a", "refresh_token": None})
        assert packed == {"access_token": "a"}

    def test_unpack_coerces_expires_at_to_int(self):
        record = _unpack_auth_record({"access_token": "a", "expires_at": "1700000000"})
        assert record == {"access_token": "a", "expires_at": 1700000000}

    def test_unpack_drops_invalid_expires_at(self):
        record = _unpack_auth_record({"access_token": "a", "expires_at": "nope"})
        assert record == {"access_token": "a"}


class TestDBAuthenticator:
    @pytest.fixture(autouse=True)
    def _reset_credentials(self, monkeypatch):
        original = list(litellm.credential_list)
        monkeypatch.setattr(litellm, "credential_list", [])
        yield
        litellm.credential_list = original

    def test_read_returns_none_when_credential_missing(self):
        auth = DBAuthenticator(credential_name="nope")
        assert auth._read_auth_file() is None

    def test_read_returns_unpacked_record_from_cache(self):
        litellm.credential_list = [
            CredentialItem(
                credential_name="test",
                credential_values={
                    "access_token": "a",
                    "refresh_token": "r",
                    "id_token": "i",
                    "account_id": "acct-1",
                    "expires_at": "1700000000",
                },
                credential_info={"type": CREDENTIAL_TYPE},
            )
        ]
        auth = DBAuthenticator(credential_name="test")
        record = auth._read_auth_file()
        assert record == {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
            "account_id": "acct-1",
            "expires_at": 1700000000,
        }

    def test_write_upserts_cache_and_schedules_db_persist(self):
        auth = DBAuthenticator(credential_name="test")
        with patch(
            "litellm.llms.chatgpt.db_authenticator._schedule_db_persist"
        ) as mock_persist:
            auth._write_auth_file(
                {
                    "access_token": "a",
                    "refresh_token": "r",
                    "id_token": "i",
                    "account_id": "acct-1",
                    "expires_at": 1700000000,
                }
            )
        # Cache was updated
        assert any(
            c.credential_name == "test" and c.credential_values["access_token"] == "a"
            for c in litellm.credential_list
        )
        # DB persist was scheduled
        mock_persist.assert_called_once()
        item = mock_persist.call_args.args[0]
        assert item.credential_name == "test"
        assert item.credential_info == {
            "type": CREDENTIAL_TYPE,
            "custom_llm_provider": "chatgpt",
        }

    def test_write_then_read_roundtrip_via_cache(self):
        auth = DBAuthenticator(credential_name="test")
        with patch("litellm.llms.chatgpt.db_authenticator._schedule_db_persist"):
            auth._write_auth_file(
                {
                    "access_token": "a",
                    "refresh_token": "r",
                    "id_token": "i",
                    "account_id": "acct-1",
                    "expires_at": 1700000000,
                }
            )
        assert auth._read_auth_file() == {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
            "account_id": "acct-1",
            "expires_at": 1700000000,
        }

    def test_ensure_token_dir_is_noop(self, tmp_path, monkeypatch):
        # Parent uses os.makedirs; DBAuthenticator must not touch the filesystem.
        calls = []
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: calls.append((a, kw)))
        monkeypatch.setattr("os.path.exists", lambda p: False)
        auth = DBAuthenticator(credential_name="test")
        auth._ensure_token_dir()
        assert calls == []


class TestPersistCredentialToDb:
    @pytest.mark.asyncio
    async def test_noop_when_prisma_missing(self, monkeypatch):
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "prisma_client", None)
        item = CredentialItem(
            credential_name="test",
            credential_values={"access_token": "a"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        await persist_credential_to_db(item)  # no raise

    @pytest.mark.asyncio
    async def test_upserts_encrypted_values(self, monkeypatch):
        import litellm.proxy.proxy_server as proxy_server

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_credentialstable.upsert = MagicMock(
            return_value=_AsyncNone()
        )
        monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
        monkeypatch.setattr(
            "litellm.proxy.common_utils.encrypt_decrypt_utils.encrypt_value_helper",
            lambda v, key=None: f"enc({v})",
        )

        item = CredentialItem(
            credential_name="test",
            credential_values={"access_token": "a", "refresh_token": "r"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        await persist_credential_to_db(item)

        fake_prisma.db.litellm_credentialstable.upsert.assert_called_once()
        kwargs = fake_prisma.db.litellm_credentialstable.upsert.call_args.kwargs
        assert kwargs["where"] == {"credential_name": "test"}
        create = kwargs["data"]["create"]
        assert create["credential_name"] == "test"
        # Prisma Json columns receive pre-serialized JSON strings.
        assert json.loads(create["credential_values"]) == {
            "access_token": "enc(a)",
            "refresh_token": "enc(r)",
        }
        assert json.loads(create["credential_info"]) == {"type": CREDENTIAL_TYPE}
        update = kwargs["data"]["update"]
        assert update["credential_values"] == create["credential_values"]


class _AsyncNone:
    def __await__(self):
        async def _coro():
            return None

        return _coro().__await__()


class TestPersistScheduling:
    def test_schedule_starts_background_thread(self, monkeypatch):
        from litellm.llms.chatgpt import db_authenticator as mod

        calls = []

        def _fake_sync(item):
            calls.append(item.credential_name)

        monkeypatch.setattr(mod, "_persist_item_sync", _fake_sync)
        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "a"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        mod._schedule_db_persist(item)
        # Thread is daemon; give it a brief moment to run.
        import time

        for _ in range(100):
            if calls:
                break
            time.sleep(0.01)
        assert calls == ["c"]

    def test_persist_item_sync_swallows_exceptions(self, monkeypatch):
        """A DB failure on the worker thread must not crash the process."""
        from litellm.llms.chatgpt import db_authenticator as mod

        async def _boom(item):
            raise RuntimeError("db offline")

        monkeypatch.setattr(mod, "persist_credential_to_db", _boom)
        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "a"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        # Should not raise; just logs.
        mod._persist_item_sync(item)
