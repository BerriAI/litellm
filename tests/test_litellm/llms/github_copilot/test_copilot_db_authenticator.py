import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.github_copilot.db_authenticator import (
    CREDENTIAL_TYPE,
    OAUTH_CREDENTIAL_API_KEY_PREFIX,
    DBAuthenticator,
    persist_credential_to_db,
    resolve_authenticator,
)
from litellm.llms.github_copilot.common_utils import GetAccessTokenError
from litellm.types.utils import CredentialItem


@pytest.fixture(autouse=True)
def _reset_credentials(monkeypatch):
    original = list(litellm.credential_list)
    monkeypatch.setattr(litellm, "credential_list", [])
    DBAuthenticator._api_key_cache.clear()
    yield
    litellm.credential_list = original
    DBAuthenticator._api_key_cache.clear()


class TestDBAuthenticatorAccessToken:
    def test_raises_when_missing(self):
        auth = DBAuthenticator(credential_name="nope")
        with pytest.raises(GetAccessTokenError):
            auth.get_access_token()

    def test_reads_from_cache(self):
        litellm.credential_list = [
            CredentialItem(
                credential_name="c",
                credential_values={"access_token": "gho_abc"},
                credential_info={"type": CREDENTIAL_TYPE},
            )
        ]
        auth = DBAuthenticator(credential_name="c")
        assert auth.get_access_token() == "gho_abc"

    def test_store_access_token_upserts_and_invalidates_cache(self):
        auth = DBAuthenticator(credential_name="c")
        DBAuthenticator._api_key_cache["c"] = {"token": "stale", "expires_at": 9**12}

        with patch(
            "litellm.llms.github_copilot.db_authenticator._schedule_db_persist"
        ) as mock_persist:
            auth.store_access_token("gho_new")

        # Cache contents updated
        assert any(
            c.credential_name == "c"
            and c.credential_values["access_token"] == "gho_new"
            for c in litellm.credential_list
        )
        # Stale API key purged
        assert "c" not in DBAuthenticator._api_key_cache
        # DB persist scheduled
        mock_persist.assert_called_once()


class TestDBAuthenticatorApiKey:
    def test_uses_cached_api_key_when_not_expired(self):
        import time

        DBAuthenticator._api_key_cache["c"] = {
            "token": "cached-key",
            "expires_at": int(time.time()) + 3600,
        }
        auth = DBAuthenticator(credential_name="c")
        assert auth.get_api_key() == "cached-key"

    def test_refreshes_when_cache_stale(self):
        import time

        DBAuthenticator._api_key_cache["c"] = {
            "token": "stale",
            "expires_at": int(time.time()) - 10,
        }
        litellm.credential_list = [
            CredentialItem(
                credential_name="c",
                credential_values={"access_token": "gho_abc"},
                credential_info={"type": CREDENTIAL_TYPE},
            )
        ]
        auth = DBAuthenticator(credential_name="c")
        with patch.object(
            DBAuthenticator,
            "_refresh_api_key",
            return_value={
                "token": "fresh",
                "expires_at": int(time.time()) + 3600,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
        ):
            assert auth.get_api_key() == "fresh"
        assert DBAuthenticator._api_key_cache["c"]["token"] == "fresh"

    def test_force_refresh_ignores_cache(self):
        import time

        DBAuthenticator._api_key_cache["c"] = {
            "token": "cached",
            "expires_at": int(time.time()) + 3600,
        }
        litellm.credential_list = [
            CredentialItem(
                credential_name="c",
                credential_values={"access_token": "gho_abc"},
                credential_info={"type": CREDENTIAL_TYPE},
            )
        ]
        auth = DBAuthenticator(credential_name="c")
        with patch.object(
            DBAuthenticator,
            "_refresh_api_key",
            return_value={"token": "forced", "expires_at": int(time.time()) + 7200},
        ) as mock_refresh:
            info = auth.force_refresh_api_key()
        mock_refresh.assert_called_once()
        assert info["token"] == "forced"
        assert DBAuthenticator._api_key_cache["c"]["token"] == "forced"

    def test_api_base_reads_from_cache(self):
        import time

        DBAuthenticator._api_key_cache["c"] = {
            "token": "tok",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.example.copilot"},
        }
        auth = DBAuthenticator(credential_name="c")
        assert auth.get_api_base() == "https://api.example.copilot"

    def test_api_base_is_none_when_uncached(self):
        auth = DBAuthenticator(credential_name="c")
        assert auth.get_api_base() is None


class TestResolveAuthenticator:
    def test_plain_api_key_returns_fallback(self):
        fallback = MagicMock()
        resolved = resolve_authenticator("sk-plain", None, fallback)
        assert resolved is fallback

    def test_oauth_prefix_returns_db_authenticator(self):
        fallback = MagicMock()
        resolved = resolve_authenticator(
            f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds", None, fallback
        )
        assert isinstance(resolved, DBAuthenticator)
        assert resolved.credential_name == "my-creds"

    def test_checks_litellm_params_api_key_when_top_level_plain(self):
        """
        ``_get_openai_compatible_provider_info`` may rewrite ``api_key`` to the
        resolved key before ``validate_environment`` runs. The raw marker
        should still be picked up from ``litellm_params``.
        """
        fallback = MagicMock()
        resolved = resolve_authenticator(
            "rewritten-copilot-api-key",
            {"api_key": f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds"},
            fallback,
        )
        assert isinstance(resolved, DBAuthenticator)
        assert resolved.credential_name == "my-creds"

    def test_handles_pydantic_litellm_params(self):
        fallback = MagicMock()

        class _FakeParams:
            api_key = f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}from-pydantic"

        resolved = resolve_authenticator(None, _FakeParams(), fallback)
        assert isinstance(resolved, DBAuthenticator)
        assert resolved.credential_name == "from-pydantic"


class TestPersistCredentialToDb:
    @pytest.mark.asyncio
    async def test_noop_when_prisma_missing(self, monkeypatch):
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "prisma_client", None)
        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "a"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        await persist_credential_to_db(item)  # no raise

    @pytest.mark.asyncio
    async def test_upserts_encrypted_values(self, monkeypatch):
        import litellm.proxy.proxy_server as proxy_server

        fake_prisma = MagicMock()

        async def _upsert(**kwargs):
            return None

        fake_prisma.db.litellm_credentialstable.upsert = MagicMock(
            side_effect=lambda **kwargs: _Awaitable(None)
        )
        monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
        monkeypatch.setattr(
            "litellm.proxy.common_utils.encrypt_decrypt_utils.encrypt_value_helper",
            lambda v, key=None: f"enc({v})",
        )

        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "gho_abc"},
            credential_info={
                "type": CREDENTIAL_TYPE,
                "custom_llm_provider": "github_copilot",
            },
        )
        await persist_credential_to_db(item)

        fake_prisma.db.litellm_credentialstable.upsert.assert_called_once()
        kwargs = fake_prisma.db.litellm_credentialstable.upsert.call_args.kwargs
        assert kwargs["where"] == {"credential_name": "c"}
        # Prisma Json columns receive pre-serialized JSON strings.
        assert json.loads(kwargs["data"]["create"]["credential_values"]) == {
            "access_token": "enc(gho_abc)"
        }
        assert (
            json.loads(kwargs["data"]["create"]["credential_info"])["type"]
            == CREDENTIAL_TYPE
        )


class _Awaitable:
    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()


class TestPersistScheduling:
    def test_schedule_starts_background_thread(self, monkeypatch):
        from litellm.llms.github_copilot import db_authenticator as mod

        calls = []

        def _fake_sync(item):
            calls.append(item.credential_name)

        monkeypatch.setattr(mod, "_persist_item_sync", _fake_sync)
        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "gho_abc"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        mod._schedule_db_persist(item)
        import time

        for _ in range(100):
            if calls:
                break
            time.sleep(0.01)
        assert calls == ["c"]

    def test_persist_item_sync_swallows_exceptions(self, monkeypatch):
        from litellm.llms.github_copilot import db_authenticator as mod

        async def _boom(item):
            raise RuntimeError("db offline")

        monkeypatch.setattr(mod, "persist_credential_to_db", _boom)
        item = CredentialItem(
            credential_name="c",
            credential_values={"access_token": "gho_abc"},
            credential_info={"type": CREDENTIAL_TYPE},
        )
        mod._persist_item_sync(item)  # must not raise
