import sys
import types
from hashlib import sha256
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.auth.ldap_auth as ldap_auth
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.ldap_auth import (
    LDAPConfig,
    LDAPDirectoryUser,
    _authenticate_ldap_credentials,
    _sync_ldap_user,
    load_ldap_config,
)


@pytest.mark.asyncio
async def test_load_ldap_config_decrypts_db_values_without_setting_env(monkeypatch):
    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.delenv("ldap_use_ssl", raising=False)
    mock_prisma = MagicMock()
    mock_record = MagicMock()
    mock_record.param_value = {
        "ldap_enabled": True,
        "ldap_url": "ldaps://ldap.example.com:636",
        "ldap_base_dn": "dc=example,dc=com",
        "ldap_use_ssl": True,
    }
    mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=mock_record)

    def fail_if_setting_env(environment_variables):
        raise AssertionError("LDAP settings must not be written to os.environ")

    monkeypatch.setattr(proxy_config, "_decrypt_and_set_db_env_variables", fail_if_setting_env)
    monkeypatch.setattr(proxy_config, "_decrypt_db_variables", lambda variables_dict: variables_dict)

    config = await load_ldap_config(mock_prisma)

    assert config.ldap_enabled is True
    assert config.ldap_url == "ldaps://ldap.example.com:636"
    assert config.ldap_base_dn == "dc=example,dc=com"
    assert config.ldap_use_ssl is True


@pytest.mark.asyncio
async def test_sync_ldap_user_serializes_metadata_for_prisma():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.upsert = AsyncMock(
        return_value={
            "user_id": "ldap:alice@example.com",
            "user_email": "alice@example.com",
            "user_role": LitellmUserRoles.INTERNAL_USER,
            "user_alias": "Alice",
            "metadata": {"auth_provider": "ldap", "ldap_dn": "uid=alice,dc=example,dc=com"},
        }
    )

    await _sync_ldap_user(
        mock_prisma,
        LDAPDirectoryUser(
            username="alice",
            dn="uid=alice,dc=example,dc=com",
            email="alice@example.com",
            display_name="Alice",
        ),
    )

    data = mock_prisma.db.litellm_usertable.upsert.call_args.kwargs["data"]
    metadata_filter = mock_prisma.db.litellm_usertable.find_first.call_args.kwargs["where"]
    principal_hash = sha256(b"uid=alice,dc=example,dc=com").hexdigest()
    expected_metadata = (
        '{"auth_provider": "ldap", "ldap_dn": "uid=alice,dc=example,dc=com", '
        f'"ldap_principal_hash": "{principal_hash}"}}'
    )
    assert data["create"]["metadata"] == expected_metadata
    assert data["update"]["metadata"] == expected_metadata
    assert type(metadata_filter["OR"][0]["metadata"]["equals"]).__name__ == "Json"
    assert type(metadata_filter["OR"][1]["metadata"]["equals"]).__name__ == "Json"
    assert data["create"]["user_role"] == "internal_user"
    assert "user_role" not in data["update"]


@pytest.mark.asyncio
async def test_sync_ldap_user_updates_role_when_admin_group_configured():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.upsert = AsyncMock(
        return_value={
            "user_id": "ldap:alice@example.com",
            "user_email": "alice@example.com",
            "user_role": LitellmUserRoles.PROXY_ADMIN,
            "user_alias": "Alice",
            "metadata": {"auth_provider": "ldap", "ldap_dn": "uid=alice,dc=example,dc=com"},
        }
    )

    await _sync_ldap_user(
        mock_prisma,
        LDAPDirectoryUser(
            username="alice",
            dn="uid=alice,dc=example,dc=com",
            email="alice@example.com",
            display_name="Alice",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        ),
        sync_user_role=True,
    )

    data = mock_prisma.db.litellm_usertable.upsert.call_args.kwargs["data"]
    assert data["create"]["user_role"] == LitellmUserRoles.PROXY_ADMIN
    assert data["update"]["user_role"] == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ldap_admin_group_dn,expected_sync_user_role",
    [
        (None, False),
        ("cn=litellm-admins,ou=Groups,dc=example,dc=com", True),
    ],
)
async def test_authenticate_ldap_user_syncs_role_only_when_admin_group_configured(
    monkeypatch,
    ldap_admin_group_dn,
    expected_sync_user_role,
):
    mock_prisma = MagicMock()
    directory_user = LDAPDirectoryUser(
        username="alice",
        dn="uid=alice,dc=example,dc=com",
        email="alice@example.com",
        display_name="Alice",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    synced_user = MagicMock()

    monkeypatch.setattr(
        ldap_auth,
        "load_ldap_config",
        AsyncMock(
            return_value=LDAPConfig(
                ldap_enabled=True,
                ldap_url="ldaps://ldap.example.com:636",
                ldap_base_dn="dc=example,dc=com",
                ldap_admin_group_dn=ldap_admin_group_dn,
            )
        ),
    )
    monkeypatch.setattr(ldap_auth, "_authenticate_ldap_credentials", MagicMock(return_value=directory_user))
    sync_mock = AsyncMock(return_value=synced_user)
    monkeypatch.setattr(ldap_auth, "_sync_ldap_user", sync_mock)

    result = await ldap_auth.authenticate_ldap_user(
        username="alice",
        password="ldap-password",
        prisma_client=mock_prisma,
    )

    assert result == synced_user
    sync_mock.assert_awaited_once_with(
        prisma_client=mock_prisma,
        directory_user=directory_user,
        sync_user_role=expected_sync_user_role,
    )


@pytest.mark.asyncio
async def test_sync_ldap_user_parses_prisma_json_string_metadata():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.upsert = AsyncMock(
        return_value={
            "user_id": "ldap:alice@example.com",
            "user_email": "alice@example.com",
            "user_role": LitellmUserRoles.INTERNAL_USER,
            "user_alias": "Alice",
            "metadata": '{"auth_provider": "ldap", "ldap_dn": "uid=alice,dc=example,dc=com"}',
        }
    )

    user = await _sync_ldap_user(
        mock_prisma,
        LDAPDirectoryUser(
            username="alice",
            dn="uid=alice,dc=example,dc=com",
            email="alice@example.com",
            display_name="Alice",
        ),
    )

    assert user.metadata == {"auth_provider": "ldap", "ldap_dn": "uid=alice,dc=example,dc=com"}


@pytest.mark.asyncio
async def test_sync_ldap_user_preserves_legacy_user_id_after_email_change():
    mock_prisma = MagicMock()
    legacy_user = {
        "user_id": "ldap:old@example.com",
        "user_email": "old@example.com",
        "user_role": LitellmUserRoles.INTERNAL_USER,
        "user_alias": "Alice",
        "metadata": {
            "auth_provider": "ldap",
            "ldap_dn": "uid=alice,dc=example,dc=com",
        },
    }
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(side_effect=[None, None])
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=legacy_user)
    mock_prisma.db.litellm_usertable.upsert = AsyncMock(
        return_value={
            **legacy_user,
            "user_email": "new@example.com",
        }
    )

    await _sync_ldap_user(
        mock_prisma,
        LDAPDirectoryUser(
            username="alice",
            dn="uid=alice,dc=example,dc=com",
            principal_id="directory-object-123",
            email="new@example.com",
            display_name="Alice",
        ),
    )

    where = mock_prisma.db.litellm_usertable.upsert.call_args.kwargs["where"]
    data = mock_prisma.db.litellm_usertable.upsert.call_args.kwargs["data"]
    assert where == {"user_id": "ldap:old@example.com"}
    assert data["update"]["user_email"] == "new@example.com"
    assert sha256(b"directory-object-123").hexdigest() in data["update"]["metadata"]


def test_ldap_directory_user_id_is_based_on_stable_principal_not_email():
    directory_user = LDAPDirectoryUser(
        username="alice",
        dn="uid=alice,dc=example,dc=com",
        principal_id="Directory-Object-123",
        email="alice@example.com",
        display_name="Alice",
    )

    expected = sha256(b"directory-object-123").hexdigest()
    assert directory_user.user_id == f"ldap:{expected}"


def test_authenticate_ldap_credentials_escapes_username_and_maps_admin_group(monkeypatch):
    captured = {}

    class FakeServer:
        def __init__(self, url, get_info=None, use_ssl=False):
            captured["server"] = {"url": url, "get_info": get_info, "use_ssl": use_ssl}

    class FakeAttribute:
        def __init__(self, values):
            self.values = values
            self.value = values[0] if values else None

    class FakeEntry:
        entry_dn = "uid=alice,ou=People,dc=example,dc=com"
        mail = FakeAttribute(["alice@example.com"])
        displayName = FakeAttribute(["Alice"])
        memberOf = FakeAttribute(["cn=litellm-admins,ou=Groups,dc=example,dc=com"])

    class FakeConnection:
        def __init__(self, server, user=None, password=None, auto_bind=False):
            self.server = server
            self.user = user
            self.password = password
            self.entries = []

        def start_tls(self):
            return True

        def bind(self):
            if self.user == "uid=alice,ou=People,dc=example,dc=com":
                return self.password == "ldap-password"
            return True

        def search(self, search_base, search_filter, search_scope, attributes, size_limit):
            captured["search"] = {
                "search_base": search_base,
                "search_filter": search_filter,
                "search_scope": search_scope,
                "attributes": attributes,
                "size_limit": size_limit,
            }
            self.entries = [FakeEntry()]
            return True

        def unbind(self):
            return True

    ldap3_module = types.ModuleType("ldap3")
    ldap3_module.NONE = "NONE"
    ldap3_module.SUBTREE = "SUBTREE"
    ldap3_module.Server = FakeServer
    ldap3_module.Connection = FakeConnection

    ldap3_utils_module = types.ModuleType("ldap3.utils")
    ldap3_conv_module = types.ModuleType("ldap3.utils.conv")
    ldap3_conv_module.escape_filter_chars = lambda value: value.replace("*", "\\2a").replace("(", "\\28")

    monkeypatch.setitem(sys.modules, "ldap3", ldap3_module)
    monkeypatch.setitem(sys.modules, "ldap3.utils", ldap3_utils_module)
    monkeypatch.setitem(sys.modules, "ldap3.utils.conv", ldap3_conv_module)

    config = LDAPConfig(
        ldap_enabled=True,
        ldap_url="ldaps://ldap.example.com:636",
        ldap_base_dn="dc=example,dc=com",
        ldap_search_base="ou=People,dc=example,dc=com",
        ldap_bind_dn="cn=admin,dc=example,dc=com",
        ldap_bind_password="bind-password",
        ldap_user_search_filter="(uid={username})",
        ldap_admin_group_dn="cn=litellm-admins,ou=Groups,dc=example,dc=com",
        ldap_start_tls=True,
    )

    result = _authenticate_ldap_credentials(config, "ali*(ce", "ldap-password")

    assert result is not None
    expected_user_id = sha256(b"uid=alice,ou=people,dc=example,dc=com").hexdigest()
    assert result.user_id == f"ldap:{expected_user_id}"
    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
    assert captured["server"]["url"] == "ldaps://ldap.example.com:636"
    assert captured["search"]["search_base"] == "ou=People,dc=example,dc=com"
    assert captured["search"]["search_filter"] == r"(uid=ali\2a\28ce)"
    assert set(captured["search"]["attributes"]) == {"mail", "displayName", "memberOf"}


def test_authenticate_ldap_credentials_rejects_plaintext_bind_by_default():
    config = LDAPConfig(
        ldap_enabled=True,
        ldap_url="ldap://ldap.example.com:389",
        ldap_base_dn="dc=example,dc=com",
    )

    with pytest.raises(ProxyException) as exc_info:
        _authenticate_ldap_credentials(config, "alice", "ldap-password")

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "ldap_allow_insecure"
