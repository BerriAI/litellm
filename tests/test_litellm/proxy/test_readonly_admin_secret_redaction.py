"""
Regression tests: management/read endpoints must not disclose decrypted secrets
to callers who are not a full proxy admin (read-only admins, org admins, plain
keys), and audit logs must not carry usable credentials.
"""

import json
import os
import sys

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import GenerateKeyResponse, LitellmUserRoles, UserAPIKeyAuth


def _user(role: LitellmUserRoles) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=role, api_key="hashed-token", user_id="u1")


# --------------------------------------------------------------------------- #
# Audit log writer / reader redaction
# --------------------------------------------------------------------------- #


def test_audit_values_without_plaintext_key_redacts_usable_key():
    from litellm.proxy.hooks.key_management_event_hooks import (
        _audit_values_without_plaintext_key,
    )

    response = GenerateKeyResponse(key="sk-this-is-a-usable-virtual-key")
    serialized = _audit_values_without_plaintext_key(response)
    data = json.loads(serialized)

    assert data["key"] == "***REDACTED***"
    assert "sk-this-is-a-usable-virtual-key" not in serialized


def test_redact_audit_log_values_masks_provider_secrets_dict_input():
    """Reader masks the secrets the write-time key-only masker misses, and keeps
    dict input as a dict (response shape is unchanged)."""
    from litellm_enterprise.proxy.audit_logging_endpoints import (
        _redact_audit_log_values,
    )

    row = {
        "id": "a1",
        "updated_values": {
            "key": "sk-plaintext-virtual-key",
            "api_key": "sk-provider-plaintext-key",
            "client_secret": "client-secret-plaintext",
            "vertex_credentials": "vertex-credentials-plaintext",
            "token": "tok-plaintext",
            "user_id": "user-123",
            "model_name": "gpt-4",
        },
        "before_value": None,
    }

    out = _redact_audit_log_values(row)
    masked = out["updated_values"]

    assert isinstance(masked, dict)  # dict in -> dict out
    blob = json.dumps(masked)
    for secret in (
        "sk-plaintext-virtual-key",
        "sk-provider-plaintext-key",
        "client-secret-plaintext",
        "vertex-credentials-plaintext",
        "tok-plaintext",
    ):
        assert secret not in blob, f"{secret} leaked through /audit"
    # Non-secret operational fields stay readable.
    assert masked["user_id"] == "user-123"
    assert masked["model_name"] == "gpt-4"


def test_redact_audit_log_values_handles_string_and_non_dict_input():
    from litellm_enterprise.proxy.audit_logging_endpoints import (
        _redact_audit_log_values,
    )

    # JSON string input -> masked JSON string output (type preserved).
    row = {"before_value": json.dumps({"client_secret": "should-be-masked"})}
    out = _redact_audit_log_values(row)
    assert isinstance(out["before_value"], str)
    assert "should-be-masked" not in out["before_value"]

    # Non-dict / unparseable / None values must pass through untouched.
    assert _redact_audit_log_values({"before_value": None})["before_value"] is None
    assert (
        _redact_audit_log_values({"updated_values": "not-json"})["updated_values"]
        == "not-json"
    )


def test_redact_audit_log_values_masks_short_secrets():
    """A short secret in an audit snapshot must still be masked, not returned
    verbatim because it falls under the masker's partial-reveal length."""
    from litellm_enterprise.proxy.audit_logging_endpoints import (
        _redact_audit_log_values,
    )

    out = _redact_audit_log_values({"updated_values": {"api_key": "sk12"}})
    assert out["updated_values"]["api_key"] != "sk12"
    assert set(out["updated_values"]["api_key"]) == {"*"}


# --------------------------------------------------------------------------- #
# Pass-through endpoint header masking
# --------------------------------------------------------------------------- #


def _make_endpoint(
    headers=None,
    default_query_params=None,
    target="https://upstream.example.com",
):
    from litellm.proxy._types import PassThroughGenericEndpoint

    return PassThroughGenericEndpoint(
        path="/foo",
        target=target,
        headers=headers or {},
        default_query_params=default_query_params or {},
    )


def test_mask_pass_through_endpoint_secrets_redacts_all_credential_fields():
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _mask_pass_through_endpoint_secrets,
    )

    target = (
        "https://user:targetpw@upstream.example.com/v1?subscription-key=tgt-qp-secret"
    )
    ep = _make_endpoint(
        headers={"Authorization": "Bearer sk-upstream-secret-token"},
        default_query_params={"api_key": "qp-secret-key-value"},
        target=target,
    )
    masked = _mask_pass_through_endpoint_secrets(ep)

    # Header value masked, name preserved.
    assert "Authorization" in masked.headers
    assert "sk-upstream-secret-token" not in masked.headers["Authorization"]
    # default_query_params value masked, name preserved.
    assert "api_key" in masked.default_query_params
    assert "qp-secret-key-value" not in masked.default_query_params["api_key"]
    # target URL userinfo password AND query-param secret both redacted.
    assert "targetpw" not in masked.target
    assert "tgt-qp-secret" not in masked.target
    assert "subscription-key" in masked.target  # param name preserved
    # Original object is not mutated.
    assert ep.headers["Authorization"] == "Bearer sk-upstream-secret-token"
    assert ep.default_query_params["api_key"] == "qp-secret-key-value"
    assert ep.target == target


@pytest.mark.asyncio
async def test_pass_through_get_masks_secrets_for_non_admin_only():
    import litellm.proxy.pass_through_endpoints.pass_through_endpoints as pt

    header_secret = "Bearer sk-upstream-secret-token"
    qp_secret = "qp-secret-key-value"
    endpoints = [
        _make_endpoint(
            headers={"Authorization": header_secret},
            default_query_params={"api_key": qp_secret},
        )
    ]

    with (
        patch.object(
            pt, "_get_pass_through_endpoints_from_db", AsyncMock(return_value=endpoints)
        ),
        patch.object(
            pt, "_get_pass_through_endpoints_from_config", MagicMock(return_value=[])
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        viewer_resp = await pt.get_pass_through_endpoints(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        )
        admin_resp = await pt.get_pass_through_endpoints(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN)
        )

    viewer_blob = json.dumps([e.model_dump() for e in viewer_resp.endpoints])
    assert header_secret not in viewer_blob
    assert qp_secret not in viewer_blob
    # Full admin still reads plaintext (the endpoint is editable for them).
    assert admin_resp.endpoints[0].headers["Authorization"] == header_secret
    assert admin_resp.endpoints[0].default_query_params["api_key"] == qp_secret


# --------------------------------------------------------------------------- #
# /config/field/info must not return credential general-settings fields
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_name",
    [
        "master_key",
        "database_url",
        "alert_to_webhook_url",
        "pass_through_endpoints",
        "database_args",
    ],
)
async def test_config_field_info_blocks_secret_fields(field_name):
    from litellm.proxy.proxy_server import get_config_general_settings

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        with pytest.raises(HTTPException) as exc:
            await get_config_general_settings(
                field_name=field_name,
                user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
            )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_config_field_info_allows_non_secret_field():
    from litellm.proxy.proxy_server import get_config_general_settings

    db_row = MagicMock()
    db_row.param_value = {"completion_model": "gpt-4"}
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=db_row)

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_config_general_settings(
            field_name="completion_model",
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        )
    assert result.field_value == "gpt-4"


# --------------------------------------------------------------------------- #
# /memory-usage-in-mem-cache-items must require a full proxy admin
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER],
)
async def test_memory_usage_endpoint_blocks_non_admins(role):
    from litellm.proxy.common_utils.debug_utils import memory_usage_in_mem_cache_items

    with pytest.raises(HTTPException) as exc:
        await memory_usage_in_mem_cache_items(user_api_key_dict=_user(role))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_memory_usage_endpoint_allows_proxy_admin():
    from litellm.proxy.common_utils.debug_utils import memory_usage_in_mem_cache_items

    with (
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
    ):
        result = await memory_usage_in_mem_cache_items(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN)
        )
    assert "user_api_key_cache" in result


# --------------------------------------------------------------------------- #
# /cache/settings must redact a password embedded in a connection URL
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cache_settings_masks_url_embedded_password():
    from litellm.proxy.management_endpoints.cache_settings_endpoints import (
        get_cache_settings,
    )

    stored = {"type": "redis", "url": "redis://:redisPlaintextPw@cache.internal:6379"}
    mock_cache_config = MagicMock()
    mock_cache_config.cache_settings = json.dumps(stored)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(
        return_value=mock_cache_config
    )
    mock_proxy_config = MagicMock()
    mock_proxy_config._decrypt_db_variables = MagicMock(return_value=dict(stored))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config),
    ):
        result = await get_cache_settings(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        )

    assert "redisPlaintextPw" not in json.dumps(result.current_values)
    assert result.current_values["url"] == "redis://:****@cache.internal:6379"


# --------------------------------------------------------------------------- #
# /get/config/callbacks masks callback secrets for non-admins only
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_config_callbacks_masks_secret_for_view_only_admin():
    import litellm.proxy.proxy_server as ps

    config = {
        "litellm_settings": {"success_callback": ["langfuse"]},
        "general_settings": {},
        "environment_variables": {
            "LANGFUSE_SECRET_KEY": "sk-langfuse-supersecret-value",
            "LANGFUSE_HOST": "https://cloud.langfuse.com",
        },
    }
    mock_proxy_config = MagicMock()
    mock_proxy_config.get_config = AsyncMock(return_value=config)

    with (
        patch.object(ps, "proxy_config", mock_proxy_config),
        patch.object(ps, "llm_router", None),
        patch(
            "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
            return_value=["LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
        ),
    ):
        viewer = await ps.get_config(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        )
        admin = await ps.get_config(
            user_api_key_dict=_user(LitellmUserRoles.PROXY_ADMIN)
        )

    viewer_secret = viewer["callbacks"][0]["variables"]["LANGFUSE_SECRET_KEY"]
    admin_secret = admin["callbacks"][0]["variables"]["LANGFUSE_SECRET_KEY"]

    assert "supersecret" not in viewer_secret
    # Full admin keeps plaintext (they can edit it; round-trip stays correct).
    assert admin_secret == "sk-langfuse-supersecret-value"
    # Non-secret routing var is never masked.
    assert (
        viewer["callbacks"][0]["variables"]["LANGFUSE_HOST"]
        == "https://cloud.langfuse.com"
    )
