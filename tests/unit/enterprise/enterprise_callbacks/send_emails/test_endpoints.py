import asyncio
import json
from typing import Final
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from litellm_enterprise.enterprise_callbacks.send_emails.endpoints import (
    _save_email_settings,
    get_email_event_settings,
    get_email_settings,
    reset_event_settings,
    router,
    update_event_settings,
)
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    DefaultEmailSettings,
    EmailEvent,
    EmailEventSettings,
    EmailEventSettingsUpdateRequest,
)

from litellm._service_logger import ServiceTypes
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from tests.unit.proxy.db.fake_prisma_engine import engine_call


# Mock user_api_key_auth dependency
@pytest.fixture
def mock_user_api_key_auth():
    return UserAPIKeyAuth(user_id="test_user", user_role=LitellmUserRoles.PROXY_ADMIN)


# Mock prisma client
@pytest.fixture
def mock_prisma_client():
    mock_client = mock.MagicMock()

    # Setup mock for async methods to work properly
    mock_db = mock.MagicMock()
    mock_config = mock.MagicMock()

    # Make find_unique return a coroutine mock
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_config.find_unique = mock_find_unique

    # Make upsert return a coroutine mock
    async def mock_upsert(*args, **kwargs):
        return None

    mock_config.upsert = mock_upsert

    mock_db.litellm_config = mock_config
    mock_client.db = mock_db
    mock_client.writer_db = mock_db
    mock_client.tx.return_value.__aenter__.return_value = mock_db
    mock_db.query_raw = mock.AsyncMock()

    return mock_client


# Test _get_email_settings helper function
@pytest.mark.asyncio
async def test_get_email_settings_empty_db(mock_prisma_client):
    """Test that default settings are returned when database has no email settings."""

    # Setup mock find_unique to return None
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    # Call the function
    result = await get_email_settings(mock_prisma_client)

    # Assert that default settings are returned
    assert result == DefaultEmailSettings.get_defaults()


@pytest.mark.asyncio
async def test_get_email_settings_with_existing_settings(mock_prisma_client):
    """Test that existing email settings are correctly retrieved from the database."""
    # Setup mock find_unique to return existing settings
    mock_settings = {
        "email_settings": {
            EmailEvent.virtual_key_created.value: True,
            EmailEvent.new_user_invitation.value: False,
        }
    }

    mock_entry = mock.MagicMock()
    mock_entry.param_value = json.dumps(mock_settings)

    async def mock_find_unique(*args, **kwargs):
        return mock_entry

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    # Call the function
    result = await get_email_settings(mock_prisma_client)

    # Assert correct settings are returned
    assert result[EmailEvent.virtual_key_created.value] is True
    assert result[EmailEvent.new_user_invitation.value] is False


@pytest.mark.asyncio
async def test_save_email_settings_new_entry(mock_prisma_client):
    """Test that email settings are properly saved to database when no previous settings exist."""

    # Setup mock find_unique to return None
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    upsert = mock.AsyncMock()
    mock_prisma_client.db.litellm_config.upsert = upsert
    settings = {EmailEvent.virtual_key_created.value: True, EmailEvent.new_user_invitation.value: False}

    await _save_email_settings(mock_prisma_client, settings)

    assert json.loads(upsert.await_args.kwargs["data"]["create"]["param_value"]) == {"email_settings": settings}


# Test the GET endpoint
@pytest.mark.asyncio
async def test_get_email_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the GET endpoint returns the correct email event settings."""

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        response = await get_email_event_settings(user_api_key_dict=mock_user_api_key_auth)

    assert {setting.event: setting.enabled for setting in response.settings} == DefaultEmailSettings().settings


# Test the PATCH endpoint
@pytest.mark.asyncio
async def test_update_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the PATCH endpoint correctly updates email event settings."""

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        request = EmailEventSettingsUpdateRequest(
            settings=[EmailEventSettings(event=EmailEvent.virtual_key_created, enabled=False)]
        )
        response = await update_event_settings(request=request, user_api_key_dict=mock_user_api_key_auth)

    assert response["message"] == "Email event settings updated successfully"


# Test the reset endpoint
@pytest.mark.asyncio
async def test_reset_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the reset endpoint correctly restores default email event settings."""

    # Mock _save_email_settings to do nothing
    async def mock_save_settings(*args, **kwargs):
        return None

    # Setup mocks
    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with mock.patch(
            "litellm_enterprise.enterprise_callbacks.send_emails.endpoints._save_email_settings",
            side_effect=mock_save_settings,
        ):
            # Call the endpoint function directly
            response = await reset_event_settings(user_api_key_dict=mock_user_api_key_auth)

            # Assert response is success
            assert response["message"] == "Email event settings reset to defaults"


# Test handling of prisma client None
@pytest.mark.asyncio
async def test_endpoint_with_no_prisma_client(mock_user_api_key_auth):
    """Test that all endpoints properly handle the case when the database is not connected."""
    # Setup mock to return None for prisma_client
    with mock.patch("litellm.proxy.proxy_server.prisma_client", None):
        # Test get endpoint
        with pytest.raises(HTTPException) as exc_info:
            await get_email_event_settings(user_api_key_dict=mock_user_api_key_auth)
        assert exc_info.value.status_code == 500
        assert "Database not connected" in exc_info.value.detail

        # Test update endpoint
        request = EmailEventSettingsUpdateRequest(settings=[])
        with pytest.raises(HTTPException) as exc_info:
            await update_event_settings(request=request, user_api_key_dict=mock_user_api_key_auth)
        assert exc_info.value.status_code == 500

        # Test reset endpoint
        with pytest.raises(HTTPException) as exc_info:
            await reset_event_settings(user_api_key_dict=mock_user_api_key_auth)
        assert exc_info.value.status_code == 500


def _prisma_recording_upserts(upserts):
    client = mock.MagicMock()

    async def find_unique(*args, **kwargs):
        return None

    async def upsert(*args, **kwargs):
        upserts.append(kwargs)

    client.db.litellm_config.find_unique = find_unique
    client.db.litellm_config.upsert = upsert
    client.tx.return_value.__aenter__.return_value = client.db
    client.db.query_raw = mock.AsyncMock()
    return client


def _proxy_config_owning(general_settings):
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    proxy_config._load_yaml_settings_stores({"general_settings": general_settings})
    return proxy_config


@pytest.mark.asyncio
async def test_save_email_settings_refuses_a_config_owned_email_settings():
    upserts = []
    client = _prisma_recording_upserts(upserts)
    proxy_config = _proxy_config_owning({"email_settings": {EmailEvent.new_user_invitation.value: True}})

    with mock.patch(
        "litellm.proxy.proxy_server.proxy_config", proxy_config
    ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
        with pytest.raises(HTTPException) as refused:
            await _save_email_settings(client, {EmailEvent.new_user_invitation.value: False})

    assert refused.value.status_code == 400
    assert refused.value.detail["keys"] == ["email_settings"]
    assert upserts == []


@pytest.mark.asyncio
async def test_update_event_settings_surfaces_the_config_owned_refusal(mock_user_api_key_auth):
    upserts = []
    client = _prisma_recording_upserts(upserts)
    proxy_config = _proxy_config_owning({"email_settings": {EmailEvent.virtual_key_created.value: False}})
    request = EmailEventSettingsUpdateRequest(
        settings=[EmailEventSettings(event=EmailEvent.virtual_key_created, enabled=True)]
    )

    with mock.patch(
        "litellm.proxy.proxy_server.prisma_client", client
    ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
        with mock.patch(
            "litellm.proxy.proxy_server.proxy_config", proxy_config
        ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
            with pytest.raises(HTTPException) as refused:
                await update_event_settings(request=request, user_api_key_dict=mock_user_api_key_auth)

    assert refused.value.status_code == 400
    assert refused.value.detail["keys"] == ["email_settings"]
    assert upserts == []


@pytest.mark.asyncio
async def test_save_email_settings_still_writes_when_the_config_file_is_silent():
    upserts = []
    client = _prisma_recording_upserts(upserts)
    proxy_config = _proxy_config_owning({})

    with mock.patch(
        "litellm.proxy.proxy_server.proxy_config", proxy_config
    ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
        await _save_email_settings(client, {EmailEvent.new_user_invitation.value: False})

    assert len(upserts) == 1
    written = json.loads(upserts[0]["data"]["create"]["param_value"])
    assert written["email_settings"] == {EmailEvent.new_user_invitation.value: False}


@pytest.mark.asyncio
async def test_reset_event_settings_surfaces_the_config_owned_refusal(mock_user_api_key_auth):
    upserts = []
    client = _prisma_recording_upserts(upserts)
    proxy_config = _proxy_config_owning({"email_settings": {EmailEvent.new_user_invitation.value: True}})

    with mock.patch(
        "litellm.proxy.proxy_server.prisma_client", client
    ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
        with mock.patch(
            "litellm.proxy.proxy_server.proxy_config", proxy_config
        ):  # test-quality-ok: the endpoint reads these proxy_server module globals at call time; there is no injection seam
            with pytest.raises(HTTPException) as refused:
                await reset_event_settings(user_api_key_dict=mock_user_api_key_auth)

    assert refused.value.status_code == 400
    assert refused.value.detail["keys"] == ["email_settings"]
    assert upserts == []


@pytest.mark.asyncio
async def test_save_email_settings_emits_a_postgres_upsert_event_for_litellm_config(mock_prisma_client):
    mock_prisma_client.db.litellm_config.upsert = engine_call()
    success = mock.AsyncMock()
    service_logging = mock.MagicMock(async_service_success_hook=success, async_service_failure_hook=mock.AsyncMock())

    with mock.patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock.MagicMock(service_logging_obj=service_logging)
    ):
        await _save_email_settings(mock_prisma_client, {"send_key_created_email": True})
        await asyncio.sleep(0)

    event = success.await_args.kwargs
    assert (event["service"], event["call_type"], event["event_metadata"]) == (
        ServiceTypes.DB,
        "save_email_settings",
        {"table_name": "LiteLLM_Config"},
    )


def test_reminder_recipient_can_be_updated_and_cleared(
    mock_prisma_client: mock.MagicMock, mock_user_api_key_auth: UserAPIKeyAuth
) -> None:
    from fastapi import FastAPI

    from litellm.proxy import proxy_server

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_api_key_auth
    table: Final = mock_prisma_client.db.litellm_config
    table.find_unique = mock.AsyncMock(return_value=None)

    async def upsert(*, where: dict[str, str], data: dict[str, dict[str, str]]) -> None:
        table.find_unique.return_value = mock.Mock(param_value=data["update"]["param_value"])

    table.upsert = upsert
    endpoint: Final = "/email/event_settings"
    field: Final = "model_deprecation_email_recipient"
    event: Final = {"event": "Model Deprecations", "enabled": True}

    with mock.patch.object(proxy_server, "prisma_client", mock_prisma_client), TestClient(app) as client:
        assert client.get(endpoint).json()[field] is None
        assert client.patch(endpoint, json={"settings": [event], field: "old@example.com"}).status_code == 200
        assert client.get(endpoint).json()[field] == "old@example.com"
        assert client.patch(endpoint, json={"settings": [], field: "invalid"}).status_code == 422
        assert client.get(endpoint).json()[field] == "old@example.com"
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
        assert client.patch(endpoint, json={"settings": [], field: "new@example.com"}).status_code == 403
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_api_key_auth
        assert client.patch(endpoint, json={"settings": [], field: "new@example.com"}).status_code == 200
        assert client.patch(endpoint, json={"settings": [event]}).status_code == 200
        assert client.get(endpoint).json()[field] == "new@example.com"
        assert client.patch(endpoint, json={"settings": [], field: None}).status_code == 200
        response: Final = client.get(endpoint).json()
        assert response[field] is None
        assert event in response["settings"]
        assert client.patch(endpoint, json={"settings": [], field: "new@example.com"}).status_code == 200
        assert client.post(f"{endpoint}/reset").status_code == 200
        assert client.get(endpoint).json()[field] is None
