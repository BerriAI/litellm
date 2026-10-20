import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import (
    GenerateKeyRequest,
    GenerateKeyResponse,
    NewUserRequest,
    NewUserResponse,
    UserAPIKeyAuth,
)
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks

_PLAINTEXT_KEY = "sk-plaintext-key-39315"
_TOKEN_ID = "a1b2c3d4e5f6789012345678abcdef01"


def _key_created_email_response() -> dict:
    """Response after generate_key_fn remaps token onto the hashed token_id."""
    return {
        "key": _PLAINTEXT_KEY,
        "token": _TOKEN_ID,
        "token_id": _TOKEN_ID,
        "user_id": "user-123",
        "team_id": "team-1",
        "key_alias": "ci-key",
        "spend": 0.0,
        "max_budget": 10.0,
    }


def _proxy_server_module(slack_alerting: object) -> SimpleNamespace:
    return SimpleNamespace(
        general_settings={"alerting": ["email"]},
        proxy_logging_obj=SimpleNamespace(slack_alerting_instance=slack_alerting),
        premium_user=False,
        prisma_client=None,
        CommonProxyErrors=SimpleNamespace(not_premium_user=SimpleNamespace(value="not premium")),
        litellm_proxy_admin_name="admin-user",
    )


@pytest.mark.asyncio
async def test_v1_user_creation_no_email_when_send_invite_email_false():
    """
    Test that user invitation email is NOT sent when send_invite_email=False
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=False,  # Should NOT send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_v1_user_creation_sends_email_when_send_invite_email_true():
    """
    Test that user invitation email IS sent when send_invite_email=True
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=True,  # Should send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_called_once()


@pytest.mark.asyncio
async def test_v2_invitation_email_suppresses_legacy_duplicate():
    """
    Regression: when a V2 enterprise email logger is registered and sends
    successfully, the modern invitation email is sent and the legacy V1 email is
    NOT also sent, so the invited user does not receive a duplicate.
    """
    pytest.importorskip("litellm_enterprise")
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )

    class RecordingEmailLogger(BaseEmailLogger):
        def __init__(self):
            super().__init__()
            self.sent_events = []

        async def send_user_invitation_email(self, event):
            self.sent_events.append(event)

    recording_logger = RecordingEmailLogger()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch(
        "litellm.logging_callback_manager.get_custom_loggers_for_type",
        return_value=[recording_logger],
    ):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=True,
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )

    assert len(recording_logger.sent_events) == 1
    mock_slack_alerting.send_key_created_or_user_invited_email.assert_not_called()


@pytest.mark.asyncio
async def test_v2_invitation_email_failure_falls_back_to_legacy():
    """
    Regression: when a V2 enterprise email logger is registered but its send
    raises (e.g. misconfigured SMTP), the legacy V1 email still fires as a
    fallback so the invited user is not left with zero emails.
    """
    pytest.importorskip("litellm_enterprise")
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )

    class FailingEmailLogger(BaseEmailLogger):
        def __init__(self):
            super().__init__()

        async def send_user_invitation_email(self, event):
            raise RuntimeError("smtp misconfigured")

    failing_logger = FailingEmailLogger()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch(
        "litellm.logging_callback_manager.get_custom_loggers_for_type",
        return_value=[failing_logger],
    ):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=True,
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )

    mock_slack_alerting.send_key_created_or_user_invited_email.assert_called_once()


@pytest.mark.asyncio
async def test_v1_key_generation_sends_email_when_send_invite_email_true():
    """
    Test that key generation email IS sent when send_invite_email=True
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with (
        patch("litellm.store_audit_logs", False),
        patch.object(KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email),
    ):
        with patch(
            "litellm.logging_callback_manager.get_custom_loggers_for_type",
            return_value=[],
        ):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=True,  # Should send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_called_once()


@pytest.mark.asyncio
async def test_v1_key_generation_no_email_when_send_invite_email_false():
    """
    Test that key generation email is NOT sent when send_invite_email=False
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with (
        patch("litellm.store_audit_logs", False),
        patch.object(KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email),
    ):
        with patch(
            "litellm.logging_callback_manager.get_custom_loggers_for_type",
            return_value=[],
        ):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=False,  # Should NOT send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(user_id="admin-user", api_key="admin-key")
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_not_called()


@pytest.mark.asyncio
async def test_v0_key_created_email_contains_plaintext_key_not_token_id(monkeypatch):
    """The v0 invite email prints webhook token, which must be the sk- key."""
    for name in ("EMAIL_LOGO_URL", "EMAIL_SUPPORT_CONTACT", "SMTP_SENDER_LOGO"):
        monkeypatch.delenv(name, raising=False)

    sent_html: list[str] = []

    async def capture_send_email(*, receiver_email, subject, html):
        sent_html.append(html)

    scheduled: list[asyncio.Task] = []
    real_create_task = asyncio.create_task

    def schedule_task(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        scheduled.append(task)
        return task

    slack_alerting = SlackAlerting(alerting=["email"])
    with (
        patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]),
        patch("litellm.proxy.utils.send_email", capture_send_email),
        patch(
            "litellm.proxy.hooks.key_management_event_hooks.asyncio.create_task",
            side_effect=schedule_task,
        ),
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": _proxy_server_module(slack_alerting)}),
    ):
        await KeyManagementEventHooks._send_key_created_email(_key_created_email_response())
        assert scheduled, "key created email was not scheduled"
        await asyncio.gather(*scheduled)

    assert sent_html, "key created email was not sent"
    assert f"Key: <pre>{_PLAINTEXT_KEY}</pre>" in sent_html[0]
    assert _TOKEN_ID not in sent_html[0]


@pytest.mark.asyncio
async def test_enterprise_key_created_email_contains_plaintext_key_not_token_id(monkeypatch):
    """The enterprise invite email prints virtual_key, which must be the sk- key."""
    pytest.importorskip("litellm_enterprise")
    monkeypatch.setenv("EMAIL_INCLUDE_API_KEY", "true")
    for name in ("EMAIL_LOGO_URL", "EMAIL_SUPPORT_CONTACT", "SMTP_SENDER_LOGO", "EMAIL_SIGNATURE"):
        monkeypatch.delenv(name, raising=False)
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )

    class RecordingEmailLogger(BaseEmailLogger):
        def __init__(self):
            super().__init__()
            self.html_bodies: list[str] = []
            self.events: list = []

        async def send_key_created_email(self, send_key_created_email_event):
            self.events.append(send_key_created_email_event)
            event = send_key_created_email_event
            if event.user_email is None:
                event = event.model_copy(update={"user_email": "user@example.com"})
            await super().send_key_created_email(send_key_created_email_event=event)

        async def send_email(self, from_email, to_email, subject, html_body):
            self.html_bodies.append(html_body)

    recording_logger = RecordingEmailLogger()
    with (
        patch(
            "litellm.logging_callback_manager.get_custom_loggers_for_type",
            return_value=[recording_logger],
        ),
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": _proxy_server_module(None)}),
    ):
        await KeyManagementEventHooks._send_key_created_email(_key_created_email_response())

    assert recording_logger.events, "enterprise key created email was not sent"
    assert recording_logger.events[0].virtual_key == _PLAINTEXT_KEY
    assert recording_logger.events[0].token == _PLAINTEXT_KEY
    assert recording_logger.html_bodies, "enterprise key created email was not sent"
    html = recording_logger.html_bodies[0]
    assert f'<div class="key">{_PLAINTEXT_KEY}</div>' in html
    assert _TOKEN_ID not in html
